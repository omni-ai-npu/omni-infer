// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

// mock_ox.cpp - A mock version of ox that skips all TCP/P-side communication.
// Requests received via ZMQ ROUTER are immediately completed (success=true)
// and the response is sent back, to stress-test the ZMQ coroutine layer.

#include <set>
#include <atomic>
#include <mutex>
#include <exception>
#include <msgpack.hpp>
#include <shared_mutex>
#include <utility>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>

#include <ox_config.hpp>
#include <zmq_coroutine.hpp>
#include <ox_log.hpp>
#include <ox_metrics.hpp>
// NOTE: ox_block_table.hpp is intentionally NOT included - mock_ox does not
//       need a BlockTable since TCP data transfer is removed. We define the
//       type aliases here instead.

#include <boost/asio/experimental/concurrent_channel.hpp>

namespace asio = boost::asio;
using asio::co_spawn;
using asio::detached;
using asio::use_awaitable;
using asio::awaitable;
using asio::experimental::concurrent_channel;
using asio::ip::tcp;

// ---------------------------------------------------------------------------
// Type aliases (normally from ox_block_table.hpp, defined here to avoid
// pulling in the full BlockTable dependency)
// ---------------------------------------------------------------------------
using request_id_t = std::string;
using client_id_t = std::string;
using block_id_t = int64_t;
using block_list_t = std::vector<block_id_t>;
using table_id_t = int64_t;

// ---------------------------------------------------------------------------
// Message types (same as real ox)
// ---------------------------------------------------------------------------
struct RequestMessage {
    request_id_t request_id;
    table_id_t table_id;
    block_list_t src_block_ids;
    block_list_t dst_block_ids;
    std::string remote_ox_shard_list;
    int src_dp_rank{0};
    MSGPACK_DEFINE_MAP(request_id, table_id, src_block_ids, dst_block_ids, remote_ox_shard_list, src_dp_rank)
};

struct ResponseMessage {
    request_id_t request_id;
    bool success;
    MSGPACK_DEFINE_MAP(request_id, success)
};

// ---------------------------------------------------------------------------
// Channel types
// ---------------------------------------------------------------------------
using ResponseTask = std::tuple<client_id_t, request_id_t, bool>;
using ZMQChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ResponseTask)>;

using ConnectionMessage = std::tuple<request_id_t, table_id_t, block_list_t, block_list_t, int>;
using ConnectionChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ConnectionMessage)>;

using ShardMessage = std::tuple<std::string, block_list_t>;
using ShardChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ShardMessage)>;

using GroupMessage = std::tuple<request_id_t, int, bool>;  // (request_id, rank, success)
using GroupChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, GroupMessage)>;

// Forward declarations
class MockTPShard;

// ---------------------------------------------------------------------------
// MockCoroutineConnection - skips TCP, immediately reports completion
// ---------------------------------------------------------------------------
class MockCoroutineConnection {
public:
    MockCoroutineConnection(asio::io_context &io_context, Config &config, int rank,
        ShardChannel &response)
        : config(config), rank(rank),
          request(io_context, 128), upstream(response)
    {}

    void start()
    {
        // No TCP connection needed - just start the run loop
        asio::co_spawn(config.get_io_context(), run(), asio::detached);
    }

    asio::awaitable<void> run()
    {
        while (true) {
            auto [request_id, table_id, src_ids, dst_ids, src_dp_rank] =
                co_await request.async_receive(asio::use_awaitable);

            if (src_ids.empty()) {
                co_await upstream.async_send(
                    boost::system::error_code{},
                    std::make_tuple(request_id, dst_ids),
                    asio::use_awaitable);
                continue;
            }

            // Apply DP offset so Python doesn't need block shapes early.
            if (src_dp_rank != 0) {
                int64_t offset = static_cast<int64_t>(config.num_blocks) * src_dp_rank;
                for (auto &id : src_ids) {
                    id += offset;
                }
            }

            // *** MOCK: simulate TCP I/O latency ***
            // In real ox, async_write(src_ids) + async_read(bufs) takes
            // some time (network round-trip + data transfer). Without this
            // delay, recv and send on the ZMQ socket can collide so hard
            // that libzmq's internal memory corrupts (double free / abort),
            // which is a different failure mode than the real-world symptom
            // (silent message loss with ox still alive). Adding a small
            // delay makes the timing more realistic: ZMQ recv and send
            // are interleaved with actual I/O wait, so collisions are
            // less violent but still possible under high concurrency.
            asio::steady_timer timer(co_await asio::this_coro::executor);
            timer.expires_after(std::chrono::milliseconds(2));
            co_await timer.async_wait(asio::use_awaitable);

            std::cout << "[MOCK-CONN] rank=" << rank
                      << " req=" << request_id
                      << " blocks=" << dst_ids.size()
                      << " - completed" << std::endl;
            // NOTE: per-request log is intentionally kept for debugging ZMQ
            // loss. If it becomes a bottleneck, comment out the above cout.

            co_await upstream.async_send(
                boost::system::error_code{},
                std::make_tuple(request_id, dst_ids),
                asio::use_awaitable);
        }
    }

    asio::awaitable<void> submit_request(
        std::string request_id, table_id_t table_id,
        block_list_t src_block_ids, block_list_t dst_block_ids, int src_dp_rank)
    {
        co_await request.async_send(boost::system::error_code{},
            std::make_tuple(std::move(request_id), table_id,
                std::move(src_block_ids), std::move(dst_block_ids), src_dp_rank),
            asio::use_awaitable);
    }

private:
    Config &config;
    int rank;
    ConnectionChannel request;
    ShardChannel &upstream;
};
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
class MockTPShard {
public:
    MockTPShard(int rank, Config &config, GroupChannel &channel)
        : rank(rank), downstream(config.get_io_context(), 128), upstream(channel)
    {
        conn_per_req = config.connections_per_req;

        // Create mock connections (no TCP needed)
        for (std::size_t i = 0; i < config.connections_per_shard; ++i) {
            connections.emplace_back(
                std::make_shared<MockCoroutineConnection>(
                    config.get_io_context(), config, rank, downstream));
            connections[i]->start();
        }

        std::cout << "[MOCK-SHARD] rank=" << rank
                  << " conns=" << connections.size() << " started" << std::endl;
    }

    asio::awaitable<void> gather(RequestMessage req)
    {
        requests_mutex.lock();
        task_status[req.request_id] = std::set<block_id_t>(req.dst_block_ids.begin(), req.dst_block_ids.end());
        assert(task_status[req.request_id].size() == req.dst_block_ids.size());
        requests_mutex.unlock();

        size_t total_ids = req.dst_block_ids.size();
        size_t num_conns = connections.size();
        num_conns = num_conns > conn_per_req ? conn_per_req : num_conns;

        if (total_ids == 0 || num_conns == 0) {
            co_await upstream.async_send(
                boost::system::error_code{}, std::make_tuple(req.request_id, rank, true),
                asio::use_awaitable);
            co_return;
        }

        size_t base_count = total_ids / num_conns;
        size_t remainder = total_ids % num_conns;
        auto it_src = req.src_block_ids.begin();
        auto it_dst = req.dst_block_ids.begin();

        for (size_t i = 0; i < num_conns; i++) {
            size_t count = base_count + (i < remainder ? 1 : 0);
            if (count == 0) break;

            block_list_t src_ids(it_src, std::next(it_src, count));
            block_list_t dst_ids(it_dst, std::next(it_dst, count));
            std::advance(it_src, count);
            std::advance(it_dst, count);

            size_t conn_index = last;
            co_spawn(co_await asio::this_coro::executor,
                connections[conn_index]->submit_request(
                    req.request_id, req.table_id, src_ids, dst_ids, req.src_dp_rank),
                detached);

            last = (last + 1) % connections.size();
        }
    }

    asio::awaitable<void> run()
    {
        while (true) {
            auto [request_id, ids] = co_await downstream.async_receive(asio::use_awaitable);

            requests_mutex.lock();
            for (auto id : ids) {
                task_status[request_id].erase(id);
            }

            if (task_status[request_id].empty()) {
                task_status.erase(request_id);
                requests_mutex.unlock();
                co_await upstream.async_send(
                    boost::system::error_code{}, std::make_tuple(request_id, rank, true),
                    asio::use_awaitable);
            } else {
                requests_mutex.unlock();
            }
        }
        co_return;
    }

private:
    mutable std::shared_mutex requests_mutex;
    int last = 0;
    size_t conn_per_req = 4;
    int rank;
    std::vector<std::shared_ptr<MockCoroutineConnection>> connections;
    std::unordered_map<request_id_t, std::set<block_id_t>> task_status;
    ShardChannel downstream;
    GroupChannel &upstream;
};

// ---------------------------------------------------------------------------
// MockTPGroup - same logic as real TPGroup but uses MockTPShard
// ---------------------------------------------------------------------------
class MockTPGroup {
public:
    MockTPGroup(Config &config, ZMQChannel &channel)
        : downstream(config.get_io_context(), 128), upstream(channel)
    {
        // If Config exposes shard_clusters (multi-cluster), build per cluster;
        // otherwise fall back to single cluster from flat shard_list.
        if (!config.shard_clusters.empty()) {
            clusters.reserve(config.shard_clusters.size());
            for (size_t c = 0; c < config.shard_clusters.size(); ++c) {
                clusters.emplace_back();
                auto &vec = clusters.back();
                vec.reserve(config.shard_clusters[c].size());
                for (size_t j = 0; j < config.shard_clusters[c].size(); ++j) {
                    auto shard = std::make_shared<MockTPShard>(static_cast<int>(j), config, downstream);
                    vec.push_back(shard);
                    co_spawn(config.get_io_context(), shard->run(), detached);
                }
            }
        } else {
            clusters.emplace_back();
            auto &vec = clusters.back();
            for (int rank = 0; rank < static_cast<int>(config.shard_list.size()); rank++) {
                auto shard = std::make_shared<MockTPShard>(rank, config, downstream);
                vec.push_back(shard);
                co_spawn(config.get_io_context(), shard->run(), detached);
            }
        }

        size_t total_shards = 0;
        for (auto &cluster : clusters) total_shards += cluster.size();
        std::cout << "[MOCK-GROUP] " << clusters.size() << " cluster(s), "
                  << total_shards << " shard(s) (mock) created" << std::endl;
    }

    asio::awaitable<void> run()
    {
        uint64_t completed_count = 0;
        try {
            while (true) {
                auto [request_id, rank, shard_ok] = co_await downstream.async_receive(asio::use_awaitable);

                requests_mutex.lock();
                auto it = requests_status.find(request_id);
                if (it == requests_status.end()) {
                    requests_mutex.unlock();
                    continue;
                }
                auto &[client_id, table_id, rank_finished, block_ids, failed] = it->second;

                if (!shard_ok) failed = true;

                if (rank >= 0 && static_cast<size_t>(rank) < rank_finished.size()) {
                    rank_finished[static_cast<size_t>(rank)] = true;
                }

                if (std::all_of(rank_finished.begin(), rank_finished.end(), [](bool b) { return b; })) {
                    bool req_failed = failed;
                    client_id_t cid = client_id;
                    requests_status.erase(request_id);
                    requests_mutex.unlock();

                    ++completed_count;
                    if (completed_count % 1000 == 0) {
                        std::cout << "[MOCK-GROUP] total completed: " << completed_count << std::endl;
                    }
                    co_await upstream.async_send(
                        boost::system::error_code{},
                        std::make_tuple(cid, request_id, !req_failed),
                        asio::use_awaitable);
                } else {
                    requests_mutex.unlock();
                }
            }
        } catch (const std::exception &e) {
            std::cout << "[MOCK-GROUP] Response sender stopped: " << e.what() << std::endl;
        }
        co_return;
    }

    asio::awaitable<void> gather(client_id_t client_id, RequestMessage req)
    {
        int cid = 0;

        {
            std::unique_lock<std::shared_mutex> lock(requests_mutex);
            size_t shard_cnt = clusters[cid].size();
            requests_status[req.request_id] =
                std::make_tuple(client_id, req.table_id,
                                std::vector<bool>(shard_cnt, false), req.dst_block_ids, false);
        }

        for (auto &shard : clusters[cid]) {
            co_spawn(co_await asio::this_coro::executor, shard->gather(req), detached);
        }
    }

public:
    mutable std::shared_mutex requests_mutex;
    std::unordered_map<request_id_t,
        std::tuple<client_id_t, table_id_t, std::vector<bool>, std::vector<block_id_t>, bool>>
        requests_status;
    std::vector<std::vector<std::shared_ptr<MockTPShard>>> clusters;
    GroupChannel downstream;
    ZMQChannel &upstream;
};

// Dynamic connection pool for mock
class MockTPGroupManager {
public:
    MockTPGroupManager(Config &config, ZMQChannel &channel)
        : config(config), channel(channel) {}

    std::shared_ptr<MockTPGroup> get_or_create(const std::string &shard_list_str)
    {
        std::unique_lock<std::shared_mutex> lock(mutex);
        auto it = groups.find(shard_list_str);
        if (it != groups.end()) {
            std::cout << "[DYNAMIC-TOPO][MOCK] Reusing existing group for "
                      << shard_list_str << std::endl;
            return it->second;
        }

        auto group = std::make_shared<MockTPGroup>(config, channel);
        co_spawn(config.get_io_context(), group->run(), detached);
        groups[shard_list_str] = group;
        std::cout << "[DYNAMIC-TOPO][MOCK] *** CREATED NEW GROUP *** for "
                  << shard_list_str << std::endl;
        return group;
    }

private:
    mutable std::shared_mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<MockTPGroup>> groups;
    Config &config;
    ZMQChannel &channel;
};

// ---------------------------------------------------------------------------
// ZMQ receiver / sender coroutines (same as real ox)
// ---------------------------------------------------------------------------
std::atomic<uint64_t> g_recv_count{0};
std::atomic<uint64_t> g_send_count{0};

asio::awaitable<void> response_sender(ZmqCoroutineSocket &router_socket, ZMQChannel &response_channel)
{
    try {
        while (true) {
            auto [client_id, request_id, success] = co_await response_channel.async_receive(asio::use_awaitable);

            ResponseMessage response = {request_id, success};
            std::stringstream buffer;
            msgpack::pack(buffer, response);
            std::string response_data = buffer.str();

            std::vector<zmq::message_t> response_messages;
            response_messages.emplace_back(client_id.data(), client_id.size());
            response_messages.emplace_back(response_data.data(), response_data.size());

            uint64_t sent = g_send_count.fetch_add(1) + 1;
            if (sent % 1000 == 0) {
                std::cout << "[ZMQ-SEND] total sent: " << sent << std::endl;
            }

            global_stats_update_running(-1);
            bool ok = co_await router_socket.async_send_multipart(std::move(response_messages));
            if (!ok) {
                std::cerr << "[ZMQ-SEND-ERROR] Failed to send response for req=" << request_id << std::endl;
            }
        }
    } catch (const std::exception &e) {
        std::cout << "[ZMQ-SEND] Response sender stopped: " << e.what() << std::endl;
    }
}

asio::awaitable<void> router_receiver(ZmqCoroutineSocket &router_socket, MockTPGroupManager &manager)
{
    while (true) {
        try {
            auto msg = co_await router_socket.async_recv_multipart();
            if (msg && msg->size() == 2) {
                std::vector<zmq::message_t> messages = std::move(*msg);
                const auto *data0 = static_cast<const uint8_t *>(messages[0].data());
                client_id_t client_id(data0, data0 + messages[0].size());

                const char *data1 = static_cast<const char *>(messages[1].data());
                std::string request_data(data1, data1 + messages[1].size());

                msgpack::object_handle handle = msgpack::unpack(request_data.data(), request_data.size());
                RequestMessage request;
                handle.get().convert(request);

                uint64_t recv = g_recv_count.fetch_add(1) + 1;
                if (recv % 1000 == 0) {
                    std::cout << "[ZMQ-RECV] total recv: " << recv << std::endl;
                }

                global_stats_update_running(1);

                std::cout << "[DYNAMIC-TOPO][MOCK] OX received req_id=" << request.request_id
                          << " remote_ox_shard_list=" << request.remote_ox_shard_list
                          << " src_blocks=" << request.src_block_ids.size()
                          << " dst_blocks=" << request.dst_block_ids.size() << std::endl;

                auto group = manager.get_or_create(request.remote_ox_shard_list);
                co_spawn(co_await asio::this_coro::executor,
                    group->gather(client_id, request), detached);
            } else {
                std::cerr << "[ZMQ-RECV-WARN] Unexpected msg size: "
                          << (msg ? std::to_string(msg->size()) : "nullopt") << std::endl;
            }
        } catch (const std::exception &e) {
            std::cerr << "[ZMQ-RECV-ERROR] " << e.what() << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
// Periodic ZMQ statistics for mock stress tests.
// ---------------------------------------------------------------------------
asio::awaitable<void> print_zmq_statistics()
{
    while (true) {
        asio::steady_timer timer(co_await asio::this_coro::executor);
        timer.expires_after(std::chrono::seconds(5));
        co_await timer.async_wait(asio::use_awaitable);

        std::cout << "[ZMQ-STATS] recv=" << g_recv_count.load()
                  << " sent=" << g_send_count.load()
                  << " diff=" << (g_recv_count.load() - g_send_count.load())
                  << std::endl;
    }
}

// ---------------------------------------------------------------------------
// main - simplified: no --addr, no --block-table-shm, but supports --shard-list
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    try {
        // Minimal config parsing - no block-table-shm needed since we skip TCP
        Config config;
        config.io_context = std::make_shared<asio::io_context>();

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "--zmq-port" || arg == "--zmq_port") && i + 1 < argc) {
                config.zmq_port = std::stoul(argv[++i]);
            } else if ((arg == "--shard-list" || arg == "--shard_list") && i + 1 < argc) {
                std::string val = argv[++i];
                parse_shard_list_grouped(val, config.shard_clusters, config.shard_list);
            } else if ((arg == "--num-blocks" || arg == "--num_blocks") && i + 1 < argc) {
                config.num_blocks = std::stoul(argv[++i]);
            } else if ((arg == "--num-layers" || arg == "--num_layers") && i + 1 < argc) {
                config.num_layers = std::stoul(argv[++i]);
            } else if ((arg == "--tokens-per-block" || arg == "--tokens_per_block") && i + 1 < argc) {
                config.num_tokens_per_block = std::stoul(argv[++i]);
            } else if ((arg == "--dims") && i + 1 < argc) {
                std::string str(argv[++i]);
                std::replace(str.begin(), str.end(), ',', ' ');
                config.block_dims.clear();
                std::stringstream ss(str);
                std::copy(std::istream_iterator<size_t>(ss),
                    std::istream_iterator<size_t>(),
                    std::back_inserter(config.block_dims));
            } else if ((arg == "--dtype") && i + 1 < argc) {
                config.element_size = std::stoul(argv[++i]);
            } else if ((arg == "--num-threads" || arg == "--num_threads") && i + 1 < argc) {
                config.num_threads = std::stoul(argv[++i]);
            } else if ((arg == "--num-connections" || arg == "--num_connections") && i + 1 < argc) {
                config.connections_per_shard = std::stoul(argv[++i]);
            } else if ((arg == "--num-connections-per-req" || arg == "--num_connections_per_req") && i + 1 < argc) {
                config.connections_per_req = std::stoul(argv[++i]);
            } else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: mock_ox [options]\n"
                          << "Options:\n"
                          << "  --zmq-port <port>              ZMQ port (default: 5555)\n"
                          << "  --shard-list <grp1;grp2;...>   Shard server IPs (same format as ox)\n"
                          << "  --num-blocks <num>             Number of blocks (default: 1024)\n"
                          << "  --num-layers <size>            Number of layers per block (default: 62)\n"
                          << "  --tokens-per-block <size>      Number of tokens per block (default: 128)\n"
                          << "  --dims <list of size>          KV dimension split (default: 512,64,128)\n"
                          << "  --dtype <size>                 Element size (default: 2 for bfloat16)\n"
                          << "  --num-threads <threads>        IO threads (default: 16)\n"
                          << "  --num-connections <conn>       Connections per shard (default: 16)\n"
                          << "  --num-connections-per-req <n>  Connections per request (default: 4)\n";
                return 0;
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                return 1;
            }
        }

        config.update_values();

        asio::io_context &io_context = config.get_io_context();

        g_program_start_time = std::chrono::steady_clock::now();

        ZMQChannel response_channel(io_context, 128);
        ZmqCoroutineSocket zmq_router(ZMQ_ROUTER, io_context);

        // Dynamic mode: MockTPGroupManager creates groups on demand
        auto manager = std::make_shared<MockTPGroupManager>(config, response_channel);

        // If --shard-list given, pre-create a default group
        if (!config.shard_list.empty() || !config.shard_clusters.empty()) {
            manager->get_or_create("mock-default");
        }

        std::string address = "tcp://*:" + std::to_string(config.zmq_port);
        zmq_router.bind(address);

        co_spawn(io_context, router_receiver(zmq_router, *manager), detached);
        co_spawn(io_context, response_sender(zmq_router, response_channel), detached);
        if (ox_debug_logging_enabled()) {
            co_spawn(io_context, print_statistics(), detached);
        }
        co_spawn(io_context, print_zmq_statistics(), detached);

        std::cout << "[MOCK-OX] Started. ZMQ: " << address
                  << " threads: " << config.num_threads << std::endl;

        std::vector<std::thread> threads;
        for (size_t i = 0; i < config.num_threads; ++i) {
            threads.emplace_back([&io_context]() { io_context.run(); });
        }

        io_context.run();

        for (auto &thread : threads)
            thread.join();
    } catch (const std::exception &e) {
        std::cerr << "[MOCK-OX] Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
