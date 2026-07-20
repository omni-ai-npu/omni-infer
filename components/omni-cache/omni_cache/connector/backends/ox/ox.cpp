// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <set>
#include <atomic>
#include <mutex>
#include <exception>
#include <msgpack.hpp>
#include <sys/mman.h>
#include <shared_mutex>
#include <utility>

#include <ox_config.hpp>
#include <zmq_coroutine.hpp>
#include <ox_metrics.hpp>
#include <ox_server.hpp>
#include <ox_log.hpp>
#include <ox_kv_merger.h>
#include <chrono>
#include <boost/asio/experimental/parallel_group.hpp>
#include <boost/asio/experimental/promise.hpp>

namespace asio = boost::asio;
using asio::co_spawn;
using asio::detached;
using asio::experimental::concurrent_channel;
using asio::ip::tcp;
using namespace boost::asio::experimental::awaitable_operators;

struct RequestMessage {
    request_id_t request_id;
    table_id_t table_id;
    block_list_t src_block_ids;  // P/server side block ids
    block_list_t dst_block_ids;  // D/client side block ids
    std::string remote_ox_shard_list;  // dynamic shard endpoints (e.g. "ip1:port,ip2:port")
    int src_dp_rank{0};          // remote DP rank — ox applies offset internally
    MSGPACK_DEFINE_MAP(request_id, table_id, src_block_ids, dst_block_ids, remote_ox_shard_list, src_dp_rank)
};

struct ResponseMessage {
    request_id_t request_id;
    bool success;
    MSGPACK_DEFINE_MAP(request_id, success)
};

using ResponseTask = std::tuple<client_id_t, request_id_t, bool>;
using ZMQChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ResponseTask)>;

using ConnectionMessage = std::tuple<request_id_t, table_id_t, block_list_t, block_list_t, int>;
using ConnectionChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ConnectionMessage)>;

using ShardMessage = std::tuple<std::string, block_list_t>;
using ShardChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ShardMessage)>;

using GroupMessage = std::tuple<request_id_t, int, bool>;  // (request_id, rank, success)
using GroupChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, GroupMessage)>;

class CoroutineConnection {
public:
    CoroutineConnection(asio::io_context &io_context, boost::asio::ip::tcp::endpoint addr, Config &config, int rank,
        BlockTable &bt, ShardChannel &response)
        : io_context(io_context), socket(io_context), config(config), rank(rank), addr(addr), bt(bt),
          request(config.get_io_context(), 128), upstream(response)
    {}

    void start()
    {
        asio::co_spawn(socket.get_executor(), run(), asio::detached);
    }

    asio::awaitable<void> run()
    {
        const int max_connect_retries = config.max_connect_retries;
        const int max_request_retries = config.max_request_retries;
        std::optional<ConnectionMessage> pending_request;
        int request_retries = 0;

        while (true) {
            if (!pending_request) {
                auto msg = co_await request.async_receive(asio::use_awaitable);
                pending_request = std::move(msg);
                request_retries = 0;
            }

            // --- Connect phase ---
            bool connected = false;
            for (int attempt = 0; attempt < max_connect_retries; ++attempt) {
                try {
                    co_await socket.async_connect(addr, asio::use_awaitable);
                    optimize_tcp_socket(socket);
                    connected = true;
                    std::cout << "[DYNAMIC-TOPO] Connection established to "
                              << addr.address() << ":" << addr.port() << std::endl;
                    break;
                } catch (const std::exception &e) {
                    std::cerr << "Connect attempt " << (attempt + 1) << " to "
                              << addr.address() << ":" << addr.port()
                              << " failed: " << e.what() << std::endl;
                    boost::system::error_code ec;
                    socket.close(ec);
                    socket = tcp::socket(io_context);
                    if (attempt + 1 < max_connect_retries) {
                        asio::steady_timer timer(co_await asio::this_coro::executor);
                        timer.expires_after(std::chrono::seconds(3));
                        co_await timer.async_wait(asio::use_awaitable);
                    }
                }
            }

            if (!connected) {
                std::cerr << "Failed to connect to " << addr.address() << ":" << addr.port()
                          << " after " << max_connect_retries << " attempts" << std::endl;
                // Drain pending request with failure
                if (pending_request) {
                    auto &[req_id, table_id, src_ids, dst_ids, src_dp_rank] = *pending_request;
                    co_await upstream.async_send(
                        boost::system::error_code{}, std::make_tuple(req_id, block_list_t{}), asio::use_awaitable);
                    pending_request.reset();
                }
                continue;
            }

            // --- Request processing phase ---
            try {
                while (true) {
                    if (!pending_request) {
                        auto msg = co_await request.async_receive(asio::use_awaitable);
                        pending_request = std::move(msg);
                        request_retries = 0;
                    }

                    auto &[request_id, table_id, src_ids, dst_ids, src_dp_rank] = *pending_request;

                    if (src_ids.empty()) {
                        co_await upstream.async_send(
                            boost::system::error_code{}, std::make_tuple(request_id, dst_ids), asio::use_awaitable);
                        pending_request.reset();
                        continue;
                    }

                    if (src_dp_rank != 0) {
                        int64_t offset = static_cast<int64_t>(config.num_blocks) * src_dp_rank;
                        for (auto &id : src_ids) {
                            id += offset;
                        }
                        // Reset src_dp_rank to 0 so we don't re-apply offset on retry
                        src_dp_rank = 0;
                    }

                    auto bufs = bt.get_buffers_layerwise(table_id, dst_ids, rank);

                    co_await (asio::async_write(socket,
                                  asio::buffer(src_ids.data(), src_ids.size() * sizeof(block_id_t)),
                                  asio::use_awaitable) &&
                              asio::async_read(socket, bufs, asio::use_awaitable));

                    global_stats_update(dst_ids.size() * bt.block_tp_size());

                    co_await upstream.async_send(
                        boost::system::error_code{}, std::make_tuple(request_id, dst_ids), asio::use_awaitable);
                    pending_request.reset();
                }
            } catch (const std::exception &e) {
                std::cerr << "Connection " << addr.address() << ":" << addr.port()
                          << " error: " << e.what() << std::endl;
                boost::system::error_code ec;
                socket.close(ec);
                socket = tcp::socket(io_context);

                if (pending_request) {
                    ++request_retries;
                    if (request_retries >= max_request_retries) {
                        auto &[req_id, table_id, src_ids, dst_ids, src_dp_rank] = *pending_request;
                        std::cerr << "Request " << req_id << " failed after "
                                  << max_request_retries << " retries" << std::endl;
                        co_await upstream.async_send(
                            boost::system::error_code{}, std::make_tuple(req_id, block_list_t{}), asio::use_awaitable);
                        pending_request.reset();
                    }
                }
                // Loop back to reconnect
            }
        }
    }

    asio::awaitable<void> submit_request(
        std::string &request_id, table_id_t table_id, block_list_t src_block_ids, block_list_t dst_block_ids, int src_dp_rank)
    {
        co_await request.async_send(boost::system::error_code{},
            std::make_tuple(request_id, table_id, std::move(src_block_ids), std::move(dst_block_ids), src_dp_rank),
            asio::use_awaitable);
    }

private:
    asio::io_context &io_context;
    tcp::socket socket;
    const Config &config;
    int rank;
    boost::asio::ip::tcp::endpoint addr;

    BlockTable &bt;
    ConnectionChannel request;
    ShardChannel &upstream;
};

class TPShard {
public:
    TPShard(boost::asio::ip::tcp::endpoint addr, int rank, Config &config, BlockTable &bt, GroupChannel &channel)
        : ip(addr), rank(rank), downstream(config.get_io_context(), 128), upstream(channel)
    {
        conn_per_req = config.connections_per_req;

        for (std::size_t i = 0; i < config.connections_per_shard; ++i) {
            connections.emplace_back(
                std::make_shared<CoroutineConnection>(config.get_io_context(), ip, config, rank, bt, downstream));
            connections[i]->start();
        }
        std::cout << "TPShard created for " << addr << " with " << connections.size() << " connections" << std::endl;
    }

    asio::awaitable<void> gather(RequestMessage req)
    {
        // Completion tracking is based on dst ids.
        requests_mutex.lock();
        task_status[req.request_id] = std::set<block_id_t>(req.dst_block_ids.begin(), req.dst_block_ids.end());
        assert(task_status[req.request_id].size() == req.dst_block_ids.size());
        requests_mutex.unlock();

        size_t total_ids = req.dst_block_ids.size();
        size_t num_conns = std::min(connections.size(), conn_per_req);
        if (total_ids == 0 || num_conns == 0 || req.src_block_ids.empty()) {
            co_await upstream.async_send(
                boost::system::error_code{}, std::make_tuple(req.request_id, rank, true), asio::use_awaitable);
            co_return;
        }

        size_t base_count = total_ids / num_conns;
        size_t remainder = total_ids % num_conns;
        auto it_src = req.src_block_ids.begin();
        auto it_dst = req.dst_block_ids.begin();

        for (size_t i = 0; i < num_conns; i++) {
            size_t count = base_count + (i < remainder ? 1 : 0);
            if (count == 0)
                break;

            block_list_t src_ids(it_src, std::next(it_src, count));
            block_list_t dst_ids(it_dst, std::next(it_dst, count));
            std::advance(it_src, count);
            std::advance(it_dst, count);

            size_t conn_index =
                last.fetch_add(1, std::memory_order_relaxed) % connections.size();
            co_spawn(co_await asio::this_coro::executor,
                connections[conn_index]->submit_request(req.request_id, req.table_id, src_ids, dst_ids, req.src_dp_rank),
                detached);
        }
    }

    asio::awaitable<void> run()
    {
        while (true) {
            auto [request_id, ids] = co_await downstream.async_receive(asio::use_awaitable);

            // Empty block list signals a connection failure for this shard
            if (ids.empty()) {
                {
                    std::unique_lock<std::shared_mutex> lock(requests_mutex);
                    task_status.erase(request_id);
                }
                co_await upstream.async_send(
                    boost::system::error_code{}, std::make_tuple(request_id, rank, false), asio::use_awaitable);
                continue;
            }

            requests_mutex.lock();
            auto task_it = task_status.find(request_id);
            if (task_it == task_status.end()) {
                requests_mutex.unlock();
                continue;
            }
            for (auto id : ids) {
                task_it->second.erase(id);
            }

            if (task_it->second.empty()) {
                task_status.erase(task_it);
                requests_mutex.unlock();
                co_await upstream.async_send(
                    boost::system::error_code{}, std::make_tuple(request_id, rank, true), asio::use_awaitable);
            } else {
                requests_mutex.unlock();
            }
        }
        co_return;
    }

private:
    mutable std::shared_mutex requests_mutex;
    std::atomic<size_t> last{0};
    size_t conn_per_req = 4;
    boost::asio::ip::tcp::endpoint ip;
    int rank;  // rank in the cluster
    std::vector<std::shared_ptr<CoroutineConnection>> connections;
    std::unordered_map<std::string, std::set<block_id_t>> task_status;

    ShardChannel downstream;
    GroupChannel &upstream;
};

class TPGroup {
public:
    // Dynamic mode: build from endpoint list (per-request shard discovery)
    TPGroup(address_list_t &endpoints, Config &config, BlockTable &bt, ZMQChannel &channel)
        : block_size(config.block_size), downstream(config.get_io_context(), 128), bt(bt), upstream(channel),
          merger(62, 128, 704, endpoints.size())
    {
        clusters.emplace_back();
        auto &vec = clusters.back();
        for (int rank = 0; rank < static_cast<int>(endpoints.size()); rank++) {
            auto &ep = endpoints[rank];
            auto shard = std::make_shared<TPShard>(ep, rank, config, bt, downstream);
            vec.push_back(shard);
            co_spawn(config.get_io_context(), shard->run(), detached);
        }
        std::cout << "TPGroup created dynamically with " << endpoints.size() << " shards" << std::endl;
    }

    asio::awaitable<void> run()
    {
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

                if (!shard_ok) {
                    failed = true;
                    std::cerr << "[DYNAMIC-TOPO] Shard rank=" << rank
                              << " reported FAILURE for req=" << request_id << std::endl;
                }

                if (rank >= 0 && static_cast<size_t>(rank) < rank_finished.size()) {
                    rank_finished[static_cast<size_t>(rank)] = true;
                }

                if (std::all_of(rank_finished.begin(), rank_finished.end(), [](bool b) { return b; })) {
                    bool req_failed = failed;
                    client_id_t cid = client_id;
                    requests_status.erase(request_id);
                    requests_mutex.unlock();

                    std::cout << "<<<<<<< Send response: " << request_id
                              << " success=" << (!req_failed ? "true" : "false") << std::endl;
                    co_await upstream.async_send(
                        boost::system::error_code{}, std::make_tuple(cid, request_id, !req_failed), asio::use_awaitable);
                } else {
                    requests_mutex.unlock();
                }
            }
        } catch (const std::exception &e) {
            std::cout << "Response sender stopped: " << e.what() << std::endl;
        }
        co_return;
    }

    asio::awaitable<void> gather(client_id_t client_id, RequestMessage &req)
    {
        // In dynamic mode, each TPGroup has exactly one cluster (index 0).
        // In static mode with multi-cluster, this still works since
        // TPGroupManager maps each shard_list to its own TPGroup.
        int cid = 0;

        {
            std::unique_lock<std::shared_mutex> lock(requests_mutex);
            size_t shard_cnt = clusters[cid].size();
            requests_status[req.request_id] =
                std::make_tuple(client_id, req.table_id, std::vector<bool>(shard_cnt, false), req.dst_block_ids, false);
        }

        for (auto &shard : clusters[cid]) {
            co_spawn(co_await asio::this_coro::executor, shard->gather(req), detached);
        }
    }

public:
    mutable std::shared_mutex requests_mutex;
    std::unordered_map<request_id_t, std::tuple<client_id_t, table_id_t, std::vector<bool>, std::vector<block_id_t>, bool>>
        requests_status;

    std::vector<std::vector<std::shared_ptr<TPShard>>> clusters;

    const size_t block_size;
    GroupChannel downstream;
    BlockTable &bt;
    ZMQChannel &upstream;
    KVCacheMerger merger;
};

// Dynamic connection pool: maps shard_list string → TPGroup
class TPGroupManager {
public:
    TPGroupManager(Config &config, BlockTable &bt, ZMQChannel &channel)
        : config(config), bt(bt), channel(channel) {}

    std::shared_ptr<TPGroup> get_or_create(const std::string &shard_list_str)
    {
        if (shard_list_str.empty()) {
            throw std::invalid_argument("remote_ox_shard_list must not be empty");
        }

        {
            std::shared_lock<std::shared_mutex> rlock(mutex);
            auto it = groups.find(shard_list_str);
            if (it != groups.end()) {
                if (ox_debug_logging_enabled()) {
                    std::cout << "[DYNAMIC-TOPO] TPGroupManager: reusing existing group for "
                              << shard_list_str << std::endl;
                }
                return it->second;
            }
        }

        std::unique_lock<std::shared_mutex> wlock(mutex);
        // Double-check after acquiring write lock
        auto it = groups.find(shard_list_str);
        if (it != groups.end()) {
            if (ox_debug_logging_enabled()) {
                std::cout << "[DYNAMIC-TOPO] TPGroupManager: reusing existing group for "
                          << shard_list_str << " (after write-lock)" << std::endl;
            }
            return it->second;
        }

        address_list_t endpoints = parse_address_list(shard_list_str);
        if (endpoints.empty()) {
            throw std::invalid_argument(
                "remote_ox_shard_list contains no valid endpoints");
        }
        auto group = std::make_shared<TPGroup>(endpoints, config, bt, channel);
        co_spawn(config.get_io_context(), group->run(), detached);
        groups[shard_list_str] = group;
        std::cout << "[DYNAMIC-TOPO] TPGroupManager: *** CREATED NEW GROUP *** for "
                  << shard_list_str << " (" << endpoints.size() << " shards)" << std::endl;
        return group;
    }

private:
    mutable std::shared_mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<TPGroup>> groups;
    Config &config;
    BlockTable &bt;
    ZMQChannel &channel;
};

std::unordered_map<request_id_t, std::tuple<int, std::chrono::high_resolution_clock::time_point>> time_record;

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

            // auto [num_ids, start] = time_record[request_id];
            // auto end = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch()).count();

            // std::cout << "Finished: " << request_id
            //           << "Num blocks: " << num_ids
            //           << " Duration:" << duration << " ms"
            //           << " per block:" << ((float)duration * 1.0) / (num_ids * 1.0)
            //              << " End: " << ms << std::endl;
            global_stats_update_running(-1);
            co_await router_socket.async_send_multipart(std::move(response_messages));
        }
    } catch (const std::exception &e) {
        std::cout << "Response sender stopped: " << e.what() << std::endl;
    }
}

asio::awaitable<void> router_receiver(ZmqCoroutineSocket &router_socket, TPGroupManager &manager)
{
    while (true) {
        client_id_t client_id;
        request_id_t request_id;
        try {
            auto msg = co_await router_socket.async_recv_multipart();
            if (msg && msg->size() == 2) {
                std::vector<zmq::message_t> messages = std::move(*msg);
                const auto *data0 = static_cast<const uint8_t *>(messages[0].data());
                client_id.assign(data0, data0 + messages[0].size());

                const char *data1 = static_cast<const char *>(messages[1].data());
                std::string request_data(data1, data1 + messages[1].size());

                msgpack::object_handle handle = msgpack::unpack(request_data.data(), request_data.size());
                RequestMessage request;
                handle.get().convert(request);
                request_id = request.request_id;

                auto start = std::chrono::high_resolution_clock::now();
                time_record[request.request_id] = make_tuple(request.src_block_ids.size(), start);

                global_stats_update_running(1);

                if (ox_debug_logging_enabled()) {
                    std::cout << "[DYNAMIC-TOPO] OX received req_id=" << request.request_id
                              << " remote_ox_shard_list=" << request.remote_ox_shard_list
                              << " src_blocks=" << request.src_block_ids.size()
                              << " dst_blocks=" << request.dst_block_ids.size() << std::endl;
                }

                auto group = manager.get_or_create(request.remote_ox_shard_list);
                co_spawn(co_await asio::this_coro::executor, group->gather(client_id, request), detached);
            } else {
            }
        } catch (const std::exception &e) {
            std::cerr << "Receiver error: " << e.what() << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    try {
        Config config = parse_arguments(argc, argv);
        std::cout << "[OX] ready for creating bt." << std::endl;
        auto t1 = std::chrono::steady_clock::now();
        BlockTable bt(config);
        auto t2 = std::chrono::steady_clock::now();
        std::cout << "[OX] bt constructed, costing " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

        block_list_t blocks = {0, 5, 11};
        bt.get_buffers_layerwise(0, blocks, 1);

        asio::io_context &io_context = config.get_io_context();

        g_program_start_time = std::chrono::steady_clock::now();

        std::vector<std::shared_ptr<Server>> server_list;
        for (auto &endpoint : config.server_list) {
            server_list.emplace_back(std::make_shared<Server>(io_context, endpoint, bt));
        }

        for (auto &server : server_list) {
            co_spawn(io_context, server->run(), detached);
        }

        ZMQChannel response_channel(io_context, 128);
        ZmqCoroutineSocket zmq_router(ZMQ_ROUTER, io_context);

        // Create TPGroupManager for dynamic shard discovery.
        // If --shard-list was given at startup (static mode), pre-build a
        // TPGroup for backward compatibility; otherwise groups are created
        // on-demand as requests arrive with remote_ox_shard_list.
        auto manager = std::make_shared<TPGroupManager>(config, bt, response_channel);

        std::string address = "tcp://*:" + std::to_string(config.zmq_port);
        zmq_router.bind(address);

        co_spawn(io_context, router_receiver(zmq_router, *manager), detached);
        co_spawn(io_context, response_sender(zmq_router, response_channel), detached);
        if (ox_debug_logging_enabled()) {
            co_spawn(io_context, print_statistics(), detached);
        }

        std::cout << "Omni Xfer started. ZMQ: " << address << std::endl;

        std::vector<std::thread> threads;
        for (size_t i = 0; i < config.num_threads; ++i) {
            threads.emplace_back([&io_context]() { io_context.run(); });
        }

        io_context.run();

        for (auto &thread : threads)
            thread.join();
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

// g++ -std=c++20 -DNDEBUG -fcoroutines -I./  -g -march=native  ox.cpp -o ox -lzmq -lpthread
