// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <set>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <exception>
#include <msgpack.hpp>
#include <sys/mman.h>
#include <shared_mutex>

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

static bool ox_env_enabled(const char *name)
{
    const char *value = std::getenv(name);
    return value != nullptr &&
           (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
               std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "yes") == 0 ||
               std::strcmp(value, "YES") == 0 || std::strcmp(value, "on") == 0 ||
               std::strcmp(value, "ON") == 0);
}

static bool ox_statistics_log_enabled()
{
    static const bool enabled = ox_env_enabled("OX_STATISTICS_LOG");
    return enabled;
}

static bool ox_request_trace_enabled()
{
    static const bool enabled = ox_env_enabled("OX_REQUEST_TRACE");
    return enabled;
}

template <typename... Args>
static void ox_trace(Args &&...args)
{
    if (!ox_request_trace_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_output_mutex);
    (std::cout << ... << args) << std::endl;
}

static bool ox_debug_logging_enabled()
{
    static const bool enabled = ox_env_enabled("OX_DEBUG_LOG");
    return enabled;
}

struct RequestMessage {
    request_id_t request_id;
    table_id_t table_id;
    block_list_t src_block_ids;  // P/server side block ids
    block_list_t dst_block_ids;  // D/client side block ids
    std::string remote_ox_shard_list;  // dynamic shard endpoints (e.g. "ip1:port,ip2:port")
    MSGPACK_DEFINE_MAP(request_id, table_id, src_block_ids, dst_block_ids, remote_ox_shard_list)
};

struct ResponseMessage {
    request_id_t request_id;
    bool success;
    MSGPACK_DEFINE_MAP(request_id, success)
};

using ResponseTask = std::tuple<client_id_t, request_id_t, bool>;
using ZMQChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ResponseTask)>;

using ConnectionMessage = std::tuple<request_id_t, table_id_t, block_list_t, block_list_t>;
using ConnectionChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ConnectionMessage)>;

using ShardMessage = std::tuple<std::string, block_list_t>;
using ShardChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, ShardMessage)>;

using GroupMessage = std::tuple<request_id_t, int, bool>;  // (request_id, rank, success)
using GroupChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, GroupMessage)>;

class CoroutineConnection {
public:
    CoroutineConnection(asio::io_context &io_context, boost::asio::ip::tcp::endpoint addr, Config &config, int rank,
        BlockTable &bt, ShardChannel &response, size_t tp_override = 0)
        : io_context(io_context), socket(io_context), config(config), rank(rank), addr(addr), bt(bt),
          request(config.get_io_context(), 128), upstream(response), tp_override(tp_override)
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
                    break;
                } catch (const std::exception &e) {
                    std::cerr << "Connect attempt " << (attempt + 1) << " to "
                              << addr.address() << ":" << addr.port()
                              << " failed: " << e.what() << std::endl;
                    print_ox_backtrace(std::cerr);
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
                if (pending_request) {
                    auto &[req_id, table_id, src_ids, dst_ids] = *pending_request;
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

                    auto &[request_id, table_id, src_ids, dst_ids] = *pending_request;

                    if (src_ids.empty()) {
                        co_await upstream.async_send(
                            boost::system::error_code{}, std::make_tuple(request_id, dst_ids), asio::use_awaitable);
                        pending_request.reset();
                        continue;
                    }

                    auto bufs = bt.get_buffers_layerwise(table_id, dst_ids, rank, tp_override);

                    const uint32_t block_count_network = htonl(static_cast<uint32_t>(src_ids.size()));
                    std::array<asio::const_buffer, 2> request_buffers = {
                        asio::buffer(&block_count_network, sizeof(block_count_network)),
                        asio::buffer(src_ids.data(), src_ids.size() * sizeof(block_id_t))};

                    ox_trace("OX trace: request=", request_id, " event=p_transfer_start peer=", addr.address(),
                        ":", addr.port(), " rank=", rank, " table=", table_id, " src_blocks=", src_ids.size(),
                        " dst_blocks=", dst_ids.size());

                    co_await (asio::async_write(socket, request_buffers, asio::use_awaitable) &&
                              asio::async_read(socket, bufs, asio::use_awaitable));

                    ox_trace("OX trace: request=", request_id, " event=p_transfer_done peer=", addr.address(),
                        ":", addr.port(), " rank=", rank, " dst_blocks=", dst_ids.size());

                    global_stats_update(dst_ids.size() * bt.block_tp_size());

                    co_await upstream.async_send(
                        boost::system::error_code{}, std::make_tuple(request_id, dst_ids), asio::use_awaitable);
                    pending_request.reset();
                }
            } catch (const std::exception &e) {
                std::cerr << "OX request processing error: peer="
                        << addr.address() << ":" << addr.port()
                        << " retry=" << request_retries
                        << " message=" << e.what() << std::endl;
                print_ox_backtrace(std::cerr);
                boost::system::error_code ec;
                socket.close(ec);
                socket = tcp::socket(io_context);

                if (pending_request) {
                    ++request_retries;
                    if (request_retries >= max_request_retries) {
                        auto &[req_id, table_id, src_ids, dst_ids] = *pending_request;
                        co_await upstream.async_send(
                            boost::system::error_code{}, std::make_tuple(req_id, block_list_t{}), asio::use_awaitable);
                        pending_request.reset();
                    }
                }
            }
        }
    }

    asio::awaitable<void> submit_request(
        std::string &request_id, table_id_t table_id, block_list_t &src_block_ids, block_list_t &dst_block_ids)
    {
        co_await request.async_send(boost::system::error_code{},
            std::make_tuple(request_id, table_id, src_block_ids, dst_block_ids),
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
    size_t tp_override;
};

class TPShard {
public:
    TPShard(boost::asio::ip::tcp::endpoint addr, int rank, Config &config, BlockTable &bt, GroupChannel &channel,
        size_t tp_override = 0)
        : ip(addr), rank(rank), downstream(config.get_io_context(), 128), upstream(channel)
    {
        conn_per_req = config.connections_per_req;

        for (std::size_t i = 0; i < config.connections_per_shard; ++i) {
            connections.emplace_back(
                std::make_shared<CoroutineConnection>(config.get_io_context(), ip, config, rank, bt, downstream, tp_override));
            connections[i]->start();
        }
        std::cout << "TPShard created for " << addr << " with " << connections.size()
                  << " connections, tp_override=" << tp_override << std::endl;
    }

    asio::awaitable<void> gather(RequestMessage &req)
    {
        // Completion tracking is based on dst ids.
        requests_mutex.lock();
        task_status[req.request_id] = std::set<block_id_t>(req.dst_block_ids.begin(), req.dst_block_ids.end());
        assert(task_status[req.request_id].size() == req.dst_block_ids.size());
        requests_mutex.unlock();

        size_t total_ids = req.dst_block_ids.size();
        size_t num_conns = connections.size();
        num_conns = num_conns > conn_per_req ? conn_per_req : num_conns;

        if (total_ids == 0 || num_conns == 0 || req.src_block_ids.empty()) {
            co_await upstream.async_send(
                boost::system::error_code{}, std::make_tuple(req.request_id, rank, true), asio::use_awaitable);
            co_return;
        }

        size_t base_count = total_ids / num_conns;
        size_t remainder = total_ids % num_conns;
        ox_trace("OX trace: request=", req.request_id, " event=shard_dispatch rank=", rank,
            " src_blocks=", req.src_block_ids.size(), " dst_blocks=", total_ids, " connections=", num_conns);
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

            size_t conn_index = last;
            ox_trace("OX trace: request=", req.request_id, " event=connection_dispatch rank=", rank,
                " connection=", conn_index, " blocks=", count);
            co_spawn(co_await asio::this_coro::executor,
                connections[conn_index]->submit_request(req.request_id, req.table_id, src_ids, dst_ids),
                detached);

            last = (last + 1) % connections.size();
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
                ox_trace("OX trace: request=", request_id, " event=shard_complete rank=", rank,
                    " completed_blocks=", ids.size());
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
    int last = 0;
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
    TPGroup(address_list_t &endpoints, Config &config, BlockTable &bt, ZMQChannel &channel)
        : block_size(config.block_size), downstream(config.get_io_context(), 128), bt(bt), upstream(channel),
          merger(62, 128, 704, endpoints.size())
    {
        size_t num_shards = endpoints.size();
        clusters.emplace_back();
        auto &vec = clusters.back();
        for (int rank = 0; rank < static_cast<int>(num_shards); rank++) {
            auto &ep = endpoints[rank];
            auto shard = std::make_shared<TPShard>(ep, rank, config, bt, downstream, num_shards);
            vec.push_back(shard);
            co_spawn(config.get_io_context(), shard->run(), detached);
        }
        std::cout << "TPGroup created dynamically with " << num_shards
                  << " shards, tp_override=" << num_shards
                  << " (config.tp_size()=" << config.tp_size() << ")" << std::endl;
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
                }

                if (rank >= 0 && static_cast<size_t>(rank) < rank_finished.size()) {
                    rank_finished[static_cast<size_t>(rank)] = true;
                }

                if (std::all_of(rank_finished.begin(), rank_finished.end(), [](bool b) { return b; })) {
                    ox_trace("OX trace: request=", request_id, " event=request_complete rank=", rank,
                        " blocks=", block_ids.size());
                    bool req_failed = failed;
                    client_id_t cid = client_id;
                    requests_status.erase(request_id);
                    requests_mutex.unlock();

                    co_await upstream.async_send(
                        boost::system::error_code{}, std::make_tuple(cid, request_id, !req_failed), asio::use_awaitable);
                } else {
                    requests_mutex.unlock();
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "Response sender stopped: " << e.what() << std::endl;
            print_ox_backtrace(std::cerr);
        }
        co_return;
    }

    asio::awaitable<void> gather(client_id_t client_id, RequestMessage &req)
    {
        int cid = 0;

        ox_trace("OX trace: request=", req.request_id, " event=request_dispatch cluster=", cid,
            " table=", req.table_id, " src_blocks=", req.src_block_ids.size(), " dst_blocks=",
            req.dst_block_ids.size(), " shards=", clusters[cid].size());

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

std::unordered_map<request_id_t, std::tuple<int, std::chrono::high_resolution_clock::time_point>> time_record;

// Dynamic connection pool: maps shard_list string -> TPGroup
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
                return it->second;
            }
        }

        std::unique_lock<std::shared_mutex> wlock(mutex);
        auto it = groups.find(shard_list_str);
        if (it != groups.end()) {
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
        if (ox_debug_logging_enabled()) {
            std::cout << "[DYNAMIC-TOPO] TPGroupManager: created new group for "
                      << shard_list_str << " (" << endpoints.size() << " shards)" << std::endl;
        }
        return group;
    }

private:
    mutable std::shared_mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<TPGroup>> groups;
    Config &config;
    BlockTable &bt;
    ZMQChannel &channel;
};

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

            global_stats_update_running(-1);
            co_await router_socket.async_send_multipart(std::move(response_messages));
            ox_trace("OX trace: request=", request_id, " event=response_sent success=", success);
        }
    } catch (const std::exception &e) {
        std::cerr << "Response sender stopped: " << e.what() << std::endl;
        print_ox_backtrace(std::cerr);
    }
}

asio::awaitable<void> router_receiver(ZmqCoroutineSocket &router_socket, TPGroupManager &manager)
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

                auto start = std::chrono::high_resolution_clock::now();
                time_record[request.request_id] = make_tuple(request.src_block_ids.size(), start);

                ox_trace("OX trace: request=", request.request_id, " event=request_received",
                    " remote_ox_shard_list=", request.remote_ox_shard_list,
                    " table=", request.table_id, " src_blocks=", request.src_block_ids.size(),
                    " dst_blocks=", request.dst_block_ids.size());

                global_stats_update_running(1);

                auto group = manager.get_or_create(request.remote_ox_shard_list);
                co_spawn(co_await asio::this_coro::executor, group->gather(client_id, request), detached);
            } else {
                std::cerr << "[DBG] router_receiver: Wrong msg size=" << msg->size() << std::endl;
                print_ox_backtrace(std::cerr);
            }
        } catch (const std::exception &e) {
            std::cerr << "[DBG] router_receiver error: " << e.what() << std::endl;
            print_ox_backtrace(std::cerr);
        }
    }
}

int main(int argc, char *argv[])
{
    std::cout << "Omni Xfer v0.7.5 (dynamic) starting..." << std::endl;
    try {
        Config config = parse_arguments(argc, argv);
        BlockTable bt(config);

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

        auto manager = std::make_shared<TPGroupManager>(config, bt, response_channel);

        std::string address = "tcp://*:" + std::to_string(config.zmq_port);
        zmq_router.bind(address);

        co_spawn(io_context, router_receiver(zmq_router, *manager), detached);
        co_spawn(io_context, response_sender(zmq_router, response_channel), detached);

        if (ox_statistics_log_enabled()) {
            co_spawn(io_context, print_statistics(), detached);
        }

        std::cout << "Omni Xfer D started. ZMQ: " << address << std::endl;

        std::vector<std::thread> threads;
        for (size_t i = 0; i < config.num_threads; ++i) {
            threads.emplace_back([&io_context]() { io_context.run(); });
        }

        io_context.run();

        for (auto &thread : threads)
            thread.join();
    } catch (const std::exception &e) {
        std::cerr << "main() Exception: " << e.what() << "\n";
        print_ox_backtrace(std::cerr);
        return 1;
    }

    return 0;
}

// g++ -std=c++20 -DNDEBUG -fcoroutines -I./  -g -march=native  ox.cpp -o ox -lzmq -lpthread
