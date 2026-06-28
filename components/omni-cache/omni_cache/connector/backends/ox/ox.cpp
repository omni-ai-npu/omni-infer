// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <set>
#include <atomic>
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

struct RequestMessage {
    request_id_t request_id;
    table_id_t table_id;
    block_list_t src_block_ids;  // P/server side block ids
    block_list_t dst_block_ids;  // D/client side block ids
    int cluster_id{0};           // chosen cluster index（0-based）
    int src_dp_rank{0};          // remote DP rank — ox applies offset internally
    MSGPACK_DEFINE_MAP(request_id, table_id, src_block_ids, dst_block_ids, cluster_id, src_dp_rank)
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

using GroupMessage = std::tuple<client_id_t, int>;
using GroupChannel = concurrent_channel<asio::any_io_executor, void(boost::system::error_code, GroupMessage)>;

class CoroutineConnection {
public:
    CoroutineConnection(asio::io_context &io_context, boost::asio::ip::tcp::endpoint addr, Config &config, int rank,
        BlockTable &bt, ShardChannel &response)
        : socket(io_context), config(config), rank(rank), addr(addr), bt(bt), request(config.get_io_context(), 128),
          upstream(response)
    {}

    void start()
    {
        asio::co_spawn(socket.get_executor(), run(), asio::detached);
    }

    asio::awaitable<void> run()
    {
        try {
            co_await socket.async_connect(addr, asio::use_awaitable);
            optimize_tcp_socket(socket);

            while (true) {
                auto [request_id, table_id, src_ids, dst_ids, src_dp_rank] = co_await request.async_receive(asio::use_awaitable);

                if (src_ids.empty()) {
                    // no task, return empty immediately
                    co_await upstream.async_send(
                        boost::system::error_code{}, std::make_tuple(request_id, dst_ids), asio::use_awaitable);
                    continue;
                }

                // Apply DP offset so Python doesn't need block shapes early.
                if (src_dp_rank != 0) {
                    int64_t offset = static_cast<int64_t>(config.num_blocks) * src_dp_rank;
                    for (auto &id : src_ids) {
                        id += offset;
                    }
                }

                // Target buffer follows dst_ids; send src_ids to P-side server.
                // auto bufs = bt.get_buffers(table_id, dst_ids, rank);

                // auto bufs = bt.get_buffers_interleaved(table_id, dst_ids, rank);
                auto bufs = bt.get_buffers_layerwise(table_id, dst_ids, rank);
                // for (auto id : src_ids) {
                //     std::cout << "Request for ID:" << id << std::endl;
                // }
                // for (auto id : dst_ids) {
                //     std::cout << "Save to ID:" << id << std::endl;
                // }
                // std::cout << "Buf size:" << bufs.size()/dst_ids.size() << std::endl;

                co_await (asio::async_write(socket,
                              asio::buffer(src_ids.data(), src_ids.size() * sizeof(block_id_t)),
                              asio::use_awaitable) &&
                          asio::async_read(socket, bufs, asio::use_awaitable));

                // #ifdef CONTENT_CHECK
                //                 for (size_t i = 0; i < src_ids.size(); i++)
                //                 {
                //                     int64_t *data = static_cast<int64_t *>(bufs[i].data());
                //                     std::cerr << "Check content: " << *data << " : " << src_ids[i] << "\n";
                //                     assert(*data == src_ids[i]);
                //                 }
                // #endif

                global_stats_update(dst_ids.size() * bt.block_tp_size());

                // return finished dst_ids
                co_await upstream.async_send(
                    boost::system::error_code{}, std::make_tuple(request_id, dst_ids), asio::use_awaitable);
            }
        } catch (const std::exception &e) {
            std::cerr << "Connection " << addr.address() << ":" << addr.port() << " error: " << e.what() << "\n";
        }
    }

    asio::awaitable<void> submit_request(
        std::string &request_id, table_id_t table_id, block_list_t &src_block_ids, block_list_t &dst_block_ids, int src_dp_rank)
    {
        co_await request.async_send(boost::system::error_code{},
            std::make_tuple(request_id, table_id, src_block_ids, dst_block_ids, src_dp_rank),
            asio::use_awaitable);
    }

private:
    tcp::socket socket;
    const Config &config;
    int rank;
    boost::asio::ip::tcp::endpoint addr;

    BlockTable &bt;
    ConnectionChannel request;
    ShardChannel &upstream;
};

static bool try_connect_once_with_timeout(const boost::asio::ip::tcp::endpoint &ep, std::chrono::seconds timeout)
{
    using tcp = boost::asio::ip::tcp;

    boost::asio::io_context ioc;
    tcp::socket sock(ioc);
    boost::asio::steady_timer timer(ioc);

    std::atomic<bool> connected{false};

    sock.async_connect(ep, [&](const boost::system::error_code &ec) {
        if (!ec)
            connected = true;
        timer.cancel();
    });

    timer.expires_after(timeout);
    timer.async_wait([&](const boost::system::error_code &ec) {
        if (!ec) {
            boost::system::error_code ignored;
            sock.cancel(ignored);
        }
    });

    ioc.run();

    boost::system::error_code ignored;
    sock.close(ignored);
    return connected.load();
}

class TPShard {
public:
    TPShard(boost::asio::ip::tcp::endpoint addr, int rank, Config &config, BlockTable &bt, GroupChannel &channel)
        : ip(addr), rank(rank), downstream(config.get_io_context(), 128), upstream(channel)
    {
        using namespace std::chrono;
        const seconds overall_timeout{3600};
        const seconds retry_interval{5};

        conn_per_req = config.connections_per_req;

        for (std::size_t i = 0; i < config.connections_per_shard; ++i) {
            const auto deadline = steady_clock::now() + overall_timeout;

            for (;;) {
                std::cout << "Try: to build connection with P server " << addr << "..." << std::endl;
                if (try_connect_once_with_timeout(ip, retry_interval)) {
                    break;
                }
                if (steady_clock::now() >= deadline) {
                    throw std::runtime_error(
                        "connect timeout after 3600s to " + ip.address().to_string() + ":" + std::to_string(ip.port()));
                }
                std::this_thread::sleep_for(retry_interval);
            }

            std::cout << "Successfully build connection with P server " << addr << "." << std::endl;
            connections.emplace_back(
                std::make_shared<CoroutineConnection>(config.get_io_context(), ip, config, rank, bt, downstream));
            connections[i]->start();
        }
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

        if (total_ids == 0 || num_conns == 0) {
            co_await upstream.async_send(
                boost::system::error_code{}, std::make_tuple(req.request_id, rank), asio::use_awaitable);
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

            size_t conn_index = last;
            co_spawn(co_await asio::this_coro::executor,
                connections[conn_index]->submit_request(req.request_id, req.table_id, src_ids, dst_ids, req.src_dp_rank),
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
                    boost::system::error_code{}, std::make_tuple(request_id, rank), asio::use_awaitable);
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
    TPGroup(Config &config, BlockTable &bt, ZMQChannel &channel)
        : block_size(config.block_size), downstream(config.get_io_context(), 128), bt(bt), upstream(channel),
          merger(62, 128, 704, config.tp_size())
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
                    auto &ep = config.shard_clusters[c][j];
                    auto shard = std::make_shared<TPShard>(ep, static_cast<int>(j), config, bt, downstream);
                    vec.push_back(shard);
                    co_spawn(config.get_io_context(), shard->run(), detached);
                }
            }
        } else {
            clusters.emplace_back();
            auto &vec = clusters.back();
            for (int rank = 0; rank < static_cast<int>(config.shard_list.size()); rank++) {
                auto &ep = config.shard_list[rank];
                auto shard = std::make_shared<TPShard>(ep, rank, config, bt, downstream);
                vec.push_back(shard);
                co_spawn(config.get_io_context(), shard->run(), detached);
            }
        }
    }

    asio::awaitable<void> merge_and_response(request_id_t request_id)
    {
        auto ex = co_await boost::asio::this_coro::executor;
        std::vector<boost::asio::awaitable<void>> tasks;

        requests_mutex.lock();
        auto it = requests_status.find(request_id);
        if (it == requests_status.end()) {
            std::cout << "unexpected invalid request: " << request_id << std::endl;
            requests_mutex.unlock();
            co_return;
        }
        auto &[client_id, table_id, rank_finished, block_ids] = it->second;
        requests_mutex.unlock();

        bool failed = false;

        for (auto block_id : block_ids) {
            tasks.push_back(boost::asio::co_spawn(
                ex,
                [table_id, block_id, request_id, &failed, this]() -> boost::asio::awaitable<void> {
                    void *ptr = bt.block_addr(table_id, block_id);

                    // auto start = std::chrono::high_resolution_clock::now();
                    void *buf = malloc(this->block_size);
                    if (buf == nullptr) {
                        failed = true;
                        std::cout << "Failed allocate buffer: " << request_id << std::endl;
                        co_return;
                    }
                    memcpy(buf, ptr, this->block_size);

                    // auto duration1 =
                    // std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                    // start).count(); auto ms =
                    // std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch()).count();
                    this->merger.merge_shards(static_cast<const short *>(buf), static_cast<short *>(ptr));
                    free(buf);
                    // std::cout << pthread_self() << std::endl;
                    co_return;
                },
                boost::asio::use_awaitable));
        }

        for (auto &task : tasks) {
            co_await std::move(task);
        }

        requests_mutex.lock();
        client_id_t cid = client_id;
        requests_status.erase(request_id);
        requests_mutex.unlock();

        co_await upstream.async_send(
            boost::system::error_code{}, std::make_tuple(cid, request_id, !failed), asio::use_awaitable);
    }

    asio::awaitable<void> run()
    {
        try {
            while (true) {
                auto [request_id, rank] = co_await downstream.async_receive(asio::use_awaitable);

                // rank is the index within the cluster;
                // the corresponding bit in the cluster's completion bitmap was already set when the request was logged.
                requests_mutex.lock();
                auto it = requests_status.find(request_id);
                if (it == requests_status.end()) {
                    requests_mutex.unlock();
                    continue;
                }
                auto &[client_id, table_id, rank_finished, block_ids] = it->second;

                if (rank >= 0 && static_cast<size_t>(rank) < rank_finished.size()) {
                    rank_finished[static_cast<size_t>(rank)] = true;
                }

                // std::cout << "Finished to pull kv for request " << request_id << " from TP-rank-" << rank <<
                // std::endl;

                if (std::all_of(rank_finished.begin(), rank_finished.end(), [](bool b) { return b; })) {
                    requests_status.erase(request_id);
                    requests_mutex.unlock();

                    client_id_t cid = client_id;
                    std::cout << "<<<<<<< Send response: " << request_id << std::endl;
                    co_await upstream.async_send(
                        boost::system::error_code{}, std::make_tuple(cid, request_id, true), asio::use_awaitable);
                    // requests_mutex.unlock();
                    // co_spawn(co_await asio::this_coro::executor, merge_and_response(request_id), detached);
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
        // Select cluster; fall back to 0 if out-of-bounds.
        int cid = req.cluster_id;
        if (cid < 0 || static_cast<size_t>(cid) >= clusters.size())
            cid = 0;

        {
            std::unique_lock<std::shared_mutex> lock(requests_mutex);
            // Size the completion bitmap to the number of shards in the selected cluster.
            size_t shard_cnt = clusters[cid].size();
            requests_status[req.request_id] =
                std::make_tuple(client_id, req.table_id, std::vector<bool>(shard_cnt, false), req.dst_block_ids);
        }

        // Send requests only to the shards of the selected cluster.
        for (auto &shard : clusters[cid]) {
            co_spawn(co_await asio::this_coro::executor, shard->gather(req), detached);
        }
    }

public:
    mutable std::shared_mutex requests_mutex;
    // request_id → (client_id, completion bitmap [within cluster])
    std::unordered_map<request_id_t, std::tuple<client_id_t, table_id_t, std::vector<bool>, std::vector<block_id_t>>>
        requests_status;

    // clusters[c] holds the TPShard instances within that cluster.
    std::vector<std::vector<std::shared_ptr<TPShard>>> clusters;

    const size_t block_size;
    GroupChannel downstream;
    BlockTable &bt;
    ZMQChannel &upstream;
    KVCacheMerger merger;
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

asio::awaitable<void> router_receiver(ZmqCoroutineSocket &router_socket, TPGroup &group)
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

                // std::cout << "Start to pull kv for request " << request.request_id << std::endl;
                global_stats_update_running(1);

                co_spawn(co_await asio::this_coro::executor, group.gather(client_id, request), detached);
            } else {
                // std::cout << "Wrong msg: " << msg->size() << std::endl;
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

        std::shared_ptr<TPGroup> tp_group;
        ZMQChannel response_channel(io_context, 128);
        ZmqCoroutineSocket zmq_router(ZMQ_ROUTER, io_context);

        if (!config.shard_list.empty() || !config.shard_clusters.empty()) {
            tp_group = std::make_shared<TPGroup>(config, bt, response_channel);

            std::string address = "tcp://*:" + std::to_string(config.zmq_port);
            zmq_router.bind(address);

            co_spawn(io_context, tp_group->run(), detached);
            co_spawn(io_context, router_receiver(zmq_router, *tp_group), detached);
            co_spawn(io_context, response_sender(zmq_router, response_channel), detached);
            co_spawn(io_context, print_statistics(), detached);

            std::cout << "Omni Xfer started. ZMQ: " << address << std::endl;
        }

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
