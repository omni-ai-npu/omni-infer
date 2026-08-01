// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <boost/asio.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/experimental/concurrent_channel.hpp>
#include <boost/asio/experimental/awaitable_operators.hpp>

#include <arpa/inet.h>
#include <execinfo.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <ox_block_table.hpp>

namespace asio = boost::asio;
using asio::co_spawn;
using asio::detached;
using asio::use_awaitable;
using asio::experimental::concurrent_channel;
using asio::ip::tcp;
using namespace boost::asio::experimental::awaitable_operators;

inline constexpr uint32_t k_ox_max_block_ids_per_message = 8192;

inline void print_ox_backtrace(std::ostream &out)
{
    void *frames[64];
    const int frame_count = backtrace(frames, static_cast<int>(sizeof(frames) / sizeof(frames[0])));
    char **symbols = backtrace_symbols(frames, frame_count);

    out << "OX backtrace (" << frame_count << " frames):" << std::endl;
    if (symbols == nullptr) {
        out << "  backtrace_symbols failed" << std::endl;
        return;
    }
    for (int i = 0; i < frame_count; ++i) {
        out << "  " << symbols[i] << std::endl;
    }
    std::free(symbols);
}

inline void optimize_tcp_socket(tcp::socket &socket)
{
    boost::system::error_code ec;

    socket.set_option(tcp::no_delay(true), ec);
    if (ec)
        std::cerr << "Failed to set TCP_NODELAY: " << ec.message() << std::endl;

#ifdef TCP_QUICKACK
    int quickack = 1;
    setsockopt(socket.native_handle(), IPPROTO_TCP, TCP_QUICKACK, &quickack, sizeof(quickack));
#endif
}

class Session : public std::enable_shared_from_this<Session> {
public:
    Session(tcp::socket socket, BlockTable &bt) : socket_(std::move(socket)), bt(bt)
    {}

    void start()
    {
        asio::co_spawn(
            socket_.get_executor(),
            [self = shared_from_this()]() -> asio::awaitable<void> { co_await self->process_connection(); },
            asio::detached);
    }

private:
    asio::awaitable<void> process_connection()
    {
        const char *phase = "read_block_count";
        uint32_t block_count = 0;
        try {
            while (true) {
                uint32_t block_count_network = 0;
                phase = "read_block_count";
                co_await asio::async_read(
                    socket_, asio::buffer(&block_count_network, sizeof(block_count_network)), use_awaitable);

                block_count = ntohl(block_count_network);
                if (block_count == 0 || block_count > k_ox_max_block_ids_per_message) {
                    throw std::out_of_range("invalid block count: " + std::to_string(block_count));
                }

                block_list_t block_list(block_count);
                phase = "read_block_ids";
                co_await asio::async_read(socket_,
                    asio::buffer(block_list.data(), block_list.size() * sizeof(block_id_t)), use_awaitable);

                phase = "write_kv";
                co_await asio::async_write(socket_, bt.get_buffers_layerwise(0, block_list, 0), use_awaitable);
            }
        } catch (const boost::system::system_error &e) {
            std::cerr << "OX P transport error: peer=" << peer() << " phase=" << phase
                      << " block_count=" << block_count << " ec=" << e.code().value()
                      << " category=" << e.code().category().name() << " message=" << e.code().message()
                      << std::endl;
            print_ox_backtrace(std::cerr);
        } catch (const std::exception &e) {
            std::cerr << "OX P request error: peer=" << peer() << " phase=" << phase
                      << " block_count=" << block_count << " message=" << e.what() << std::endl;
            print_ox_backtrace(std::cerr);
        }
    }

    std::string peer() const
    {
        boost::system::error_code ec;
        const auto endpoint = socket_.remote_endpoint(ec);
        if (ec) {
            return "unavailable(" + ec.message() + ")";
        }
        return endpoint.address().to_string() + ":" + std::to_string(endpoint.port());
    }

    tcp::socket socket_;
    BlockTable &bt;
    int rank = 0;
};

class Server {
public:
    Server(asio::io_context &io_context, boost::asio::ip::tcp::endpoint &ep, BlockTable &bt)
        : acceptor_(io_context, ep), bt(bt)
    {
        std::cout << "OX Listening Port:" << ep.port() << std::endl;
    }

    uint16_t port() const
    {
        return acceptor_.local_endpoint().port();
    }

    asio::awaitable<void> run()
    {
        while (true) {
            try {
                tcp::socket socket = co_await acceptor_.async_accept(asio::use_awaitable);

                optimize_tcp_socket(socket);

                std::cout << "New connection from: " << socket.remote_endpoint().address().to_string() << ":"
                          << socket.remote_endpoint().port() << std::endl;

                std::make_shared<Session>(std::move(socket), bt)->start();
            } catch (const boost::system::system_error &e) {
                std::cerr << "Accept error: " << e.what() << std::endl;
                if (e.code() == boost::asio::error::operation_aborted) {
                    break;
                }
            }
        }
        co_return;
    }

private:
    tcp::acceptor acceptor_;
    BlockTable &bt;
};
