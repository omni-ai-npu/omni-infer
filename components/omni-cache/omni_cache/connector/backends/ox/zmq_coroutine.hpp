// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <zmq.hpp>

#include <boost/asio.hpp>
#include <boost/asio/experimental/concurrent_channel.hpp>
#include <boost/asio/experimental/as_tuple.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace asio = boost::asio;
using asio::awaitable;
using asio::use_awaitable;

class ZmqCoroutineSocket {
public:
    using Multipart = std::vector<zmq::message_t>;
    static constexpr std::size_t kChanCapicity = 4096;

    ZmqCoroutineSocket(int type, asio::io_context &io_ctx)
        : _ctx(1),
          _sock(_ctx, static_cast<zmq::socket_type>(type)),
          _exec(io_ctx.get_executor()),
          _recv_ch(_exec, kChanCapicity),
          _send_ch(_exec, kChanCapicity),
          _ctrl_tx(_ctx, zmq::socket_type::pair)
    {
        _sock.set(zmq::sockopt::linger, 0);
        _sock.set(zmq::sockopt::sndhwm, 10000);
        _sock.set(zmq::sockopt::rcvhwm, 10000);
        _ctrl_tx.set(zmq::sockopt::linger, 0);

        _ctrl_ep = "inproc://ox_zmq_coro_ctrl_" +
                   std::to_string(reinterpret_cast<std::uintptr_t>(this));
        _ctrl_tx.bind(_ctrl_ep);
    }

    ~ZmqCoroutineSocket() {
        stop();
    }

    void bind(const std::string& address) {
        _sock.bind(address);
        start_worker_once();
    }

    asio::awaitable<std::optional<Multipart>> async_recv_multipart() {
        auto [ec, msg] =
            co_await _recv_ch.async_receive(asio::experimental::as_tuple(asio::use_awaitable));
        if (ec) co_return std::nullopt;
        co_return std::move(msg);
    }

    asio::awaitable<bool> async_send_multipart(Multipart msg) {
        auto [ec] =
            co_await _send_ch.async_send(
                boost::system::error_code{},             
                std::move(msg),                           
                asio::experimental::as_tuple(asio::use_awaitable));
        if (ec) co_return false;
        co_return true;
    }

private:
    void start_worker_once() {
        bool expected = false;
        if (!_started.compare_exchange_strong(expected, true)) return;
        _worker = std::thread([this]() { worker_loop(); });
    }

    void stop() {
        if (!_started.load()) return;

        bool expected = false;
        if (!_stopping.compare_exchange_strong(expected, true)) return;

        _stop_flag.store(true);

        // Wake worker thread so it exits its poll loop
        try {
            zmq::message_t ping(1);
            *static_cast<std::uint8_t*>(ping.data()) = 0;
            (void)_ctrl_tx.send(ping, zmq::send_flags::dontwait);
        } catch (...) {}

        if (_worker.joinable()) _worker.join();

        _recv_ch.try_send(asio::error::operation_aborted, Multipart{});

        // Shutdown order: context first, then sockets.
        // Closing a socket while its context's internal I/O thread is still
        // active triggers libzmq assert(false) in object.cpp.  Calling
        // _ctx.shutdown() first terminates the internal I/O thread, making
        // subsequent socket closes safe.
        try { _ctx.shutdown(); } catch (...) {}
        try { _sock.close(); } catch (...) {}
        try { _ctrl_tx.close(); } catch (...) {}
        try { _ctx.close(); } catch (...) {}
    }

    static bool recv_multipart_blocking(zmq::socket_t& s, Multipart& out) {
        out.clear();
        out.reserve(8);

        zmq::message_t part;
        auto r = s.recv(part, zmq::recv_flags::none);
        if (!r.has_value()) return false;
        out.push_back(std::move(part));

        while (s.get(zmq::sockopt::rcvmore)) {
            zmq::message_t more;
            auto rr = s.recv(more, zmq::recv_flags::none);
            if (!rr.has_value()) return false;
            out.push_back(std::move(more));
        }
        return true;
    }

    static bool send_multipart_nonblock(
        zmq::socket_t& s,
        Multipart& msg,
        std::size_t& index_inout)
    {
        while (index_inout < msg.size()) {
            const bool more = (index_inout + 1 < msg.size());
            zmq::send_flags flags =
                more ? zmq::send_flags::sndmore : zmq::send_flags::none;

            zmq::message_t copy(msg[index_inout].data(), msg[index_inout].size());
            auto r = s.send(copy, flags | zmq::send_flags::dontwait);
            if (!r.has_value()) {
                if (errno == EAGAIN) return false;
                return false;
            }
            ++index_inout;
        }
        return true;
    }

    void worker_loop() {
        zmq::socket_t ctrl_rx(_ctx, zmq::socket_type::pair);
        ctrl_rx.set(zmq::sockopt::linger, 0);
        ctrl_rx.connect(_ctrl_ep);

        std::deque<Multipart> send_queue;
        std::optional<Multipart> sending;
        std::size_t sending_idx = 0;

        Multipart incoming;

        zmq::pollitem_t items[] = {
            { _sock,   0, static_cast<short>(ZMQ_POLLIN | ZMQ_POLLOUT), 0 },
            { ctrl_rx, 0, static_cast<short>(ZMQ_POLLIN), 0 }
        };

        while (!_stop_flag.load()) {
            for (;;) {
                bool got_one = _send_ch.try_receive([&](boost::system::error_code ec, Multipart m) {
                    if (!ec) send_queue.emplace_back(std::move(m));
                });
                if (!got_one) break;
            }

            int timeout_ms = (!send_queue.empty() || sending.has_value()) ? 0 : -1;
            if (timeout_ms < 0) {
                zmq::poll(items, 2, std::chrono::milliseconds{-1});
            } else {
                zmq::poll(items, 2, std::chrono::milliseconds{timeout_ms});
            }

            if (items[1].revents & ZMQ_POLLIN) {
                zmq::message_t dummy;
                (void)ctrl_rx.recv(dummy, zmq::recv_flags::dontwait);
                break;
            }

            if (items[0].revents & ZMQ_POLLIN) {
                try {
                    if (recv_multipart_blocking(_sock, incoming) && !incoming.empty()) {
                        _recv_ch.try_send(boost::system::error_code{}, std::move(incoming));
                        incoming.clear();
                    }
                } catch (...) {
                    _recv_ch.try_send(asio::error::operation_aborted, Multipart{});
                    break;
                }
            }

            if (items[0].revents & ZMQ_POLLOUT) {
                try {
                    if (!sending && !send_queue.empty()) {
                        sending = std::move(send_queue.front());
                        send_queue.pop_front();
                        sending_idx = 0;
                    }

                    while (sending) {
                        bool done = send_multipart_nonblock(_sock, *sending, sending_idx);
                        if (!done) break;

                        sending.reset();
                        sending_idx = 0;

                        if (!send_queue.empty()) {
                            sending = std::move(send_queue.front());
                            send_queue.pop_front();
                        }
                    }
                } catch (...) {
                    _recv_ch.try_send(asio::error::operation_aborted, Multipart{});
                    break;
                }
            }
        }

        try { ctrl_rx.close(); } catch (...) {}
    }

private:
    zmq::context_t _ctx;
    zmq::socket_t _sock;

    asio::any_io_executor _exec;
    asio::experimental::concurrent_channel<
        asio::any_io_executor,
        void(boost::system::error_code, Multipart)
    > _recv_ch;

    asio::experimental::concurrent_channel<
        asio::any_io_executor,
        void(boost::system::error_code, Multipart)
    > _send_ch;

    zmq::socket_t _ctrl_tx;
    std::string   _ctrl_ep;

    std::thread _worker;
    std::atomic<bool> _started{false};
    std::atomic<bool> _stopping{false};
    std::atomic<bool> _stop_flag{false};
};