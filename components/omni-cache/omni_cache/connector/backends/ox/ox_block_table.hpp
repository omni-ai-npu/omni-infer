// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cassert>
#include <cerrno>
#include <system_error>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <functional>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <boost/asio/buffer.hpp>

#include "ox_config.hpp"

namespace asio = boost::asio;
using asio::co_spawn;
using asio::detached;
using asio::experimental::concurrent_channel;
using asio::ip::tcp;

using request_id_t = std::string;
using client_id_t = std::string;
using block_id_t = int64_t;
using block_list_t = std::vector<block_id_t>;
using table_id_t = int64_t;

class BlockTable : public std::enable_shared_from_this<BlockTable> {
public:
    explicit BlockTable(Config &config) : num_blocks(config.num_blocks), config(config)
    {
        const std::string &path = config.block_table_shm;

        // check block_table_shm: shm path or bin file path
        const bool path_ok = path_exists(path);
        if (!path_ok) {
            open_from_file_or_fallback_shm(path);
        } else if (is_directory(path)) {
            // if block_table_shm is a folder path --> go shm_open
            std::string shm_name = "/" + basename_of(path);
            open_posix_shm(shm_name);
        } else if (is_regular_file(path)) {
            // if block_table_shm is a file path --> go file open
            open_from_file_or_fallback_shm(path);
        } else {
            throw std::system_error(
                EINVAL, std::generic_category(), "block_table_shm is neither a directory nor a regular file");
        }

        //[(62, 1024, 128, 512), (62, 1024, 128, 64), (62, 1024, 128, 128)]
        for (auto dim : config.block_dims) {
            blocks.emplace_back(shape_t{config.num_layers, num_blocks, config.num_tokens_per_block, dim});
        }

        std::transform(blocks.begin(), blocks.end(), std::back_inserter(block_offset), [&config](const shape_t &block) {
            return std::accumulate(block.begin(), block.end(), config.element_size, std::multiplies<size_t>());
        });

        std::exclusive_scan(block_offset.begin(), block_offset.end(), block_offset.begin(), 0ULL);

        std::transform(
            blocks.begin(), blocks.end(), std::back_inserter(block_layer_size), [&config](const shape_t &block) {
                return std::accumulate(block.begin() + 2, block.end(), config.element_size, std::multiplies<size_t>());
            });

        std::transform(blocks.begin(), blocks.end(), std::back_inserter(layer_size), [&config](const shape_t &block) {
            return std::accumulate(block.begin() + 1, block.end(), config.element_size, std::multiplies<size_t>());
        });

        for (size_t i = 0; i < blocks.size(); i++) {
            std::cout << "Block: " << i << "  offset: " << block_offset[i]
                      << " block_layer_size: " << block_layer_size[i] << " layer_size: " << layer_size[i] << std::endl;
        }

        maybe_pin_as_leader();
    }

    ~BlockTable()
    {
        if (base && base != MAP_FAILED) {
            if (did_pin_) {
                ::munlock(base, map_size);
            }
            ::munmap(base, map_size);
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    inline int max_blocks() const noexcept
    {
        return num_blocks;
    }

    inline char *table_addr(table_id_t table_id)
    {
        assert(table_id < config.num_block_tables);
        return static_cast<char *>(base) + table_id * static_cast<ptrdiff_t>(config.table_size);
    }

    inline void *block_addr(table_id_t table_id, size_t block_id)
    {
        assert(block_id < static_cast<size_t>(num_blocks));
        char *ptr = static_cast<char *>(base) + table_id * static_cast<ptrdiff_t>(config.block_table_size()) +
                    block_id * static_cast<ptrdiff_t>(config.block_size);

        assert(ptr >= static_cast<char *>(base));
        assert(ptr + config.block_size <= static_cast<char *>(base) + map_size);
        return ptr;
    }

    inline size_t block_tp_size()
    {
        return config.block_size / config.tp_size();
    }

    inline void *block_tp_addr(table_id_t table_id, size_t block_id, int rank)
    {
        return static_cast<char *>(block_addr(table_id, block_id)) + block_tp_size() * rank;
    }

    std::vector<boost::asio::mutable_buffer> get_buffers(table_id_t table_id, block_list_t &block_ids, int rank)
    {
        std::vector<boost::asio::mutable_buffer> buffers;
        buffers.reserve(block_ids.size());

        for (auto id : block_ids) {
            buffers.emplace_back(block_tp_addr(table_id, static_cast<size_t>(id), rank), block_tp_size());
        }
        return buffers;
    }

    std::vector<boost::asio::mutable_buffer> get_buffers_layerwise(
        table_id_t table_id, block_list_t &block_ids, int rank)
    {
        char *ptr = table_addr(table_id);
        size_t tp_size = config.tp_size();

        std::vector<boost::asio::mutable_buffer> buffers;
        for (auto block_id : block_ids) {
            for (size_t i = 0; i < blocks.size(); i++) {  // blocks: ..512, ..64, ..128
                char *base = ptr + block_offset[i];
                for (size_t j = 0; j < blocks[i][0]; j++) {  // layers
                    void *layer = base + j * layer_size[i] + block_id * block_layer_size[i] +
                                  (block_layer_size[i] * rank) / tp_size;
                    buffers.emplace_back(layer, block_layer_size[i] / tp_size);
                    // std::cout << "block id: " << block_id << " segment: " << i << " layer: " << j
                    //           << " addr: " << (((char *)layer) - ptr) << " size:" << block_layer_size[i] / tp_size
                    //           << std::endl;
                }
            }
        }

        return buffers;
    }

    std::vector<boost::asio::mutable_buffer> get_buffers_interleaved(
        table_id_t table_id, block_list_t &block_ids, int rank)
    {
        std::vector<boost::asio::mutable_buffer> buffers;
        size_t num_layers = 62;
        size_t tokens_per_block = 128;
        size_t dtype_size = 2;
        size_t num_tp = config.tp_size();

        std::vector<size_t> layer_sizes = {512 * dtype_size * tokens_per_block,
            64 * dtype_size * tokens_per_block,
            128 * dtype_size * tokens_per_block};
        std::vector<size_t> segment_offset = {0};

        size_t offset = 0;
        for (auto layer_size : layer_sizes) {
            offset += layer_size * num_layers;
            segment_offset.emplace_back(offset);
        }

        buffers.reserve(block_ids.size() * num_layers * layer_sizes.size());

        for (auto id : block_ids) {
            char *base = static_cast<char *>(block_addr(table_id, id));
            for (size_t i = 0; i < layer_sizes.size(); i++) {
                size_t segment_len = layer_sizes[i] / num_tp;
                char *ptr = base + segment_offset[i] + segment_len * rank;

                for (size_t layer = 0; layer < num_layers; layer++) {
                    buffers.emplace_back(ptr, segment_len);
                    ptr += layer_sizes[i];
                }
            }
        }
        return buffers;
    }

    // inline size_t block_L_count(size_t D_dim) const {
    //     assert(D_dim > 0);
    //     assert(config.block_size % D_dim == 0);
    //     return config.block_size / D_dim;
    // }

    // inline size_t block_rank_slice_size(size_t D_dim) const {
    //     assert(config.tp_size() > 0);
    //     assert(D_dim % static_cast<size_t>(config.tp_size()) == 0);
    //     return D_dim / static_cast<size_t>(config.tp_size());
    // }

    // std::vector<boost::asio::mutable_buffer>
    // get_buffers(table_id_t table_id, block_list_t& block_ids, int rank, size_t D_dim)
    // {
    //     std::vector<boost::asio::mutable_buffer> buffers;

    //     const size_t L = block_L_count(D_dim);
    //     const size_t slice_sz = block_rank_slice_size(D_dim);

    //     buffers.reserve(block_ids.size() * L);

    //     for (auto id : block_ids) {
    //         char* block_start = static_cast<char*>(block_addr(table_id, static_cast<size_t>(id)));

    //         for (size_t l = 0; l < L; ++l) {
    //             char* chunk_start = block_start + l * D_dim;
    //             char* rank_chunk_start = chunk_start + rank * slice_sz;

    //             assert(rank_chunk_start >= static_cast<char*>(base));
    //             assert(rank_chunk_start + slice_sz <= static_cast<char*>(base) + map_size);

    //             buffers.emplace_back(rank_chunk_start, slice_sz);
    //         }
    //     }
    //     return buffers;
    // }

private:
    //     static bool try_mlock_onfault(void* addr, size_t len) {
    // // #if defined(__linux__) && defined(MLOCK_ONFAULT)
    //         return (::mlock2(addr, len, MLOCK_ONFAULT) == 0);
    // // #else
    // //         (void)addr; (void)len; return false;
    // // #endif
    //     }

    static void mlock_in_chunks(void *addr, size_t len, size_t chunk = (64ULL << 20))
    {
        auto *p = static_cast<unsigned char *>(addr);
        for (size_t off = 0; off < len; off += chunk) {
            size_t now = std::min(chunk, len - off);
            if (::mlock(p + off, now) != 0) {
            }
        }
    }

    bool try_become_mlock_leader_by_fd(int fd_)
    {
        struct flock fl {};
        fl.l_type = F_WRLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start = 0;
        fl.l_len = 1;
        return (::fcntl(fd_, F_SETLK, &fl) == 0);
    }

    void maybe_pin_as_leader()
    {
        if (fd < 0)
            return;

        constexpr int kRetries = 50;
        constexpr int kSleepMs = 20;
        for (int i = 0; i < kRetries; ++i) {
            if (try_become_mlock_leader_by_fd(fd)) {
                is_leader_ = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMs));
        }

        if (!is_leader_) {
            return;
        }

        // if (try_mlock_onfault(base, map_size)) {
        //     did_pin_ = true;
        //     return;
        // }

        mlock_in_chunks(base, map_size, 64ULL << 20);
        did_pin_ = true;
    }

private:
    static bool path_exists(const std::string &p)
    {
        struct stat st {};
        return ::stat(p.c_str(), &st) == 0;
    }

    static bool is_directory(const std::string &p)
    {
        struct stat st {};
        if (::stat(p.c_str(), &st) != 0)
            return false;
        return S_ISDIR(st.st_mode);
    }

    static bool is_regular_file(const std::string &p)
    {
        struct stat st {};
        if (::stat(p.c_str(), &st) != 0)
            return false;
        return S_ISREG(st.st_mode);
    }

    static std::string basename_of(const std::string &path)
    {
        if (path.empty())
            return {};
        auto pos = path.find_last_of('/');
        if (pos == std::string::npos)
            return path;
        if (pos + 1 >= path.size())
            return {};
        return path.substr(pos + 1);
    }

    void open_from_file_or_fallback_shm(const std::string &filepath)
    {
        std::string base = basename_of(filepath);
        if (base.empty()) {
            throw std::system_error(EINVAL, std::generic_category(), "invalid path for deriving shm name");
        }
        std::string shm_name = "/" + base;

        open_file_backed(filepath);
    }

    void open_posix_shm(const std::string &name)
    {
        fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd >= 0) {
            map_size = config.shm_mem_size();
            if (::ftruncate(fd, static_cast<off_t>(map_size)) != 0) {
                int e = errno;
                ::close(fd);
                fd = -1;
                throw std::system_error(e, std::generic_category(), "ftruncate(shm) failed");
            }
        } else {
            if (errno == EEXIST) {
                fd = ::shm_open(name.c_str(), O_RDWR, 0);
                if (fd < 0) {
                    throw std::system_error(errno, std::generic_category(), "shm_open open failed");
                }
                map_size = config.shm_mem_size();
            } else {
                throw std::system_error(errno, std::generic_category(), "shm_open create failed");
            }
        }
        map_fd_shared();
    }

    void open_file_backed(const std::string &filepath)
    {
        const size_t desired = config.shm_mem_size();

        int flags = O_RDWR | O_CREAT;
        fd = ::open(filepath.c_str(), flags, 0600);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open file failed");
        }

        struct stat st {};
        size_t current = 0;
        if (::fstat(fd, &st) == 0 && S_ISREG(st.st_mode)) {
            current = static_cast<size_t>(st.st_size);
        }

        if (current < desired) {
            if (::ftruncate(fd, static_cast<off_t>(desired)) != 0) {
                int e = errno;
                ::close(fd);
                fd = -1;
                throw std::system_error(e, std::generic_category(), "ftruncate(file) failed");
            }
        }
        map_size = desired;
        map_fd_shared();
    }

    void map_fd_shared()
    {
        base = ::mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            int e = errno;
            ::close(fd);
            fd = -1;
            throw std::system_error(e, std::generic_category(), "mmap failed");
        }
    }

private:
    std::vector<size_t> block_offset;
    std::vector<size_t> block_layer_size;
    std::vector<size_t> layer_size;
    std::vector<std::vector<size_t>> layer_offset;

    block_t blocks;
    size_t num_blocks{0};
    int fd{-1};
    void *base{MAP_FAILED};
    size_t map_size{0};
    Config &config;

    bool is_leader_{false};
    bool did_pin_{false};
};
