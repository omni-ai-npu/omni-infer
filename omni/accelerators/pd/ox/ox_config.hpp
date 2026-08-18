// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <numeric>

#pragma once

namespace asio = boost::asio;

using address_list_t = std::vector<boost::asio::ip::tcp::endpoint>;
using address_clusters_t = std::vector<address_list_t>;  // clustered shard list

using shape_t = std::vector<size_t>;
using block_t = std::vector<shape_t>;

struct Config {
    std::shared_ptr<asio::io_context> io_context;

    asio::io_context &get_io_context()
    {
        return *io_context;
    }

    address_list_t server_list;
    address_list_t shard_list;          // flat shard list (backward compatible)
    address_clusters_t shard_clusters;  // clustered shard list: each inner list is a cluster

    std::string block_table_shm;
    size_t num_block_tables = 1;
    size_t block_size;  // Deepseek v3 MLA size with BF16

    size_t num_blocks = 1024;
    size_t num_layers = 62;
    size_t num_tokens_per_block = 128;
    size_t element_size = 2;
    std::vector<size_t> block_dims = {512, 64, 128};

    size_t table_size;

    size_t num_threads = 16;
    size_t connections_per_shard = 16;
    size_t connections_per_req = 4;
    int zmq_port = 5555;
    int max_connect_retries = 10;
    int max_request_retries = 3;

    inline size_t block_table_size() const
    {
        return num_blocks * block_size;
    }

    inline size_t shm_mem_size() const
    {
        return num_block_tables * block_table_size();
    }

    inline size_t tp_size() const
    {
        // Prefer clustered setting if provided; otherwise fall back to flat list
        if (!shard_clusters.empty()) {
            return shard_clusters.front().size();
        }
        if (!shard_list.empty()) {
            return shard_list.size();
        }
        return 1;
    }

    void update_values()
    {
        size_t total_dims = std::accumulate(block_dims.begin(), block_dims.end(), 0);

        block_size = num_layers * num_tokens_per_block * total_dims * element_size;
        table_size = num_blocks * block_size;

        std::cout << "Number of blocks: " << num_blocks << std::endl 
                  << "Number of layers: " << num_layers << std::endl
                  << "Tokens per block: " << num_tokens_per_block  << std::endl
                  << "Number of KV TP: " << tp_size() << std::endl
                  << "KV dimension split: "; 
        for(auto dim : block_dims) 
        {
            std::cout << dim << " ";
        }
        std::cout << "\n";
    }
};

void print_usage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --addr <ip:port,ip:port>                 Server IP list (7.150.8.141:9000,7.150.8.141:9001)\n"
              << "  --shard-list <grp1;grp2;...>             Shard server IPs; ';' separates clusters, ',' separates "
                 "nodes in a cluster\n"
              << "                                           e.g. "
                 "\"10.0.0.1:15000,10.0.0.2:15000;10.0.1.1:15000,10.0.1.2:15000\"\n"
              << "  --zmq-port <port>                        ZMQ port\n"
              << "  --block-table-shm <name>                 Shared memory name for block table\n"
              << "  --num-block-tables <num>                 Number of block tables (default: 1)\n"
              << "  --num-blocks <num>                       Number of blocks in each block table\n"
              << "  --num-layers <size>                      Number of layers per block (e.g., 61 for DS\n"
              << "  --tokens-per-block <size>                Number of tokens per block (e.g., 128)\n"
              << "  --dims <list of size>                    KV dimension split (e.g., 512,64 )\n"
              << "  --dtype <size>                           Block size with unit (e.g., 1 for int8, 2 for bfloat16)\n"
              << "  --num-threads <threads>                  Number of threads (default: 16)\n"
              << "  --num-connections <conn>                 Number of connections per shard (default: 16)\n"
              << "  --num-connections-per-req <conn>               Number of connections per req (default: 4)\n"
              << "  -h                                       Show this help message\n";
}

address_list_t parse_address_list(const std::string &address_str)
{
    address_list_t endpoints;
    std::vector<std::string> address_pairs;
    boost::split(address_pairs, address_str, boost::is_any_of(","), boost::token_compress_on);

    for (const auto &addr_pair : address_pairs) {
        if (addr_pair.empty())
            continue;

        std::vector<std::string> parts;
        boost::split(parts, addr_pair, boost::is_any_of(":"), boost::token_compress_on);

        if (parts.size() != 2) {
            throw std::runtime_error("Invalid address format: " + addr_pair);
        }

        std::string ip = parts[0];
        int port = std::stoi(parts[1]);

        boost::asio::ip::address address = boost::asio::ip::address::from_string(ip);
        endpoints.emplace_back(address, port);
    }

    return endpoints;
}

// Parse clustered shard list where ';' separates clusters and ',' separates nodes within a cluster.
// Fills both clustered (out_clusters) and flat (out_flat) representations.
// If no ';' exists, treats the whole input as a single cluster (backward compatible).
void parse_shard_list_grouped(
    const std::string &address_str, address_clusters_t &out_clusters, address_list_t &out_flat)
{
    out_clusters.clear();
    out_flat.clear();

    if (address_str.empty())
        return;

    if (address_str.find(';') == std::string::npos) {
        // Old format: single cluster, comma-separated
        out_flat = parse_address_list(address_str);
        if (!out_flat.empty()) {
            out_clusters.push_back(out_flat);
        }
        return;
    }

    // New format: multiple clusters separated by ';'
    std::vector<std::string> groups;
    boost::split(groups, address_str, boost::is_any_of(";"), boost::token_compress_on);

    for (const auto &grp : groups) {
        if (grp.empty())
            continue;
        address_list_t eps = parse_address_list(grp);
        if (!eps.empty()) {
            out_clusters.push_back(eps);
            // append to flat
            out_flat.insert(out_flat.end(), eps.begin(), eps.end());
        }
    }

    // If after parsing groups, we somehow got no clusters but flat had entries,
    // normalize to one cluster.
    if (out_clusters.empty() && !out_flat.empty()) {
        out_clusters.push_back(out_flat);
    }
}

Config parse_arguments(int argc, char *argv[])
{
    Config config;
    config.io_context = std::make_shared<asio::io_context>();

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "--addr" && i + 1 < argc) {
                config.server_list = parse_address_list(argv[++i]);
            } else if ((arg == "--shard-list" || arg == "--shard_list") && i + 1 < argc) {
                std::string val = argv[++i];
                parse_shard_list_grouped(val, config.shard_clusters, config.shard_list);
            } else if (arg == "--block-table-shm" || arg == "--block_table_shm") {
                config.block_table_shm = argv[++i];
            } else if ((arg == "--num-block-tables" || arg == "--num_block_tables") && i + 1 < argc) {
                config.num_block_tables = std::stoul(argv[++i]);
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
            } else if ((arg == "--zmq-port" || arg == "--zmq_port") && i + 1 < argc) {
                config.zmq_port = std::stoul(argv[++i]);
            } else if ((arg == "--max-connect-retries" || arg == "--max_connect_retries") && i + 1 < argc) {
                config.max_connect_retries = std::stoi(argv[++i]);
            } else if ((arg == "--max-request-retries" || arg == "--max_request_retries") && i + 1 < argc) {
                config.max_request_retries = std::stoi(argv[++i]);
            } else if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                exit(0);
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                print_usage(argv[0]);
                exit(1);
            }
        }

        if (config.block_table_shm.empty()) {
            throw std::runtime_error("--block-table-shm must be specified");
        }

        if (config.num_blocks == 0) {
            throw std::runtime_error("--num-blocks must be specified and greater than 0");
        }

        config.update_values();

        if (config.block_size == 0) {
            throw std::runtime_error("--block-size must be specified and greater than 0");
        }
    } catch (const std::exception &e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        print_usage(argv[0]);
        exit(1);
    }

    return config;
}
