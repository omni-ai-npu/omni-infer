// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <iostream>
#include <cstring>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include "ox_kv_merger.h"


// Function verification test with small dimensions [2, 1, 22]
void run_verification_test() {
    std::cout << "=== FUNCTION VERIFICATION TEST ===" << std::endl;
    std::cout << "Testing with dimensions: [2, 1, 22], TP=2" << std::endl;

    // Create merger with test dimensions - using 1 block for simplicity
    KVCacheMerger merger(2, 1, 22, 2);  // [num_layers, block_size, kv_dimension, tp]

    // Allocate memory
    const size_t total_size = merger.get_total_memory_size();
    short* src_memory = new short[total_size / sizeof(short)]();
    short* dst_memory = new short[total_size / sizeof(short)]();

    // Initialize with simple sequential pattern
    merger.initialize_simple_pattern(src_memory);

    // Perform merge using sequential version
    merger.merge_shards(src_memory, dst_memory);

    // Verify with detailed printing (single block)
    bool success = merger.verify_results_single_block(src_memory, dst_memory);

    if (success) {
        std::cout << "\n   Verification test PASSED!" << std::endl;
    } else {
        std::cout << "\n❌ Verification test FAILED!" << std::endl;
    }

    delete[] src_memory;
    delete[] dst_memory;
}

// Performance benchmark comparing sequential vs parallel versions
void run_performance_benchmark() {
    std::cout << "\n\n=== PERFORMANCE BENCHMARK ===" << std::endl;
    std::cout << "Testing with default dimensions: [61, 128, 704]" << std::endl;
    // std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;

    std::vector<int> tp_values = {2, 4, 8};

    for (int tp : tp_values) {
        std::cout << "\n--- Benchmark TP=" << tp << " ---" << std::endl;
        KVCacheMerger merger(62, 128, 704, tp);
        merger.print_config();

        // Allocate memory
        const size_t total_size = merger.get_total_memory_size();
        short* src_memory = new short[total_size / sizeof(short)];
        short* dst_memory_seq = new short[total_size / sizeof(short)];
        short* dst_memory_par = new short[total_size / sizeof(short)];

        // Initialize test data
        merger.initialize_simple_pattern(src_memory);

        // Test sequential version
        std::vector<double> seq_times;
        for (int i = 0; i < 5; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            merger.merge_shards(src_memory, dst_memory_seq);
            auto end = std::chrono::high_resolution_clock::now();
            seq_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        // Test parallel version
        // std::vector<double> par_times;
        // for (int i = 0; i < 5; ++i) {
        //     auto start = std::chrono::high_resolution_clock::now();
        //     merger.merge_shards_parallel(src_memory, dst_memory_par);
        //     auto end = std::chrono::high_resolution_clock::now();
        //     par_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        // }

        // Verify both versions produce same results
        // bool results_match = std::memcmp(dst_memory_seq, dst_memory_par, total_size) == 0;
        // std::cout << "Results verification: " << (results_match ? "✓ PASSED" : "✗ FAILED") << std::endl;

        // Calculate statistics
        double seq_avg = std::accumulate(seq_times.begin(), seq_times.end(), 0.0) / seq_times.size();
        // double par_avg = std::accumulate(par_times.begin(), par_times.end(), 0.0) / par_times.size();
        // double speedup = seq_avg / par_avg;
        double data_size_mb = total_size / (1024.0 * 1024.0);

        std::cout << "Performance Results:" << std::endl;
        std::cout << "  Sequential: " << seq_avg << " ms" << std::endl;
        // std::cout << "  Parallel:   " << par_avg << " ms" << std::endl;
        // std::cout << "  Speedup:    " << speedup << "x" << std::endl;
        // std::cout << "  Bandwidth (parallel): " << (data_size_mb / (par_avg / 1000.0)) << " MB/s" << std::endl;

        delete[] src_memory;
        delete[] dst_memory_seq;
        delete[] dst_memory_par;
    }
}

int main() {
    std::cout << "KV Cache Merger - Enhanced Version with OpenMP" << std::endl;

    // Run verification test with [2, 1, 22], TP=2
    run_verification_test();

    // Run performance benchmarks comparing sequential vs parallel
    run_performance_benchmark();

    return 0;
}
