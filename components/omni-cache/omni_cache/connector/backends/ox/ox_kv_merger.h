// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#include <iostream>
#include <cstring>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

class KVCacheMerger {
private:
    const int num_layers_;
    const int block_size_;
    const int kv_dimension_;
    const int tp_;

    // Segment dimensions calculation
    struct SegmentDims {
        int seg1;  // kv_dimension * 8 / (tp * 11)
        int seg2;  // kv_dimension / (tp * 11)
        int seg3;  // kv_dimension * 2 / (tp * 11)
        int total_shard;  // seg1 + seg2 + seg3
        int total_merged; // Sum of all shards' segments
    };

    SegmentDims dims_;

public:
    // Constructor with TP parameter
    KVCacheMerger(int num_layers = 61, int block_size = 128, int kv_dimension = 704, int tp = 1)
        : num_layers_(num_layers), block_size_(block_size), kv_dimension_(kv_dimension), tp_(tp) {
        // Calculate segment dimensions once during construction
        dims_.seg1 = (kv_dimension_ * 8) / (tp_ * 11);
        dims_.seg2 = kv_dimension_ / (tp_ * 11);
        dims_.seg3 = (kv_dimension_ * 2) / (tp_ * 11);
        dims_.total_shard = dims_.seg1 + dims_.seg2 + dims_.seg3;
        dims_.total_merged = dims_.seg1 * tp_ + dims_.seg2 * tp_ + dims_.seg3 * tp_;
    }

    // Get TP value
    int get_tp() const { return tp_; }

    void merge_shards(const short* src, short* dst) const {
        if (tp_ == 1) {
            std::memcpy(dst, src, num_layers_ * block_size_ * kv_dimension_ * sizeof(short));
            return;
        }

        const int shard_layer_size = block_size_ * dims_.total_shard;
        const int merged_layer_size = block_size_ * dims_.total_merged;

        for (int layer = 0; layer < num_layers_; ++layer) {
            short* layer_dst = dst + layer * merged_layer_size;
            
            for (int shard = 0; shard < tp_; ++shard) {
                const short* layer_src = src + shard * (num_layers_ * shard_layer_size) 
                                    + layer * shard_layer_size;
                
                // Calculate offsets for the three segments
                const int seg1_src_offset = 0;
                const int seg2_src_offset = dims_.seg1 * block_size_;
                const int seg3_src_offset = (dims_.seg1 + dims_.seg2) * block_size_;
                
                const int seg1_dst_offset = shard * dims_.seg1 * block_size_;
                const int seg2_dst_offset = dims_.seg1 * tp_ * block_size_ + shard * dims_.seg2 * block_size_;
                const int seg3_dst_offset = (dims_.seg1 * tp_ + dims_.seg2 * tp_) * block_size_ + shard * dims_.seg3 * block_size_;
                
                // Copy all three segments for this shard and layer
                std::memcpy(layer_dst + seg1_dst_offset, layer_src + seg1_src_offset, 
                        dims_.seg1 * block_size_ * sizeof(short));
                std::memcpy(layer_dst + seg2_dst_offset, layer_src + seg2_src_offset, 
                        dims_.seg2 * block_size_ * sizeof(short));
                std::memcpy(layer_dst + seg3_dst_offset, layer_src + seg3_src_offset, 
                        dims_.seg3 * block_size_ * sizeof(short));
            }
        }
    }

   // Initialize source memory with simple sequential pattern for verification
    void initialize_simple_pattern(short* src) const {
        const int elements_per_shard = num_layers_ * block_size_ * dims_.total_shard;

        short counter = 1;
        for (int shard = 0; shard < tp_; ++shard) {
            for (int layer = 0; layer < num_layers_; ++layer) {
                for (int block = 0; block < block_size_; ++block) {
                    const int base_idx = shard * elements_per_shard +
                                       layer * block_size_ * dims_.total_shard +
                                       block * dims_.total_shard;

                    // Fill all three segments sequentially
                    for (int i = 0; i < dims_.total_shard; ++i) {
                        src[base_idx + i] = counter++;
                    }
                }
            }
        }
    }

    // Verify merged results with detailed printing for single block
    bool verify_results_single_block(const short* src, const short* dst) const {
        if (tp_ == 1) return true;

        const int elements_per_layer = block_size_ * dims_.total_merged;
        const int elements_per_shard = num_layers_ * block_size_ * dims_.total_shard;

        std::cout << "\n=== Detailed Verification (Single Block) ===" << std::endl;
        std::cout << "Configuration: [" << num_layers_ << ", " << block_size_ << ", " << kv_dimension_ << "]" << std::endl;
        std::cout << "TP: " << tp_ << std::endl;
        std::cout << "Shard dimensions: [" << dims_.seg1 << ", " << dims_.seg2 << ", " << dims_.seg3 << "]" << std::endl;
        std::cout << "Merged dimensions: [" << dims_.seg1 * tp_ << ", " << dims_.seg2 * tp_ << ", " << dims_.seg3 * tp_ << "]" << std::endl;

        bool all_correct = true;
        const int block = 0;  // Only test first block

        // Print source shards for both layers, block 0
        std::cout << "\n--- Source Shards (Block " << block << ") ---" << std::endl;
        for (int layer = 0; layer < num_layers_; ++layer) {
            std::cout << "\nLayer " << layer << ":" << std::endl;
            for (int shard = 0; shard < tp_; ++shard) {
                const int src_base = shard * elements_per_shard +
                                   layer * block_size_ * dims_.total_shard +
                                   block * dims_.total_shard;

                std::cout << "  Shard " << shard << " (seg1): ";
                for (int i = 0; i < dims_.seg1; ++i) {
                    std::cout << std::setw(4) << src[src_base + i] << " ";
                }
                std::cout << std::endl;

                std::cout << "  Shard " << shard << " (seg2): ";
                for (int i = 0; i < dims_.seg2; ++i) {
                    std::cout << std::setw(4) << src[src_base + dims_.seg1 + i] << " ";
                }
                std::cout << std::endl;

                std::cout << "  Shard " << shard << " (seg3): ";
                for (int i = 0; i < dims_.seg3; ++i) {
                    std::cout << std::setw(4) << src[src_base + dims_.seg1 + dims_.seg2 + i] << " ";
                }
                std::cout << std::endl;
            }
        }

        // Print merged result in memory order
        std::cout << "\n--- Merged Result (Memory Order, Block " << block << ") ---" << std::endl;
        for (int layer = 0; layer < num_layers_; ++layer) {
            std::cout << "\nLayer " << layer << ":" << std::endl;
            const int dst_base = layer * elements_per_layer + block * dims_.total_merged;

            std::cout << "  Segment 1: ";
            for (int i = 0; i < dims_.seg1 * tp_; ++i) {
                std::cout << std::setw(4) << dst[dst_base + i] << " ";
            }
            std::cout << std::endl;

            std::cout << "  Segment 2: ";
            for (int i = 0; i < dims_.seg2 * tp_; ++i) {
                std::cout << std::setw(4) << dst[dst_base + dims_.seg1 * tp_ + i] << " ";
            }
            std::cout << std::endl;

            std::cout << "  Segment 3: ";
            for (int i = 0; i < dims_.seg3 * tp_; ++i) {
                std::cout << std::setw(4) << dst[dst_base + dims_.seg1 * tp_ + dims_.seg2 * tp_ + i] << " ";
            }
            std::cout << std::endl;

            // Print complete memory layout for this layer
            std::cout << "  Complete memory: ";
            for (int i = 0; i < dims_.total_merged; ++i) {
                std::cout << std::setw(4) << dst[dst_base + i] << " ";
            }
            std::cout << std::endl;
        }

        // Verify correctness for all layers and the single block
        for (int layer = 0; layer < num_layers_; ++layer) {
            const int dst_base = layer * elements_per_layer + block * dims_.total_merged;

            for (int shard = 0; shard < tp_; ++shard) {
                const int src_base = shard * elements_per_shard +
                                   layer * block_size_ * dims_.total_shard +
                                   block * dims_.total_shard;

                // Verify segment 1
                for (int i = 0; i < dims_.seg1; ++i) {
                    short expected = src[src_base + i];
                    short actual = dst[dst_base + shard * dims_.seg1 + i];
                    if (actual != expected) {
                        std::cout << "ERROR: Layer " << layer << " Segment1 mismatch at shard=" << shard
                                  << " i=" << i << " expected=" << expected
                                  << " actual=" << actual << std::endl;
                        all_correct = false;
                    }
                }

                // Verify segment 2
                for (int i = 0; i < dims_.seg2; ++i) {
                    short expected = src[src_base + dims_.seg1 + i];
                    short actual = dst[dst_base + dims_.seg1 * tp_ + shard * dims_.seg2 + i];
                    if (actual != expected) {
                        std::cout << "ERROR: Layer " << layer << " Segment2 mismatch at shard=" << shard
                                  << " i=" << i << " expected=" << expected
                                  << " actual=" << actual << std::endl;
                        all_correct = false;
                    }
                }

                // Verify segment 3
                for (int i = 0; i < dims_.seg3; ++i) {
                    short expected = src[src_base + dims_.seg1 + dims_.seg2 + i];
                    short actual = dst[dst_base + dims_.seg1 * tp_ + dims_.seg2 * tp_ + shard * dims_.seg3 + i];
                    if (actual != expected) {
                        std::cout << "ERROR: Layer " << layer << " Segment3 mismatch at shard=" << shard
                                  << " i=" << i << " expected=" << expected
                                  << " actual=" << actual << std::endl;
                        all_correct = false;
                    }
                }
            }
        }

        if (all_correct) {
            std::cout << "\n✓ All values verified correctly!" << std::endl;
        } else {
            std::cout << "\n✗ Some values are incorrect!" << std::endl;
        }

        return all_correct;
    }

    // Get total memory size in bytes
    size_t get_total_memory_size() const {
        return num_layers_ * block_size_ * dims_.total_shard * tp_ * sizeof(short);
    }

    // Print configuration info
    void print_config() const {
        std::cout << "Configuration: [" << num_layers_ << ", " << block_size_ << ", " << kv_dimension_ << "]" << std::endl;
        std::cout << "TP: " << tp_ << std::endl;
        std::cout << "Segment dimensions - Shard: [" << dims_.seg1 << ", " << dims_.seg2 << ", " << dims_.seg3 << "]" << std::endl;
        std::cout << "Segment dimensions - Merged: [" << dims_.seg1 * tp_ << ", " << dims_.seg2 * tp_ << ", " << dims_.seg3 * tp_ << "]" << std::endl;
        std::cout << "Total memory: " << get_total_memory_size() / (1024.0 * 1024.0) << " MB" << std::endl;
    }
};
