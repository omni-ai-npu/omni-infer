// SPDX-License-Identifier: MIT
// Mock OX for unit testing - simulates various failure scenarios
//
// Compile with: g++ -std=c++20 -o mock_<type> mock_ox.cpp -DMOCK_<TYPE>
// Types:
//   MOCK_STARTUP_FAIL    - Exit immediately with code 1
//   MOCK_RUNTIME_CRASH   - Run 2 seconds then exit with code 1
//   MOCK_SIGNAL_CRASH    - Run 2 seconds then kill self with SIGABRT
//   MOCK_NORMAL          - Run indefinitely (normal operation)

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <csignal>
#include <cstdlib>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --help              Show this help\n"
              << "  --crash-after N     Crash after N seconds (default: 2)\n"
              << "  --exit-code N       Exit with code N (default: 1)\n"
              << "  --signal N          Kill with signal N instead of exit\n";
}

int main(int argc, char* argv[]) {
    int crash_after_sec = 2;
    int exit_code = 1;
    int signal_num = -1;  // -1 means use exit(), otherwise use raise()

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--crash-after" && i + 1 < argc) {
            crash_after_sec = std::stoi(argv[++i]);
        } else if (arg == "--exit-code" && i + 1 < argc) {
            exit_code = std::stoi(argv[++i]);
        } else if (arg == "--signal" && i + 1 < argc) {
            signal_num = std::stoi(argv[++i]);
        }
    }

    std::cout << "[MOCK_OX] Starting mock ox process" << std::endl;
    std::cout << "[MOCK_OX] PID: " << getpid() << std::endl;
    std::cout.flush();

#if defined(MOCK_STARTUP_FAIL)
    // Immediately fail - simulates startup failure (bad config, missing file, etc.)
    std::cerr << "[MOCK_OX] STARTUP_FAIL: Exiting immediately with code " << exit_code << std::endl;
    return exit_code;

#elif defined(MOCK_RUNTIME_CRASH)
    // Run for a while then crash - simulates runtime error (segfault logic, assertion fail, etc.)
    std::cout << "[MOCK_OX] RUNTIME_CRASH: Will exit after " << crash_after_sec << " seconds" << std::endl;
    std::cout.flush();
    std::this_thread::sleep_for(std::chrono::seconds(crash_after_sec));
    std::cerr << "[MOCK_OX] RUNTIME_CRASH: Crashing now with exit code " << exit_code << std::endl;
    return exit_code;

#elif defined(MOCK_SIGNAL_CRASH)
    // Run then kill self with signal - simulates external kill or fatal signal
    std::cout << "[MOCK_OX] SIGNAL_CRASH: Will signal(" << signal_num << ") after " << crash_after_sec << " seconds" << std::endl;
    std::cout.flush();
    std::this_thread::sleep_for(std::chrono::seconds(crash_after_sec));
    std::cerr << "[MOCK_OX] SIGNAL_CRASH: Raising signal " << signal_num << std::endl;
    std::raise(signal_num);
    // Should not reach here
    return 99;

#elif defined(MOCK_NORMAL)
    // Run forever - simulates normal healthy operation
    std::cout << "[MOCK_OX] NORMAL: Running indefinitely (healthy operation)" << std::endl;
    std::cout.flush();
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        std::cout << "[MOCK_OX] NORMAL: Still running..." << std::endl;
        std::cout.flush();
    }
    return 0;

#else
    // Default: run indefinitely
    std::cout << "[MOCK_OX] DEFAULT: No mock type defined, running indefinitely" << std::endl;
    std::cout.flush();
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    return 0;
#endif
}