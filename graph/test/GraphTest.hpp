#pragma once

#include <gtest/gtest.h>

#include <limits>

#include <cuda_runtime_api.h>

constexpr auto max = std::numeric_limits<double>::max();

/**
 * @brief Check if cudaGetDevice can be executed with no error
 *
 * Checks if a CUDA capable device (default device) is available
 *
 * @return true CUDA_FOUND is true and cudaGetDevice executes with no error
 * @return false else
 */
[[nodiscard]] inline bool can_run_cuda() {
    if constexpr (CUDA_FOUND) {
        int device{};
        const auto err = cudaGetDevice(&device);
        return err == cudaSuccess;
    }
    return false;
}

const auto enable_cuda = can_run_cuda();
