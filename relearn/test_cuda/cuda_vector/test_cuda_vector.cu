/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cuda_vector/test_cuda_vector.cuh"

#include "../harness/adapter/random/RandomAdapter.h"

#include "../../source/gpu/structure/CudaArray.cuh"
#include "RelearnGPUException.h"

/*
namespace test_variables {
__device__ utils::Vector::CudaArray<double> test_array_a;
utils::Vector::CudaArrayDeviceHandle<double> handle_test_array_a = CudaArrayFromDeviceSymbol(test_array_a);
};

TEST_F(CudaTest, deviceVariableTest) {
        using namespace test_variables;
        const auto size = RandomAdapter::get_random_integer<size_t>(10, 100, mt);
        const auto c = RandomAdapter::get_random_double(mt);
        ASSERT_TRUE(handle_test_array_a.is_empty());
        handle_test_array_a.resize(size, c);
        ASSERT_EQ(size, handle_test_array_a.get_size());
        std::vector<double> data;
        handle_test_array_a.copy_to_host(data);
        ASSERT_EQ(size, data.size());
        for (const auto& d : data) {
            ASSERT_EQ(c, d);
    }
}*/

TEST_F(CudaVectorTest, cudaMallocTest) {
    const auto size = RandomAdapter::get_random_integer<size_t>(10, 100, mt);
    const auto c = RandomAdapter::get_random_integer<size_t>(mt);
    auto handle = gpu::Vector::create_array_in_device_memory<size_t>();
    ASSERT_TRUE(handle.is_empty());
    handle.resize(size, c);
    ASSERT_EQ(size, handle.get_size());
    std::vector<size_t> data;
    handle.copy_to_host(data);
    ASSERT_EQ(size, data.size());
    for (const auto& d : data) {
        ASSERT_EQ(c, d);
    }
}

TEST_F(CudaVectorTest, cudaFreeTest) {
    const auto size = RandomAdapter::get_random_integer<size_t>(10, 100, mt);
    const auto c = RandomAdapter::get_random_integer<size_t>(mt);
    auto handle = gpu::Vector::create_array_in_device_memory<size_t>();
    handle.resize(size, c);
    ASSERT_TRUE(handle.usable());
    handle.~CudaArrayDeviceHandle<size_t>();
}

TEST_F(CudaVectorTest, cudaFreeTest2) {
    auto handle = gpu::Vector::create_array_in_device_memory<size_t>();
    const auto size = RandomAdapter::get_random_integer<size_t>(10, 100, mt);
    const auto c = RandomAdapter::get_random_integer<size_t>(mt);
    handle.resize(size, c);
    ASSERT_TRUE(handle.usable());
    handle.free();
    ASSERT_FALSE(handle.usable());
    ASSERT_THROW(handle.fill(c), RelearnGPUException);
    ASSERT_THROW(handle.fill(0, 1, c), RelearnGPUException);
    std::vector<size_t> data;
    ASSERT_THROW(handle.copy_to_device(data), RelearnGPUException);
    ASSERT_THROW(handle.copy_to_host(data), RelearnGPUException);
    ASSERT_THROW(handle.free(), RelearnGPUException);
    ASSERT_THROW(handle.free_contents(), RelearnGPUException);
    ASSERT_THROW(handle.is_empty(), RelearnGPUException);
    ASSERT_THROW(handle.data(), RelearnGPUException);
    ASSERT_THROW(handle.get_size(), RelearnGPUException);
    ASSERT_THROW(handle.get_max_size(), RelearnGPUException);
    ASSERT_THROW(handle.resize(42, 42), RelearnGPUException);
    ASSERT_THROW(handle.resize(42), RelearnGPUException);
}

TEST_F(CudaVectorTest, initTest) {
    auto handle = gpu::Vector::create_array_in_device_memory<size_t>();
    ASSERT_TRUE(handle.is_empty());
    ASSERT_EQ(nullptr, handle.data());
    ASSERT_EQ(0, handle.get_size());
    ASSERT_EQ(0, handle.get_max_size());
}

TEST_F(CudaVectorTest, emptyExceptionTest) {
    auto handle = gpu::Vector::create_array_in_device_memory<size_t>();
    ASSERT_THROW(handle.fill(RandomAdapter::get_random_integer<size_t>(mt)), RelearnGPUException);
    ASSERT_THROW(handle.fill(0, 1, RandomAdapter::get_random_integer<size_t>(mt)), RelearnGPUException);
}

void check_filled(gpu::Vector::CudaArrayDeviceHandle<size_t>& handle, size_t from, size_t to, size_t value) {
    std::vector<size_t> data;
    handle.copy_to_host(data);
    ASSERT_GE(data.size(), to);

    for (auto i = from; i < to; i++) {
        ASSERT_EQ(value, data[i]);
    }
}

TEST_F(CudaVectorTest, fillTest) {
    const auto init_size = RandomAdapter::get_random_integer(30, 100, mt);
    const auto v1 = RandomAdapter::get_random_integer<size_t>(mt);
    auto handle = gpu::Vector::create_array_in_device_memory<size_t>();
    handle.resize(init_size);
    check_filled(handle, 0, init_size, 0);
    handle.fill(v1);
    check_filled(handle, 0, init_size, v1);
}

TEST_F(CudaVectorTest, resizeTest) {
    const auto init_size = RandomAdapter::get_random_integer<size_t>(30, 100, mt);
    auto handle = gpu::Vector::create_array_in_device_memory<size_t>();
    handle.resize(init_size);
    check_filled(handle, 0, init_size, 0);
    const auto bigger = RandomAdapter::get_random_integer<size_t>(init_size + 1, 500, mt);
    const auto v2 = RandomAdapter::get_random_integer<size_t>(mt);
    handle.resize(bigger, v2);
    check_filled(handle, 0, init_size, 0);
    check_filled(handle, init_size, bigger, v2);
    const auto smaller = RandomAdapter::get_random_integer<size_t>(1, init_size, mt);
    handle.resize(smaller);
    ASSERT_EQ(smaller, handle.get_size());
    ASSERT_EQ(bigger, handle.get_max_size());
    check_filled(handle, 0, smaller, 0);
    const auto v3 = RandomAdapter::get_random_integer<size_t>(mt);
    handle.fill(v3);
    ASSERT_EQ(smaller, handle.get_size());
    ASSERT_EQ(bigger, handle.get_max_size());
    check_filled(handle, 0, smaller, v3);

    handle.minimize_memory_usage();
    ASSERT_EQ(smaller, handle.get_size());
    ASSERT_EQ(smaller, handle.get_max_size());

    const auto v4 = RandomAdapter::get_random_integer<size_t>(mt);
    handle.resize(bigger, v4);
    ASSERT_EQ(bigger, handle.get_size());
    ASSERT_EQ(bigger, handle.get_max_size());
    check_filled(handle, 0, smaller, v3);
    check_filled(handle, smaller, bigger, v4);
    handle.minimize_memory_usage();
    ASSERT_EQ(bigger, handle.get_size());
    ASSERT_EQ(bigger, handle.get_max_size());
    check_filled(handle, 0, smaller, v3);
    check_filled(handle, smaller, bigger, v4);
    const auto v5 = RandomAdapter::get_random_integer<size_t>(mt);
    handle.resize(smaller, v5);
    handle.minimize_memory_usage();
    ASSERT_EQ(smaller, handle.get_size());
    ASSERT_EQ(smaller, handle.get_max_size());
    check_filled(handle, 0, smaller, v3);
}