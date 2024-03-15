/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_cuda_coalesced_stack.cuh"

#include "../harness/adapter/random/RandomAdapter.h"
#include "../../source/gpu/structure/CudaCoalescedStack.cuh"

#include "RelearnGPUException.h"

using StackType = double;

static constexpr size_t max_size = 3;
static constexpr size_t num_threads = 5;

__global__ void do_push(gpu::Vector::CudaCoalescedStack<StackType>* stack, StackType* _return, size_t* new_size, StackType to_push, int thread_id) {
    stack->push(to_push, thread_id);
    _return[0] = stack->top(thread_id);
    new_size[0] = stack->sizes[thread_id];
}

__global__ void do_pop(gpu::Vector::CudaCoalescedStack<StackType>* stack, StackType* _return, size_t* new_size, int thread_id) {
    stack->pop(thread_id);
    _return[0] = stack->top(thread_id);
    new_size[0] = stack->sizes[thread_id];
}

__global__ void do_reset(gpu::Vector::CudaCoalescedStack<StackType>* stack, size_t* new_size, int thread_id) {
    stack->reset(thread_id);
    new_size[0] = stack->sizes[thread_id];
}

__global__ void do_empty(gpu::Vector::CudaCoalescedStack<StackType>* stack, bool* _return, int thread_id) {
    _return[0] = stack->empty(thread_id);
}

__global__ void give_max_size(gpu::Vector::CudaCoalescedStack<StackType>* stack, size_t* _return) {
    _return[0] = stack->max_size;
}

// Checks if resize and all operations work correctly on the GPU
TEST_F(CudaCoalescedStackTest, cudaCoalescedStackBasicOpTestAndResize) {
    // Resize setup
    gpu::Vector::CudaCoalescedStack<StackType>* dev_stack = init_class_on_device<gpu::Vector::CudaCoalescedStack<StackType>>();
    gpu::Vector::CudaCoalescedStackDeviceHandle<StackType> stack(dev_stack);
    stack.resize(max_size, num_threads);
    size_t* device_max_size = (size_t*)cuda_malloc(sizeof(size_t));
    give_max_size<<<1,1>>>(dev_stack, device_max_size);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    size_t max_size;
    cuda_memcpy_to_host(device_max_size, &max_size, sizeof(size_t), 1);
    ASSERT_EQ(max_size, max_size);

    // Push
    const auto thread_to_push = RandomAdapter::get_random_integer<size_t>(0, num_threads - 1, mt);

    StackType* device_return = (StackType*)cuda_malloc(sizeof(StackType));
    StackType cpu_return;
    size_t* device_new_size = (size_t*)cuda_malloc(sizeof(size_t));
    size_t new_size;
    do_push<<<1,1>>>(dev_stack, device_return, device_new_size, 0.5, thread_to_push);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host(device_return, &cpu_return, sizeof(StackType), 1);
    cuda_memcpy_to_host(device_new_size, &new_size, sizeof(size_t), 1);
    ASSERT_EQ(new_size, 1);
    ASSERT_EQ(cpu_return, 0.5);

    // Pop
    do_push<<<1,1>>>(dev_stack, device_return, device_new_size, 0.9, thread_to_push);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    do_push<<<1,1>>>(dev_stack, device_return, device_new_size, 0.8, thread_to_push);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    do_pop<<<1,1>>>(dev_stack, device_return, device_new_size, thread_to_push);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host(device_return, &cpu_return, sizeof(StackType), 1);
    cuda_memcpy_to_host(device_new_size, &new_size, sizeof(size_t), 1);
    ASSERT_EQ(new_size, 2);
    ASSERT_EQ(cpu_return, 0.9);

    // Reset
    do_reset<<<1,1>>>(dev_stack, device_new_size, thread_to_push);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host(device_new_size, &new_size, sizeof(size_t), 1);
    ASSERT_EQ(new_size, 0);

    // Empty
    bool* device_empty = (bool*)cuda_malloc(sizeof(bool));
    bool empty;
    do_empty<<<1,1>>>(dev_stack, device_empty, thread_to_push);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host(device_empty, &empty, sizeof(bool), 1);
    ASSERT_EQ(empty, true);

    do_push<<<1,1>>>(dev_stack, device_return, device_new_size, 0.9, thread_to_push);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    do_empty<<<1,1>>>(dev_stack, device_empty, thread_to_push);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host(device_empty, &empty, sizeof(bool), 1);
    ASSERT_EQ(empty, false);

    for (int i = 0; i < num_threads; i++) {
        if (i == thread_to_push) {
            continue;
        }

        do_empty<<<1,1>>>(dev_stack, device_empty, i);
        cudaDeviceSynchronize();
        gpu_check_last_error();
        cuda_memcpy_to_host(device_empty, &empty, sizeof(bool), 1);
        ASSERT_EQ(empty, true);
    }
}

TEST_F(CudaCoalescedStackTest, cudaCoalescedStackFreeTest) {
    gpu::Vector::CudaCoalescedStack<StackType>* dev_stack = init_class_on_device<gpu::Vector::CudaCoalescedStack<StackType>>();
    gpu::Vector::CudaCoalescedStackDeviceHandle<StackType> stack(dev_stack);

    ASSERT_TRUE(stack.usable());
    stack.free();
    ASSERT_FALSE(stack.usable());
    std::vector<StackType> data;
    ASSERT_THROW(stack.free(), RelearnGPUException);
    ASSERT_THROW(stack.free_contents(), RelearnGPUException);
    ASSERT_THROW(stack.resize(42, 42), RelearnGPUException);
}