/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_cuda_stack.cuh"

#include "../harness/adapter/random/RandomAdapter.h"
#include "../../source/gpu/structure/CudaStack.cuh"

#include "RelearnGPUException.h"

using StackType = double;

__global__ void do_push(gpu::Vector::CudaStack<StackType>* stack, StackType* _return, size_t* new_size, StackType to_push) {
    stack->push(to_push);
    _return[0] = stack->top();
    new_size[0] = stack->size;
}

__global__ void do_pop(gpu::Vector::CudaStack<StackType>* stack, StackType* _return, size_t* new_size) {
    stack->pop();
    _return[0] = stack->top();
    new_size[0] = stack->size;
}

__global__ void do_reset(gpu::Vector::CudaStack<StackType>* stack, size_t* new_size) {
    stack->reset();
    new_size[0] = stack->size;
}

__global__ void do_empty(gpu::Vector::CudaStack<StackType>* stack, bool* _return) {
    _return[0] = stack->empty();
}

__global__ void give_max_size(gpu::Vector::CudaStack<StackType>* stack, size_t* _return) {
    _return[0] = stack->max_size;
}

// Checks if resize and all operations work correctly on the GPU
TEST_F(CudaStackTest, cudaStackBasicOpTestAndResize) {
    // Resize setup
    gpu::Vector::CudaStack<StackType>* dev_stack = init_class_on_device<gpu::Vector::CudaStack<StackType>>();
    gpu::Vector::CudaStackDeviceHandle<StackType> stack(dev_stack);
    stack.resize(3);
    size_t* device_max_size = (size_t*)cuda_malloc(sizeof(size_t));
    give_max_size<<<1,1>>>(dev_stack, device_max_size);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    size_t max_size;
    cuda_memcpy_to_host(device_max_size, &max_size, sizeof(size_t), 1);
    ASSERT_EQ(max_size, 3);

    // Push
    StackType* device_return = (StackType*)cuda_malloc(sizeof(StackType));
    StackType cpu_return;
    size_t* device_new_size = (size_t*)cuda_malloc(sizeof(size_t));
    size_t new_size;
    do_push<<<1,1>>>(dev_stack, device_return, device_new_size, 0.5);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host(device_return, &cpu_return, sizeof(StackType), 1);
    cuda_memcpy_to_host(device_new_size, &new_size, sizeof(size_t), 1);
    ASSERT_EQ(new_size, 1);
    ASSERT_EQ(cpu_return, 0.5);

    // Pop
    do_push<<<1,1>>>(dev_stack, device_return, device_new_size, 0.9);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    do_push<<<1,1>>>(dev_stack, device_return, device_new_size, 0.8);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    do_pop<<<1,1>>>(dev_stack, device_return, device_new_size);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host(device_return, &cpu_return, sizeof(StackType), 1);
    cuda_memcpy_to_host(device_new_size, &new_size, sizeof(size_t), 1);
    ASSERT_EQ(new_size, 2);
    ASSERT_EQ(cpu_return, 0.9);

    // Reset
    do_reset<<<1,1>>>(dev_stack, device_new_size);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host(device_new_size, &new_size, sizeof(size_t), 1);
    ASSERT_EQ(new_size, 0);

    // Empty
    bool* device_empty = (bool*)cuda_malloc(sizeof(bool));
    bool empty;
    do_empty<<<1,1>>>(dev_stack, device_empty);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host(device_empty, &empty, sizeof(bool), 1);
    ASSERT_EQ(empty, true);

    do_push<<<1,1>>>(dev_stack, device_return, device_new_size, 0.9);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    do_empty<<<1,1>>>(dev_stack, device_empty);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host(device_empty, &empty, sizeof(bool), 1);
    ASSERT_EQ(empty, false);
}

// Checks if the copy functionality works
TEST_F(CudaStackTest, cudaStackCopyToAndBack) {
    
    gpu::Vector::CudaStack<StackType>* dev_stack = init_class_on_device<gpu::Vector::CudaStack<StackType>>();
    gpu::Vector::CudaStackDeviceHandle<StackType> stack(dev_stack);

    std::vector<StackType> copy_to = {0.4, 5.3, 1.0};
    stack.copy_to_device(copy_to);

    std::vector<StackType> copy_back;
    stack.copy_to_host(copy_back);

    for (int i = 0; i < copy_to.size(); i++) {
        ASSERT_EQ(copy_to[i], copy_back[i]);
    }

    copy_to = {0.4, 5.3};
    copy_back.clear();
    stack.copy_to_device(copy_to);
    stack.copy_to_host(copy_back);

    for (int i = 0; i < copy_to.size(); i++) {
        ASSERT_EQ(copy_to[i], copy_back[i]);
    }
}

TEST_F(CudaStackTest, cudaStackFreeTest) {
    gpu::Vector::CudaStack<StackType>* dev_stack = init_class_on_device<gpu::Vector::CudaStack<StackType>>();
    gpu::Vector::CudaStackDeviceHandle<StackType> stack(dev_stack);

    ASSERT_TRUE(stack.usable());
    stack.free();
    ASSERT_FALSE(stack.usable());
    std::vector<StackType> data;
    ASSERT_THROW(stack.copy_to_device(data), RelearnGPUException);
    ASSERT_THROW(stack.copy_to_host(data), RelearnGPUException);
    ASSERT_THROW(stack.free(), RelearnGPUException);
    ASSERT_THROW(stack.free_contents(), RelearnGPUException);
    ASSERT_THROW(stack.is_empty(), RelearnGPUException);
    ASSERT_THROW(stack.data(), RelearnGPUException);
    ASSERT_THROW(stack.get_size(), RelearnGPUException);
    ASSERT_THROW(stack.get_max_size(), RelearnGPUException);
    ASSERT_THROW(stack.resize(42), RelearnGPUException);
}