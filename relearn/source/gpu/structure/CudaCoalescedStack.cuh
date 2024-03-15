#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "../Commons.cuh"
#include "../RelearnGPUException.h"

#include <iostream>
#include <vector>

namespace gpu::Vector {
template <typename T>
struct CudaCoalescedStack {
    /**
     * Stack structure on the GPU which holds one stack per thread and allows for coalesced access
     */
    T* data = nullptr;
    size_t* sizes = nullptr;
    size_t max_size = 0;
    size_t num_threads = 0;

    /**
     * @brief Returns the top element of the stack for the given id
     * @param thread_id The id of the thread to which the stack belongs
     * @return The top element of the stack
     */
    __device__ T& top(int thread_id) {
        RelearnGPUException::device_check(thread_id < num_threads, "CudaCoalescedStack::top: Thread id too large");
        RelearnGPUException::device_check(sizes[thread_id] >= 1, "CudaStack::top: Stack is empty for thread %d", thread_id);
        return data[thread_id + (sizes[thread_id] - 1) * num_threads];
    }

    /**
     * @brief Puts an element at the top of the stack
     * @param element The element to be put at the top of the stack
     * @param thread_id The id of the thread to which the stack belongs
     */
    __device__ void push(const T& element, int thread_id) {
        RelearnGPUException::device_check(sizes[thread_id] < max_size, "CudaCoalescedStack::push: Stack is full, max size: %llu", max_size);
        data[thread_id + sizes[thread_id] * num_threads] = element;
        sizes[thread_id]++;
    }

    /**
     * @brief Removes the top element from the stack
     * @param thread_id The id of the thread to which the stack belongs
     */
    __device__ void pop(int thread_id) {
        RelearnGPUException::device_check(sizes[thread_id] >= 1, "CudaCoalescedStack::top: Stack is empty");
        sizes[thread_id]--;
    }

    /**
     * @brief Resets the size of the stack to 0
     * @param thread_id The id of the thread to which the stack belongs
     */
    __device__ void reset(int thread_id) {
        sizes[thread_id] = 0;
    }

    /**
     * @brief Returns true if the stack is empty, false if not
     * @param thread_id The id of the thread to which the stack belongs
     * @return True if the stack is empty, false if not
     */
    __device__ bool empty(int thread_id) {
        return sizes[thread_id] == 0;
    }
};

template <typename T>
class CudaCoalescedStackDeviceHandle {
    /**
     * A handle to control a CudaCoalescedStack from the cpu
     */

public:
    CudaCoalescedStackDeviceHandle() {
        struct_dev_ptr = nullptr;
    }

    /**
     * @param struct_device_pointer Pointer to a CudaCoalescedStack instance on the gpu
     */
    CudaCoalescedStackDeviceHandle(CudaCoalescedStack<T>* struct_device_ptr)
        : struct_dev_ptr((void*)struct_device_ptr) {
    }

    /**
     * @param struct_device_pointer Pointer to a CudaCoalescedStack instance on the gpu
     */
    CudaCoalescedStackDeviceHandle(void* struct_device_ptr)
        : struct_dev_ptr(struct_device_ptr) {
    }

    ~CudaCoalescedStackDeviceHandle() {
        if (usable()) {
            free();
        }
    }

    /**
     * @brief Creates num_threads stacks and resizes all of them to new_size, any existing data concerning the stack on the GPU is overriden
     * @param new_size The new size of the stacks
     * @param num_threads The number of threads for which to manage a stack each
     */
    void resize(size_t new_size, size_t num_threads) {
        RelearnGPUException::check(usable(), "CudaCoalescedStackDeviceHandle::resize: Stack was already freed");

        void* new_dev_ptr = cuda_calloc(new_size * num_threads * sizeof(T));
        void* new_sizes_ptr = cuda_calloc(num_threads * sizeof(size_t));

        if (struct_copy.data != nullptr) {
            cudaFree(struct_copy.data);
            gpu_check_last_error();
            cudaDeviceSynchronize();
        }

        if (struct_copy.sizes != nullptr) {
            cudaFree(struct_copy.sizes);
            gpu_check_last_error();
            cudaDeviceSynchronize();
        }

        struct_copy.data = (T*)new_dev_ptr;
        struct_copy.max_size = new_size;
        struct_copy.sizes = (size_t*)new_sizes_ptr;
        struct_copy.num_threads = num_threads;

        cuda_memcpy_to_device(struct_dev_ptr, &struct_copy, sizeof(CudaCoalescedStack<T>), 1);
    }

    /**
     * @brief Frees the data allocated for the stack contents
     */
    void free_contents() {
        RelearnGPUException::check(struct_copy.data != nullptr, "CudaCoalescedStackDeviceHandle::free_contents: No contents to be freed");
        cudaFree(struct_copy.data);
        cudaFree(struct_copy.sizes);
        gpu_check_last_error();
        cudaDeviceSynchronize();
        struct_copy = CudaCoalescedStack<T>{};
        update_struct_copy_to_device();
    }

    /**
     * @brief Frees the data allocated for the stack data structure
     */
    void free() {
        RelearnGPUException::check(usable(), "CudaCoalescedStackDeviceHandle::free: Stack was already freed");
        if (struct_copy.data != nullptr) {
            free_contents();
        }
        struct_dev_ptr = nullptr;
    }

    /**
     * @brief Returns wether or not the handle is usable
     * @returns True if the handle is usable
     */
    bool usable() const {
        return struct_dev_ptr != nullptr;
    }

private:
    void update_struct_copy_from_device() {
        cuda_memcpy_to_host(struct_dev_ptr, &struct_copy, sizeof(CudaCoalescedStack<T>), 1);
    }

    void update_struct_copy_to_device() {
        cuda_memcpy_to_device(struct_dev_ptr, &struct_copy, sizeof(CudaCoalescedStack<T>), 1);
    }

private:
    CudaCoalescedStack<T> struct_copy;
    void* struct_dev_ptr;
};
};