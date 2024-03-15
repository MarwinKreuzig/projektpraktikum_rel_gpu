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
struct CudaStack {
    /**
     * Stack on GPU
     */
    T* data = nullptr;
    size_t size = 0;
    size_t max_size = 0;

    /**
     * @brief Returns the top element of the stack
     * @return The top element of the stack
     */
    __device__ T& top() {
        RelearnGPUException::device_check(size >= 1, "CudaStack::top: Stack is empty");
        return data[size - 1];
    }

    /**
     * @brief Puts an element at the top of the stack
     * @param element The element to be put at the top of the stack
     */
    __device__ void push(const T& element) {
        RelearnGPUException::device_check(size < max_size, "CudaStack::push: Stack is full, max size: %llu", max_size);
        data[size] = element;
        size++;
    }

    /**
     * @brief Removes the top element from the stack
     */
    __device__ void pop() {
        RelearnGPUException::device_check(size >= 1, "CudaStack::top: Stack is empty");
        size--;
    }

    /**
     * @brief Resets the size of the stack to 0
     */
    __device__ void reset() {
        size = 0;
    }

    /**
     * @brief Returns true if the stack is empty, false if not
     * @return True if the stack is empty, false if not
     */
    __device__ bool empty() {
        return size == 0;
    }
};

template <typename T>
class CudaStackDeviceHandle {
    /**
     * A handle to control a CudaStack from the cpu
     */

public:
    CudaStackDeviceHandle() {
        struct_dev_ptr = nullptr;
    }

    /**
     * @param struct_device_pointer Pointer to a CudaStack instance on the gpu
     */
    CudaStackDeviceHandle(CudaStack<T>* struct_device_ptr)
        : struct_dev_ptr((void*)struct_device_ptr) {
    }

    /**
     * @param struct_device_pointer Pointer to a CudaStack instance on the gpu
     */
    CudaStackDeviceHandle(void* struct_device_ptr)
        : struct_dev_ptr(struct_device_ptr) {
    }

    ~CudaStackDeviceHandle() {
        if (usable()) {
            free();
        }
    }

    /**
     * @brief Resizes the stack by increasing its max size, not its current size
     * @param new_size The new max size of the stack
     */
    void resize(size_t new_size) {
        RelearnGPUException::check(usable(), "CudaStackDeviceHandle::resize: Stack was already freed");
        if (new_size > get_max_size()) {
            void* new_dev_ptr = cuda_calloc(new_size * sizeof(T));
            resize_copy(new_dev_ptr, new_size);
        } else {
            struct_copy.max_size = new_size;
            if (new_size < struct_copy.size) {
                struct_copy.size = new_size;
            }
            update_struct_copy_to_device();
        }
    }

    /**
     * @brief Copies the host_data to the stack on the gpu, overriding any current values on the stack
     * @param host_data The host data to override the stack with
     */
    void copy_to_device(const std::vector<T>& host_data) {
        RelearnGPUException::check(usable(), "CudaStackDeviceHandle::copy_to_device: Stack was already freed");
        const auto num_elements = host_data.size();
        if (num_elements > struct_copy.max_size) {
            resize(num_elements);
        }
        cuda_memcpy_to_device(struct_copy.data, (void*)host_data.data(), sizeof(T), num_elements);
        struct_copy.size = num_elements;
        update_struct_copy_to_device();
    }

    /**
     * @brief Copies the stack contents to the host_data vector
     * @param host_data The host data vector to put the stack contents into
     */
    void copy_to_host(std::vector<T>& host_data) {
        RelearnGPUException::check(usable(), "CudaStackDeviceHandle::copy_to_host: Stack was already freed");
        host_data.resize(struct_copy.size);
        cuda_memcpy_to_host(struct_copy.data, host_data.data(), sizeof(T), struct_copy.size);
    }

    /**
     * @brief Copies the host_data to the stack on the gpu, overriding any current values on the stack
     * @param host_data The host data to override the stack with
     * @param num_elements The number of elements to use from the host_data
     */
    void copy_to_device(const T* host_data, size_t num_elements) {
        RelearnGPUException::check(usable(), "CudaStackDeviceHandle::copy_to_device: Stack was already freed");
        if (num_elements > struct_copy.max_size) {
            resize(num_elements);
        }
        cuda_memcpy_to_device(struct_copy.data, (void*)host_data, sizeof(T), num_elements);
        struct_copy.size = num_elements;
        update_struct_copy_to_device();
    }

    /**
     * @brief Copies the stack contents to the host_data array
     * @param host_data The host data to put the stack contents into
     * @param num_elements The number of elements to copy over
     */
    void copy_to_host(T* host_data, size_t num_elements) {
        RelearnGPUException::check(usable(), "CudaStackDeviceHandle::copy_to_host: Stack was already freed");
        cuda_memcpy_to_host(struct_copy.data, host_data, sizeof(T), struct_copy.size);
    }

    /**
     * @brief Frees the data allocated for the stack contents
     */
    void free_contents() {
        RelearnGPUException::check(struct_copy.data != nullptr, "CudaStackDeviceHandle::free_contents: No contents to be freed");
        cudaFree(struct_copy.data);
        gpu_check_last_error();
        cudaDeviceSynchronize();
        struct_copy = CudaStack<T>{};
        update_struct_copy_to_device();
    }

    /**
     * @brief Frees the data allocated for the stack data structure
     */
    void free() {
        RelearnGPUException::check(usable(), "CudaStackDeviceHandle::free: Stack was already freed");
        if (struct_copy.data != nullptr) {
            free_contents();
        }
        struct_dev_ptr = nullptr;
    }

    /**
     * @brief Frees the memory not currently actively occupied by data on the stack
     */
    void minimize_memory_usage() {
        RelearnGPUException::check(usable(), "CudaStackDeviceHandle::minimize_memory_usage: Vector was already freed");
        if (get_max_size() == get_size()) {
            return;
        }

        void* new_dev_ptr = cuda_calloc(get_size() * sizeof(T));
        resize_copy(new_dev_ptr, get_size());
    }

    /**
     * @brief Returns wether or not the handle is usable
     * @returns True if the handle is usable
     */
    bool usable() const {
        return struct_dev_ptr != nullptr;
    }

    /**
     * @brief Returns the size of the stack on the gpu
     * @returns The size of the stack on the gpu
     */
    size_t get_size() const {
        RelearnGPUException::check(usable(), "CudaStackDeviceHandle::get_size: Stack was already freed");
        return struct_copy.size;
    }

    /**
     * @brief Returns the max size of the stack on the gpu
     * @returns The max size of the stack on the gpu
     */
    size_t get_max_size() const {
        RelearnGPUException::check(usable(), "CudaStackDeviceHandle::get_max_size: Stack was already freed");
        return struct_copy.max_size;
    }

    /**
     * @brief Returns the pointer to the data of the stack on the gpu
     * @returns The pointer to the data of the stack on the gpu
     */
    T* data() const {
        RelearnGPUException::check(usable(), "CudaStackDeviceHandle::data: Stack was already freed");
        return struct_copy.data;
    }

    /**
     * @brief Returns true if the stack is empty
     * @returns True if the stack is empty
     */
    bool is_empty() const {
        RelearnGPUException::check(usable(), "CudaStackDeviceHandle::is_empty: Stack was already freed");
        return struct_copy.data == nullptr;
    }

private:
    void update_struct_copy_from_device() {
        cuda_memcpy_to_host(struct_dev_ptr, &struct_copy, sizeof(CudaStack<T>), 1);
    }

    void update_struct_copy_to_device() {
        cuda_memcpy_to_device(struct_dev_ptr, &struct_copy, sizeof(CudaStack<T>), 1);
    }

    void resize_copy(void* new_dev_ptr, size_t new_size) {
        if (struct_copy.data != nullptr) {
            const auto s = struct_copy.size < new_size ? struct_copy.size : new_size;
            cudaMemcpy(new_dev_ptr, struct_copy.data, s * sizeof(T), cudaMemcpyDeviceToDevice);
            gpu_check_last_error();
            cudaDeviceSynchronize();
            cudaFree(struct_copy.data);
            gpu_check_last_error();
            cudaDeviceSynchronize();
        }
        struct_copy.data = (T*)new_dev_ptr;
        struct_copy.max_size = new_size;
        if (new_size < struct_copy.size) {
            struct_copy.size = new_size;
        }

        update_struct_copy_to_device();
    }

private:
    CudaStack<T> struct_copy;
    void* struct_dev_ptr;
};
};