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
struct CudaArray {
    /**
     * Vector on utils that can only be accessed element wise. All other methods are managed by the cpu
     */
    T* data = nullptr;
    size_t size = 0;
    size_t max_size = 0;

    __device__ T& operator[](size_t index) {
        RelearnGPUException::device_check(index < size, "CudaVector::[]: Out of bounds array access");
        return data[index];
    }
};

template <typename T>
class CudaArrayDeviceHandle {
    /**
     * A handle to control a CudaArray from the cpu
     * Internally, it keeps a pointer to a CudaArray allocated on the GPU, called the struct_dev_ptr, as
     * well as a copy in regular memory called the struct_copy.
     */

public:
    /**
     * @brief Creates a CudaArrayDeviceHandle with an empty struct_dev_ptr
     */
    CudaArrayDeviceHandle() {
        struct_dev_ptr = nullptr;
    }

    /**
     * @brief Creates a CudaArrayDeviceHandle with struct_device_pointer as struct_dev_ptr
     * @param struct_device_pointer Pointer to an allocated CudaArray instance
     */
    CudaArrayDeviceHandle(CudaArray<T>* struct_device_ptr)
        : struct_dev_ptr((void*)struct_device_ptr) {
    }

    CudaArrayDeviceHandle(void* struct_device_ptr)
        : struct_dev_ptr(struct_device_ptr) {
    }

    /**
     * @brief This Deconstructor frees memory if struct_dev_ptr != nullptr
     */
    ~CudaArrayDeviceHandle() {
        if (usable()) {
            free();
        }
    }

    /**
     * @brief Resizes the CudaArray
     * @param new_size New size of the CudaArray
     */
    void resize(size_t new_size) {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        if (new_size > get_size()) {
            void* new_dev_ptr = cuda_calloc(new_size * sizeof(T));
            resize_copy(new_dev_ptr, new_size);
        } else {
            struct_copy.size = new_size;
            update_struct_copy_to_device();
        }
    }

    /**
     * @brief Resizes the CudaArray and sets all indices to value
     * @param new_size New size of the CudaArray
     * @param value New type of the CudaArray
     */
    void resize(size_t new_size, T value) {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        if (new_size > get_size()) {
            void* new_dev_ptr = cuda_malloc(new_size * sizeof(T));
            set_array((T*)new_dev_ptr, new_size, value);

            resize_copy(new_dev_ptr, new_size);
        } else {
            struct_copy.size = new_size;
            update_struct_copy_to_device();
        }
    }

    /**
     * @brief Sets num_indices indices from indices to value
     * @param indices The starting index (inclusive) within the vector to set
     * @param num_indices Number of indices to set
     * @param value New value for specified indices
     */
    void set(const size_t* indices, size_t num_indices, T value) {
        RelearnGPUException::check(num_indices > 0, "CudaVector::set: Num indices is 0");
        cuda_set_for_indices(struct_copy.data, indices, num_indices, struct_copy.size, value);
    }

    /**
     * @brief Fills an empty CudaArray with value
     * @param value New value for all indices
     */
    void fill(T value) {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        RelearnGPUException::check(!is_empty(), "CudaVector::fill: Cannot fill an empty vector");
        set_array(struct_copy.data, struct_copy.size, value);
    }

    /**
     * @brief Sets the indices from begin to end with value
     * @param begin The starting index (inclusive) within the array to fill
     * @param end The ending index (exclusive) within the array to fill
     * @param value The value to fill the specified range with
     */
    void fill(size_t begin, size_t end, T value) {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        RelearnGPUException::check(!is_empty(), "CudaVector::fill: Cannot fill an empty vector");
        RelearnGPUException::check(begin < end, "CudaVector::fill: End {} < begin {}", end, begin);
        T* p = struct_copy.data;
        p += begin;
        size_t size = end - begin;
        set_array(struct_copy.data, size, value);
    }

    /**
     * @brief Prints the content of the array in the console
     */
    void print_content() {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        std::vector<T> cpy;
        cpy.resize(struct_copy.size);
        copy_to_host(cpy);
        for (const auto& e : cpy) {
            std::cout << e << ", ";
        }
    }

    /**
     * @brief Copys data from host to a CudaVector, overwriting the current data.
     * @param host_data Data from the host-side
     */
    void copy_to_device(const std::vector<T>& host_data) {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        const auto num_elements = host_data.size();
        if (num_elements > struct_copy.max_size) {
            resize(num_elements);
        }
        cuda_memcpy_to_device(struct_copy.data, (void*)host_data.data(), sizeof(T), num_elements);
        struct_copy.size = num_elements;
        update_struct_copy_to_device();
    }

    /**
     * @brief Copys data from host to a CudaVector with an offset. host_data will be copied to the indices offset to offset + host_data.size()
     * @param host_data Data from the host-side
     * @param offset Offset within a CudaVector
     */
    void copy_to_device_at(const std::vector<T>& host_data, size_t offset) {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        const auto furthest_element_pos = offset + host_data.size();
        if (furthest_element_pos > struct_copy.max_size) {
            resize(furthest_element_pos);
            struct_copy.size = furthest_element_pos;
        }
        cuda_memcpy_to_device(struct_copy.data + offset, (void*)host_data.data(), sizeof(T), host_data.size());
        update_struct_copy_to_device();
    }

    /**
     * @brief Copys data from gpu to host
     * @param host_data Destination of data transfer
     */
    void copy_to_host(std::vector<T>& host_data) const {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        host_data.resize(struct_copy.size);
        cuda_memcpy_to_host(struct_copy.data, host_data.data(), sizeof(T), struct_copy.size);
    }

    /**
     * @brief Copies num_elements from host to gpu
     * @param host_data Data from the host
     * @param num_elements Number of elements to copy
     */
    void copy_to_device(const T* host_data, size_t num_elements) {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        if (num_elements > struct_copy.max_size) {
            resize(num_elements);
        }
        cuda_memcpy_to_device(struct_copy.data, (void*)host_data, sizeof(T), num_elements);
        struct_copy.size = num_elements;
        update_struct_copy_to_device();
    }

    /**
     * @brief Copys num_elements from gpu to host
     * @param host_data Destination of data transfer
     * @param num_elements Number of elements to copy
     */
    void copy_to_host(T* host_data, size_t num_elements) {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        // host_data.resize(struct_copy.size);
        cuda_memcpy_to_host(struct_copy.data, host_data, sizeof(T), struct_copy.size);
    }

    /**
     * @brief Copys num_elements from gpu to host from begin to end index, begin is inclusive, end exclusive
     * @param host_data Destination of data transfer
     * @param begin Index of the first element to copy
     * @param end Index of the element before the last element to copy
     */
    void copy_to_host_from_to(T* host_data, size_t begin, size_t end) {
        RelearnGPUException::check(usable(), "CudaVector::copy_to_host_from_to: Vector was already freed");
        RelearnGPUException::check(end > begin, "CudaVector::copy_to_host_from_to: end index is not greater than begin index");

        cuda_memcpy_to_host(struct_copy.data + begin, host_data, sizeof(T), end - begin);
    }

    /**
     * @brief Frees the memory allocated for the contents of a CudaVector
     */
    void free_contents() {
        RelearnGPUException::check(struct_copy.data != nullptr, "CudaVector::free_contents: No contents to be freed");
        cudaFree(struct_copy.data);
        gpu_check_last_error();
        cudaDeviceSynchronize();
        struct_copy = CudaArray<T>{};
        update_struct_copy_to_device();
    }

    /**
     * @brief Frees the memory allocated for a CudaVector
     */
    void free() {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        if (struct_copy.data != nullptr) {
            free_contents();
        }
        struct_dev_ptr = nullptr;
    }

    /**
     * @brief Minimizing memory usage by deleting empty indices
     */
    void minimize_memory_usage() {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        if (get_max_size() == get_size()) {
            return;
        }

        void* new_dev_ptr = cuda_calloc(get_size() * sizeof(T));
        resize_copy(new_dev_ptr, get_size());
    }

    /**
     * @brief Checks if the handle contains a valid pointer to a CudaArray
     * @return Is struct_dev_ptr valid
     */
    bool usable() const {
        return struct_dev_ptr != nullptr;
    }

    /**
     * @brief Returns actual size of CudaArray
     * @return CudaArray size
     */
    size_t get_size() const {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        return struct_copy.size;
    }

    /**
     * @brief Returns the maximum size of CudaArray
     * @return CudaArray maximum size
     */
    size_t get_max_size() const {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        return struct_copy.max_size;
    }

    /**
     * @brief Returns raw data of CudaArray
     * @return Raw CudaArray data
     */
    T* data() const {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        return struct_copy.data;
    }

    /**
     * @brief Checks if CudaArray is empty
     * @return Is CudaArray empty
     */
    bool is_empty() const {
        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
        return struct_copy.data == nullptr;
    }

private:
    /**
     * @brief Updates the host struct_dev_ptr
     */
    void update_struct_copy_from_device() {
        cuda_memcpy_to_host(struct_dev_ptr, &struct_copy, sizeof(CudaArray<T>), 1);
    }

    /**
     * @brief Updates the gpu CudaArray
     */
    void update_struct_copy_to_device() {
        cuda_memcpy_to_device(struct_dev_ptr, &struct_copy, sizeof(CudaArray<T>), 1);
    }

    /**
     * @brief Resizes CudaArray on gpu and updates struct_copy on the cpu
     * @param new_dev_ptr Destination of copy
     * @param new_size New size of CudaArray
     */
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
        struct_copy.size = new_size;

        update_struct_copy_to_device();
    }

private:
    CudaArray<T> struct_copy;
    void* struct_dev_ptr;
};

template <typename T>
static CudaArrayDeviceHandle<T> create_array_in_device_memory() {
    const auto size = sizeof(CudaArray<T>);
    void* devPtr = cuda_malloc(size);
    CudaArray<T> arr;
    cuda_memcpy_to_device(devPtr, &arr, size, 1);
    return CudaArrayDeviceHandle<T>((CudaArray<T>*)devPtr);
}

#define gpu_get_handle_for_device_symbol(T, handle, symbol) \
    {                                                       \
        void* p;                                            \
        gpu_check_last_error();                             \
        cudaGetSymbolAddress(&p, symbol);                   \
        gpu_check_last_error();                             \
        handle = gpu::Vector::CudaArrayDeviceHandle<T>(p);  \
    }

};
