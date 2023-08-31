#pragma once

#include "Commons.cuh"
#include "RelearnGPUException.h"

#include <iostream>
#include <vector>

namespace gpu::Vector {
    template<typename T>
    struct CudaArray {
        T* data = nullptr;
        size_t size = 0;
        size_t max_size = 0;

        __device__ T& operator[](size_t index) {
            return data[index];
        }
    };

    template<typename T>
    class CudaArrayDeviceHandle {

        public:
        CudaArrayDeviceHandle(CudaArray<T>& struct_device_symbol) {
            void* p;
            gpu_check_last_error();
            cudaGetSymbolAddress(&p, struct_device_symbol);
            gpu_check_last_error();
            struct_dev_ptr = (CudaArray<T>*)p;
            is_device_symbol = true;
        }

        CudaArrayDeviceHandle(CudaArray<T>* struct_device_ptr) : struct_dev_ptr(struct_device_ptr) {
            is_device_symbol = false;
        }

        ~CudaArrayDeviceHandle() {
            if(usable()){
            free();
            }
        }

        void resize(size_t new_size) {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            if(new_size > get_size()){
                void* new_dev_ptr = cuda_calloc(new_size * sizeof(T));
                resize_copy(new_dev_ptr, new_size);
            } else {
                struct_copy.size = new_size;
                update_struct_copy_to_device();
            }
            }

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

        void fill(T value) {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            RelearnGPUException::check(!is_empty(), "CudaVector::fill: Cannot fill an empty vector");
            set_array(struct_copy.data, struct_copy.size, value);
        }

        void fill(size_t begin, size_t end,T value) {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            RelearnGPUException::check(!is_empty(), "CudaVector::fill: Cannot fill an empty vector");
            RelearnGPUException::check(begin<end, "CudaVector::fill: End {} < begin {}", end,begin);
            T* p = struct_copy.data;
            p += begin;
            size_t size = end-begin;
            set_array(struct_copy.data, size, value);
        }

        void print_content() {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            std::vector<T> cpy;
            cpy.resize(struct_copy.size);
            copy_to_host(cpy);
            for(const auto& e:cpy) {
                std:: cout << e << ", ";
            }
        }

        void reserve(size_t n) {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            if(n < struct_copy.max_size) {
                
            }
        }

        void copy_to_device(const std::vector<T>& host_data) {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            const auto num_elements = host_data.size();
            if (num_elements > struct_copy.max_size) {
                resize(num_elements, 0);
            }
            cuda_memcpy_to_device(struct_copy.data, (void*)host_data.data(), sizeof(T), num_elements);
            struct_copy.size = num_elements;
            update_struct_copy_to_device();
        }

        void copy_to_host(std::vector<T>& host_data) {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            host_data.resize(struct_copy.size);
            cuda_memcpy_to_host(struct_copy.data, host_data.data(), sizeof(T), struct_copy.size);
        }

        void copy_to_device(const T* host_data, size_t num_elements) {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            if (num_elements > struct_copy.max_size) {
                resize(num_elements, 0);
            }
            cuda_memcpy_to_device(struct_copy.data, (void*)host_data, sizeof(T), num_elements);
            struct_copy.size = num_elements;
            update_struct_copy_to_device();
        }

        void copy_to_host(T* host_data, size_t num_elements) {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            //host_data.resize(struct_copy.size);
            cuda_memcpy_to_host(struct_copy.data, host_data, sizeof(T), struct_copy.size);
        }

        void free_contents() {
            RelearnGPUException::check(struct_copy.data != nullptr, "CudaVector::free_contents: No contents to be freed");
            cudaFree(struct_copy.data);
            gpu_check_last_error();
            cudaDeviceSynchronize();
            struct_copy = CudaArray<T>{};
            update_struct_copy_to_device();
        }

        void free() {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            if(struct_copy.data !=nullptr){
                free_contents();
            }
            if(!is_device_symbol){
                cudaFree(struct_dev_ptr);
                gpu_check_last_error();
            }
            struct_dev_ptr = nullptr;
        }

        void minimize_memory_usage() {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            if(get_max_size() == get_size()) {
                return;
            }
            
            void* new_dev_ptr = cuda_calloc(get_size() * sizeof(T));
            resize_copy(new_dev_ptr, get_size());
        }        

        bool usable() const {
            return struct_dev_ptr != nullptr;
        }

        size_t get_size() const {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            return struct_copy.size;
        }

        size_t get_max_size() const {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            return struct_copy.max_size;
        }

        T* data() const {
                        RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            return struct_copy.data;
        }

        bool is_empty() const {
            RelearnGPUException::check(usable(), "CudaVector::free: Vector was already freed");
            return struct_copy.data == nullptr;
        }

        private:
            void update_struct_copy_from_device() {
            cuda_memcpy_to_host(struct_dev_ptr, &struct_copy, sizeof(CudaArray<T>), 1);
            }

            void update_struct_copy_to_device() {
            cuda_memcpy_to_device(struct_dev_ptr, &struct_copy, sizeof(CudaArray<T>), 1);
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
                struct_copy.data = (T*) new_dev_ptr;
                struct_copy.max_size = new_size;
                struct_copy.size = new_size;

                update_struct_copy_to_device();
            }

        private:
        CudaArray<T> struct_copy;
        CudaArray<T>* struct_dev_ptr;

        bool is_device_symbol;

    };

    template<typename T>
    static CudaArrayDeviceHandle<T> create_array_in_device_memory() {
        const auto size = sizeof(CudaArray<T>);
        void* devPtr = cuda_malloc(size);
        CudaArray<T> arr;
        cuda_memcpy_to_device(devPtr, &arr, size, 1);
        return CudaArrayDeviceHandle<T>((CudaArray<T>*)devPtr);
    }

/*
    template<typename T>
    __global__ void free_pointers(T* dev_ptr, size_t size) {
            const auto thread_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);
            if(thread_id < size) {
                T* p = dev_ptr[thread_id];
                if(p != nullptr) {
                cudaFree(p);
                }
            }
        }

    template<typename T>
    class CudaPtrArrayDeviceHandle : public CudaArrayDeviceHandle<T>  {
        public:
        CudaPtrArrayDeviceHandle(CudaArray<T>& struct_device_symbol) : CudaArrayDeviceHandle<T>(struct_device_symbol) {}

        CudaPtrArrayDeviceHandle(CudaArray<T>* _struct_device_ptr) : CudaArrayDeviceHandle<T>(_struct_device_ptr) {}

        ~CudaPtrArrayDeviceHandle() {
            free();
        }


        void free() override {
            if(get_size() > 0){
            const auto num_threads = get_number_threads(free_pointers<T>, this->get_max_size());
            const auto num_blocks = get_number_blocks(num_threads, this->get_max_size());
            free_pointers<T><<<num_blocks,num_threads>>>(this->data(), this->get_max_size());
            }

            CudaArrayDeviceHandle<T>::free();
        }

    };
    */
};