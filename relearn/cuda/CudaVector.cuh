#pragma once

#include "Commons.cuh"


namespace gpu::Vector {
template<typename T>
    class CudaVector {
        public:

        struct meta {
        T* data = nullptr;
        size_t size = 0;
        size_t max_size = 0;
        } ;

        __device__ CudaVector() {}

        __device__ void resize(size_t new_size) {
            if(new_size > get_size()){
                void* new_dev_ptr = device_calloc(new_size * sizeof(T));
                resize_copy(new_dev_ptr, new_size);
            } else {
                meta_data.size = new_size;
            }
            }

        __device__ void resize(size_t new_size, T value) {
            if (new_size > get_size()) {
                void* new_dev_ptr = device_malloc(new_size * sizeof(T));
                device_set_array((T*)new_dev_ptr, new_size, value);

                resize_copy(new_dev_ptr, new_size);
            } else {
                meta_data.size = new_size;
            }
            }

        __device__ void fill(T value) {
            RelearnGPUException::check(!is_empty(), "CudaVector::fill: Cannot fill an empty vector");
            set_array(meta_data.data, meta_data.size, value);
        }

        __device__ void fill(size_t begin, size_t end,T value) {
            RelearnGPUException::device_check(!is_empty(), "CudaVector::fill: Cannot fill an empty vector");
            RelearnGPUException::device_check(begin<end, "CudaVector::fill: End {} < begin {}", end,begin);
            T* p = meta_data.data;
            p += begin;
            size_t size = end-begin;
            device_set_array(meta_data.data, size, value);
        }

        __device__ void reserve(size_t n) {
            RelearnGPUException::fail("TODO");
        }


        __device__ T& operator[](size_t index) {
            return meta_data.data[index];
        }

        __device__ void minimize_memory_usage() {
            if(get_max_size() == get_size()) {
                return;
            }
            
            void* new_dev_ptr = device_calloc(get_size() * sizeof(T));
            resize_copy(new_dev_ptr, get_size());
        }        

        __device__ size_t get_size() const {
            return meta_data.size;
        }

        /*__device__ void set(const size_t* indices, size_t num_indices, T value) {
            RelearnGPUException::device_check(num_indices>0, "CudaVector::set: Num indices is 0");
            device_set_for_indices(meta_data.data,indices,meta_data.size,value);
        }*/

        __device__ size_t get_max_size() const {
            return meta_data.max_size;
        }

        __device__ T* data_() const {
            return meta_data.data;
        }

        __device__ bool is_empty() const {
            return meta_data.size == 0;
        }
meta meta_data;
        private:
        


        __device__ void resize_copy(void* new_dev_ptr, size_t new_size) {
                if (meta_data.data != nullptr) {
                    const auto s = meta_data.size < new_size ? meta_data.size : new_size;
                    cudaMemcpyAsync(new_dev_ptr, meta_data.data, s * sizeof(T), cudaMemcpyDeviceToDevice);
                    cudaFree(meta_data.data);
                }
                meta_data.data = (T*) new_dev_ptr;
                meta_data.max_size = new_size;
                meta_data.size = new_size;
            }

        
    };
/*
    template<typename T>
    class CudaVectorDeviceHandle {

        public:

        CudaVectorDeviceHandle(void* struct_device_ptr) : struct_dev_ptr(struct_device_ptr) {
            update_meta_data_from_device();
        }

        CudaVectorDeviceHandle() {
        }

        ~CudaVectorDeviceHandle() {
        }

        void print_content() {
            std::vector<T> cpy;
            cpy.resize(meta_data.size);
            copy_to_host(cpy);
            for(const auto& e:cpy) {
                std:: cout << e << ", ";
            }
        }

        void copy_to_host(std::vector<T>& host_data) {
            update_meta_data_from_device();
            RelearnGPUException::check(meta_data.data != nullptr, "CudaVector::copy_to_host: Vector is empty");
            host_data.resize(meta_data.size);
            cuda_memcpy_to_host(meta_data.data, host_data.data(), sizeof(T), meta_data.size);
        }

        void copy_to_host(T* host_data, size_t num_elements) {
            update_meta_data_from_device();
            RelearnGPUException::check(meta_data.data != nullptr, "CudaVector::copy_to_host: Vector is empty");
            cuda_memcpy_to_host(meta_data.data, host_data, sizeof(T), meta_data.size);
        }      

        size_t get_size() const {
            return meta_data.size;
        }

        size_t get_max_size() const {
            return meta_data.max_size;
        }

        T* data() const {
            return meta_data.data;
        }

        bool is_empty() const {
            return meta_data.size == 0;
        }

        void update_meta_data_from_device() {
            this->meta_data = execute_and_copy<typename CudaVector<T>::meta>([=] __device__ (void* struct_dev_ptr)  -> typename CudaVector<T>::meta { return ((CudaVector<T>*) struct_dev_ptr)->meta_data;}, struct_dev_ptr);
        }

        private:
        typename CudaVector<T>::meta meta_data;
        void* struct_dev_ptr;

    };*/
};