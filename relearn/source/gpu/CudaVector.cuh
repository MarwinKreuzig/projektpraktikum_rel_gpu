#pragma once

#include "gpu/Commons.cuh"

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
        }

        void resize(size_t new_size) {
            void* new_dev_ptr = cuda_calloc(new_size * sizeof(T));
            
            resize_copy(new_dev_ptr, new_size);
        }

        void resize(size_t new_size, T value) {
            void* new_dev_ptr = cuda_malloc(new_size * sizeof(T));
            set_array((T*)new_dev_ptr, new_size, value);
            
            resize_copy(new_dev_ptr, new_size);            
        }

        void reserve(size_t n) {
            if(n < struct_copy.max_size) {
                
            }
        }

        void copy_to_device(const std::vector<T>& host_data) {
            const auto num_elements = host_data.size();
            if (num_elements > struct_copy.max_size) {
                resize(num_elements, 0);
            }
            cuda_memcpy_to_device(struct_copy.data, (void*)host_data.data(), sizeof(T), num_elements);
            struct_copy.size = num_elements;
            update_struct_copy_to_device();
        }

        void copy_to_host(std::vector<T>& host_data) {
            host_data.resize(struct_copy.size);
            cuda_memcpy_to_host(struct_copy.data, host_data.data(), sizeof(T), struct_copy.size);
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
                    cudaMemcpy(new_dev_ptr, struct_copy.data, struct_copy.size * sizeof(T), cudaMemcpyDeviceToDevice);
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

    };
};