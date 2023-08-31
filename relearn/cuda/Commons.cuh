#pragma once

#include "gpu/Macros.h"
#include "RelearnGPUException.h"

#include <iostream>


#define gpu_check_error(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

#define gpu_check_last_error()                 \
    {                                          \
        const auto error = cudaGetLastError(); \
        gpuAssert(error, __FILE__, __LINE__);  \
    }

#define cuda_copy_to_device(tgt, src)               \
    {                                               \
        gpu_check_last_error();                     \
        cudaMemcpyToSymbol(tgt, &src, sizeof(src)); \
        gpu_check_last_error();                     \
    }

#define cuda_copy_to_host(tgt, src)               \
    {                                               \
        gpu_check_last_error();                     \
        cudaMemcpyFromSymbol(&tgt, src, sizeof(tgt)); \
        gpu_check_last_error();                     \
    }

#define cuda_malloc_symbol(symbol, size)               \
    {                                               \
        void* p; \
        gpu_check_last_error();                     \
        cudaGetSymbolAddress(&p, symbol); \
        gpu_check_last_error();                     \
        cuda_malloc(size, p); \
    }

#define cuda_calloc_symbol(symbol, size) \
    {                                           \
        void* p;                                \
        gpu_check_last_error();                 \
        cudaGetSymbolAddress(&p, symbol);       \
        gpu_check_last_error();                 \
        cuda_calloc(size, p);            \
    }

#define cuda_memcpy_to_host_symbol(symbol, hostPtr, size_type, number_elements) \
    { \
void* p; \
gpu_check_last_error();\
cudaGetSymbolAddress(&p, symbol);\
gpu_check_last_error();\
void* pp;\
cuda_memcpy_to_host(p, &pp, sizeof(void*), 1);\
gpu_check_last_error();\
cuda_memcpy_to_host(pp, hostPtr, size_type, number_elements);\
}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code == cudaSuccess) {
        return;
    }

    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
        exit(code);
    }
}

inline void cuda_memcpy_to_host( void* devPtr, void* hostPtr, size_t size_type, size_t number_elements) {
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cudaMemcpy(hostPtr, devPtr, size_type * number_elements, cudaMemcpyDeviceToHost);
    gpu_check_last_error();
    cudaDeviceSynchronize();
    gpu_check_last_error();
}

inline void cuda_memcpy_to_device(void* devPtr, void* hostPtr, size_t size_type, size_t number_elements) {
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cudaMemcpy(devPtr,hostPtr, size_type * number_elements, cudaMemcpyHostToDevice);
    gpu_check_last_error();
    cudaDeviceSynchronize();
    gpu_check_last_error();
}

inline void* cuda_malloc(size_t size, void* devPtr) {
    void* devPtrMalloc;
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cudaMalloc(&devPtrMalloc, size);
    gpu_check_last_error();
    cudaDeviceSynchronize();

    cuda_memcpy_to_device(devPtr, &devPtrMalloc, sizeof(void*), 1);

    return devPtrMalloc;
}

inline void* cuda_malloc(size_t size) {
    void* devPtrMalloc;
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cudaMalloc(&devPtrMalloc, size);
    gpu_check_last_error();
    cudaDeviceSynchronize();

    return devPtrMalloc;
}

inline void* cuda_calloc(size_t size, void* devPtr) {
    void* devPtrMalloc = cuda_malloc(size, devPtr);
    cudaMemset(devPtrMalloc, 0, size);
    gpu_check_last_error();
    return devPtrMalloc;
}

inline void* cuda_calloc(size_t size) {
    void* devPtrMalloc = cuda_malloc(size);
    cudaMemset(devPtrMalloc, 0, size);
    gpu_check_last_error();
    return devPtrMalloc;
}

inline __device__ size_t block_thread_to_neuron_id(size_t block_id, size_t thread_id, size_t block_size) {
    return block_id * block_size + thread_id;
}

inline int get_number_blocks(int number_threads_per_block, int number_total_threads) {
    RelearnGPUException::check(number_total_threads > 0, "Commmons::get_number_blocks:numer_total_threads is 0");
    RelearnGPUException::check(number_threads_per_block > 0, "Commmons::get_number_blocks:numer_threads_per_block is 0");
    int number_blocks = number_total_threads / number_threads_per_block;
    if (number_total_threads % number_threads_per_block != 0) {
        number_blocks++;
    }

    if(number_blocks == 0) {
        return 1;
    }

    return number_blocks;
}

template<typename func_type>
size_t get_number_threads(func_type kernel, size_t number_neurons) {
    int blocks = 0;
    int threads = 0;
    gpu_check_error(cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel));

    if(number_neurons < threads) {
        return number_neurons;
    }

    return threads;
}

template <typename T>
__global__ void set_array_kernel(T* arr, size_t size, T value) {
    auto thread = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);
    if (thread < size) {
        arr[thread] = value;
    }
}

template <typename T>
__host__ void set_array(T* arr, const size_t size, const T value) {
    const auto threads = get_number_threads(set_array_kernel<T>, size);
    const auto blocks = get_number_blocks(threads, size);
    set_array_kernel<<<blocks,threads>>>(arr,size,value);
    gpu_check_last_error();
    cudaDeviceSynchronize();
    gpu_check_last_error();
}


template <typename T>
__global__ void cuda_set_for_indices_kernel(T* arr, const size_t* indices, size_t size, T value) {
    auto thread = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);
    if (thread < size) {
        const auto key = indices[thread];
        arr[key] = value;
    }
}

template <typename T>
__host__  void cuda_set_for_indices(T* arr, const size_t* indices,const size_t size, T value) {
    RelearnGPUException::check(size > 0, "Commmons::Cuda_set_for_indices: Size of indices is 0");
    void* dev_ptr = cuda_malloc(sizeof(size_t) * size);
    cuda_memcpy_to_device(dev_ptr, (void*)indices, sizeof(size_t), size);
    gpu_check_last_error();
    cudaDeviceSynchronize();
    const auto threads = get_number_threads(cuda_set_for_indices_kernel<T>, size);
    const auto blocks = get_number_blocks(threads, size);
    cuda_set_for_indices_kernel<T><<<blocks,threads>>>(arr,(size_t*)dev_ptr,size,value);
    gpu_check_last_error();
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cudaFree(dev_ptr);
    gpu_check_last_error();
}

template <typename T, typename ... Args>
__global__ void init_class_kernel(void** ptr, Args ... constructor_args) {
    auto* constant = new T(constructor_args...);
    *ptr = (void*) constant;
}

template <typename T, typename ... Args>
inline void* init_class_on_device(Args ... constructor_args) {
    void** ptr = (void**) cuda_malloc(sizeof(void**));
    init_class_kernel<T, Args...><<<1,1>>>(ptr, std::forward<Args>(constructor_args)...);

    cudaDeviceSynchronize();
    gpu_check_last_error();

    void* dev_ptr_class;
    cuda_memcpy_to_host(ptr, &dev_ptr_class, sizeof(void**), 1);

    return dev_ptr_class;


}