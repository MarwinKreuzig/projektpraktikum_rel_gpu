#pragma once

#include "gpu/Macros.h"
#include "RelearnGPUException.h"

#include <iostream>
#include <functional>

#include <cuda.h>


#define gpu_check_error(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

#define device_check_error(ans) \
{ deviceAssert((ans), __FILE__, __LINE__); }

#define gpu_check_last_error()                 \
    {                                          \
        const auto error = cudaGetLastError(); \
        gpuAssert(error, __FILE__, __LINE__);  \
    }

#define device_check_last_error()                 \
{                                          \
    const auto error = cudaGetLastError(); \
    deviceAssert(error, __FILE__, __LINE__);  \
}


#define cuda_copy_to_device(tgt, src)               \
    {                                               \
        gpu_check_last_error();                     \
        cudaMemcpyToSymbol(tgt, &src, sizeof(src)); \
        cudaDeviceSynchronize(); \
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

inline __device__ void deviceAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code == cudaSuccess) {
        return;
    }

    printf("GPU device assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
        __trap();
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

inline __device__ void* device_malloc(size_t size) {
    //#warning Do not use malloc in device code
    void* devPtrMalloc;
    cudaMalloc(&devPtrMalloc, size);
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

inline __device__ void* device_calloc(size_t size) {
    void* devPtrMalloc = device_malloc(size);
    cudaMemsetAsync(devPtrMalloc, 0, size);
    return devPtrMalloc;
}

inline __device__ size_t block_thread_to_neuron_id(size_t block_id, size_t thread_id, size_t block_size) {
    return block_id * block_size + thread_id;
}

inline __device__ __host__  int get_number_blocks(int number_threads_per_block, int number_total_threads) {
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
__host__ size_t get_number_threads(func_type kernel, size_t number_neurons) {
    int blocks = 0;
    int threads = 0;
    const auto e = cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel);
    gpu_check_error(e);

    if(number_neurons < threads) {
        return number_neurons;
    }

    return threads;
}

template<typename func_type>
__device__ size_t device_get_number_threads(func_type kernel, size_t number_neurons) {
    int blocks = 0;
    int threads = 0;
    const auto e = cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel);
    device_check_error(e);

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
__device__ void device_set_array(T* arr, const size_t size, const T value) {
    const auto threads = device_get_number_threads(set_array_kernel<T>, size);
    const auto blocks = get_number_blocks(threads, size);
    set_array_kernel<<<blocks,threads>>>(arr,size,value);
    device_check_last_error();
    //cudaDeviceSynchronize();
    device_check_last_error();
}


template <typename T>
__global__ void cuda_set_for_indices_kernel(T* arr, const size_t* indices, size_t size, size_t array_size, T value) {
    RelearnGPUException::device_check(size > 0,"Commons::cuda_set_for_indices_kernel: Size is 0");
    auto thread = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);
    if (thread < size) {
        const auto key = indices[thread];
        RelearnGPUException::device_check(key < array_size, "Commons::cuda_set_for_indices_kernel: Index {} is out of bound {}", key, array_size);
        arr[key] = value;
    }
}

template <typename T>
__host__  void cuda_set_for_indices(T* arr, const size_t* indices,const size_t size, size_t array_size, T value) {
    RelearnGPUException::check(size > 0, "Commmons::Cuda_set_for_indices: Size of indices is 0");
    void* dev_ptr = cuda_malloc(sizeof(size_t) * size);
    cuda_memcpy_to_device(dev_ptr, (void*)indices, sizeof(size_t), size);
    gpu_check_last_error();
    cudaDeviceSynchronize();
    const auto threads = get_number_threads(cuda_set_for_indices_kernel<T>, size);
    const auto blocks = get_number_blocks(threads, size);
    cuda_set_for_indices_kernel<T><<<blocks,threads>>>(arr,(size_t*)dev_ptr,size, array_size,value);
    gpu_check_last_error();
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cudaFree(dev_ptr);
    gpu_check_last_error();
}


template <typename T>
__device__  void device_set_for_indices(T* arr, const size_t* indices,const size_t size, T value) {
    RelearnGPUException::device_check(size > 0, "Commmons::Cuda_set_for_indices: Size of indices is 0");
    device_check_last_error();
    const auto threads = device_get_number_threads(cuda_set_for_indices_kernel<T>, size);
    const auto blocks = get_number_blocks(threads, size);
    printf("Set %i\n", size);
    cuda_set_for_indices_kernel<T><<<blocks,threads>>>(arr,indices,size,value);
    device_check_last_error();
    cudaDeviceSynchronize();
    device_check_last_error();
}

template <typename T, typename ... Args>
__global__ void init_class_kernel(void* ptr, Args ... constructor_args) {
    new (ptr) T(constructor_args...);
}

template <typename T, typename ... Args>
inline T* init_class_on_device(Args ... constructor_args) {

    cudaDeviceSynchronize();
    gpu_check_last_error();

    void* dev_ptr = cuda_malloc(sizeof(T));

    init_class_kernel<T><<<1,1>>>(dev_ptr, constructor_args...);

    cudaDeviceSynchronize();
    gpu_check_last_error();

    
    return (T*) dev_ptr;
}


template<typename Function, typename ... Args>
__global__ void cuda_generic_kernel(Function f,Args... args)
    {
        f(args...);
    }

template<typename T,typename F, typename ...Args>
__global__ void cuda_generic_kernel2(T* ptr,F f, Args... args)
    {
        *ptr = f(args...);
    }

template<typename T,typename F, typename ...Args>
T execute_and_copy(F f, Args... args ) {
    T* dev_ptr = (T*) cuda_malloc(sizeof(T));

    cuda_generic_kernel2<<<1,1>>>(dev_ptr, f, args...);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    T host_data;
    cuda_memcpy_to_host(dev_ptr, &host_data, sizeof(T), 1);

    cudaFree(dev_ptr);
    gpu_check_last_error();

    return host_data;
}