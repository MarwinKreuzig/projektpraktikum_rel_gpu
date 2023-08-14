#pragma once

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
        cudaMemcpyToSymbol(&tgt, src, sizeof(tgt)); \
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

#define cuda_calloc_symbol(symbol, size, value) \
    {                                           \
        void* p;                                \
        gpu_check_last_error();                 \
        cudaGetSymbolAddress(&p, symbol);       \
        gpu_check_last_error();                 \
        cuda_calloc(size, p, value);            \
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

    cuda_memcpy_to_device(devPtr,&devPtrMalloc, sizeof(void*), 1);

    return devPtrMalloc;
}

inline void* cuda_calloc( size_t size, void* devPtr, size_t value) {
    void* devPtrMalloc = cuda_malloc(size, devPtr);
    cudaMemset(devPtrMalloc, value, size);
    gpu_check_last_error();
    return devPtr;
}

inline __device__ size_t block_thread_to_neuron_id(size_t block_id, size_t thread_id, size_t block_size) {
    return block_id * block_size + thread_id;
}

inline int get_number_blocks(int number_threads_per_block, int number_total_threads) {
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
