#include "Commons.cuh"


inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort) {
    if (code == cudaSuccess) {
        return;
    }

    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
        exit(code);
    }
}

inline __device__ void deviceAssert(cudaError_t code, const char* file, int line, bool abort) {
    if (code == cudaSuccess) {
        return;
    }

    printf("GPU device assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
        __trap();
    }
}

inline void cuda_memcpy_to_host(void* devPtr, void* hostPtr, size_t size_type, size_t number_elements) {
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
    cudaMemcpy(devPtr, hostPtr, size_type * number_elements, cudaMemcpyHostToDevice);
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
    // #warning Do not use malloc in device code
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

inline __device__ __host__ int get_number_blocks(int number_threads_per_block, int number_total_threads) {
    int number_blocks = number_total_threads / number_threads_per_block;
    if (number_total_threads % number_threads_per_block != 0) {
        number_blocks++;
    }

    if (number_blocks == 0) {
        return 1;
    }

    return number_blocks;
}