#include <iostream>


#define gpu_check_error(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

#define gpu_check_last_error()                 \
    {                                          \
        const auto error = cudaGetLastError(); \
        gpuAssert(error, __FILE__, __LINE__);  \
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
