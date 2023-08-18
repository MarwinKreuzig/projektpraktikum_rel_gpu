#pragma once

#define cuda_available true


#if __CUDACC__
    #define GPU_AND_HOST __device__
#else
    #define GPU_AND_HOST
#endif


class CudaHelper {
public:

    static void set_use_cuda(bool u) {
        CudaHelper::use_cuda = u;
    }

    static bool is_cuda_available() {
        return cuda_available && CudaHelper::use_cuda;
    }

    private:
    static inline bool use_cuda = true;
};