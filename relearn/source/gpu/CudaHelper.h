#pragma once

#define cuda_available true


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