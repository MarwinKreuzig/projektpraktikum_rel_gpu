#pragma once

#define cuda_available true


class CudaHelper {
public:
    static bool is_cuda_available() {
        return cuda_available;
    }
};