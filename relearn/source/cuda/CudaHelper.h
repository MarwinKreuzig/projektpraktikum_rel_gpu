#pragma once

#define cuda_available

class CudaHelper {
public:
    static bool is_cuda_available() {
        return true;
    }
};