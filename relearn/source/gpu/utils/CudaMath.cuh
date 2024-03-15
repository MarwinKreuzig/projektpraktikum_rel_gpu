#pragma once

#include <cuda.h>

class CudaMath {
public:
    /**
     * @brief Calculates the 2-norm of the vector
     * @param vec The vector as double3
     * @return The calculated 2-norm
     */
    static __host__ __device__ double calculate_2_norm(const double3& vec) {
        const auto xx = vec.x * vec.x;
        const auto yy = vec.y * vec.y;
        const auto zz = vec.z * vec.z;

        const auto sum = xx + yy + zz;
        const auto norm = sqrt(sum);
        return norm;
    }

private:
    CudaMath() = default;
};