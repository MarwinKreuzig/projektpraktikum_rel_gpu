#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

namespace gpu::Vector {
    struct CudaDouble3 {
        /**
         * Wrapper for the double3 cuda vectory type that defines different operations over it
         */
        
        double3 v;

        __host__ __device__ CudaDouble3(double x, double y, double z)
        : v(make_double3(x, y, z))
        {}

        __host__ __device__ CudaDouble3(double3 d)
        : v(d)
        {}

        __host__ __device__ CudaDouble3(const CudaDouble3& d)
        : v(d.v)
        {}

        __host__ __device__ CudaDouble3()
        : v(make_double3(0.0, 0.0, 0.0))
        {}

        __host__ __device__ bool operator!=(const CudaDouble3& rhs) const {
            return (v.x != rhs.get_x()) || (v.y != rhs.get_y()) || (v.z != rhs.get_z());
        }

        __host__ __device__ bool operator==(const CudaDouble3& rhs) const {
            return (v.x == rhs.get_x()) && (v.y == rhs.get_y()) && (v.z == rhs.get_z());
        }

        __host__ __device__ CudaDouble3 operator-(const int& rhs) const {
            return CudaDouble3((v.x - rhs), (v.y - rhs), (v.z - rhs));
        }

        __host__ __device__ CudaDouble3 operator-(const CudaDouble3& rhs) const {
            return CudaDouble3((v.x - rhs.get_x()), (v.y - rhs.get_y()), (v.z - rhs.get_z()));
        }

        __host__ __device__ CudaDouble3 operator+(const int& rhs) const {
            return CudaDouble3((v.x + rhs), (v.y + rhs), (v.z + rhs));
        }

        __host__ __device__ CudaDouble3 operator+(const CudaDouble3& rhs) const {
            return CudaDouble3((v.x + rhs.get_x()), (v.y + rhs.get_y()), (v.z + rhs.get_z()));
        }

        __host__ __device__ CudaDouble3 operator*(const int& rhs) const {
            return CudaDouble3((v.x * rhs), (v.y * rhs), (v.z * rhs));
        }

        __host__ __device__ CudaDouble3 operator*(const CudaDouble3& rhs) const {
            return CudaDouble3((v.x * rhs.get_x()), (v.y * rhs.get_y()), (v.z * rhs.get_z()));
        }

        __host__ __device__ CudaDouble3 operator/(const int& rhs) const {
            return CudaDouble3((v.x / rhs), (v.y / rhs), (v.z / rhs));
        }

        __host__ __device__ double max() const {
            double max = v.x;
            if (v.y > max) max = v.y;
            if (v.z > max) max = v.z;
            return max;
        }

        __host__ __device__ double min() const {
            double min = v.x;
            if (v.y < min) min = v.y;
            if (v.z < min) min = v.z;
            return min;
        }

        __host__ __device__ double3 to_double3() const {
            return v;
        }

        __host__ __device__ double get_x() const {
            return v.x;
        }

        __host__ __device__ double get_y() const {
            return v.y;
        }

        __host__ __device__ double get_z() const {
            return v.z;
        }
    };
}
