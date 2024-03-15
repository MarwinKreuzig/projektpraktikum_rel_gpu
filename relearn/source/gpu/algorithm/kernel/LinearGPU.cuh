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

#include "KernelGPUInterface.h"
#include "../../Commons.cuh"
#include "../../utils/GpuTypes.h"
#include "../../utils/Interface.h"
#include "../../structure/CudaArray.cuh"
#include "../../structure/CudaVector.cuh"
#include "../../utils/CudaMath.cuh"
#include "../../structure/CudaDouble3.cuh"

#include <iostream>
#include <cuda.h>

namespace gpu::kernel {

struct LinearDistributionKernel {
    /**
     * Struct representing LinearDistributionKernel on the gpu. Contains most of the data contained by the original cpu class
     */

    double cutoff_point;

    /**
     * @brief Constructs a LinearDistributionKernel on the GPU
     * @param cutoff_point The cutoff_point algorithm parameter
     */
    __device__ LinearDistributionKernel(double cutoff_point)
        : cutoff_point(cutoff_point) { }

    /**
     * @brief Calculates the attractiveness to connect on the basis of the linear distribution
     * @param source_position The source position s
     * @param target_position The target position t
     * @param number_free_elements The linear scaling factor k
     * @return The calculated attractiveness
     */
    __device__ double calculate_attractiveness_to_connect(const gpu::Vector::CudaDouble3& source_position, const gpu::Vector::CudaDouble3& target_position,
        const RelearnGPUTypes::number_elements_type& number_free_elements) {
        if (number_free_elements <= 0) {
            return 0.0;
        }

        const auto cast_number_elements = static_cast<double>(number_free_elements);

        if (isinf(cutoff_point)) {
            return cast_number_elements;
        }

        double3 diff_vector = (source_position - target_position).to_double3();
        const auto x = CudaMath::calculate_2_norm(diff_vector);
        if (x > cutoff_point) {
            return 0.0;
        }

        const auto factor = x / cutoff_point;

        return (1 - factor) * cast_number_elements;
    }
};

class LinearDistributionKernelHandleImpl : public LinearDistributionKernelHandle {
    /**
     * Implementation of the handle for the cpu that controls the gpu object
     */

public:
    /**
     * @brief Constructs the LinearDistributionKernelHandle Implementation
     * @param _dev_ptr The pointer to the LinearDistributionKernel object on the GPU
     */
    LinearDistributionKernelHandleImpl(LinearDistributionKernel* _dev_ptr);

    /**
     * @brief Init function called by the constructor, has to be public in order to be allowed to use device lamdas in it, do not call from outside
     */
    void _init();

    /**
     * @brief Sets the cutoff_point kernel parameter used in calculation
     * @param cutoff_point The cutoff_point kernel parameter used in calculation
     */
    void set_cutoff(const double cutoff_point) override;

    /**
     * @brief Returns the pointer to the LinearDistributionKernel stored on the GPU
     * @return The pointer to the LinearDistributionKernel stored on the GPU
     */
    [[nodiscard]] void* get_device_pointer() override;

private:
    LinearDistributionKernel* device_ptr;

    double* handle_cutoff_point;
};
};