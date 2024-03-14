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

struct WeibullDistributionKernel {
/**
* Struct representing WeibullDistributionKernel on the gpu. Contains most of the data contained by the original cpu class
*/
    double k;
    double b;

    /**
     * @brief Constructs a WeibullDistributionKernel on the GPU 
     * @param k The k algorithm parameter
     * @param b The b algorithm parameter
     */
    __device__ WeibullDistributionKernel(double k, double b)
        : k(k), b(b) {}

    /**
     * @brief Calculates the attractiveness to connect on the basis of the Weibull distribution
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

        const double factor_1 = number_free_elements * b * k;

        double3 diff_vector = (source_position - target_position).to_double3();
        const double x = CudaMath::calculate_2_norm(diff_vector);

        const double factor_2 = pow(x, k - 1);
        const double exponent = -b * factor_2 * x;
        const double factor_3 = exp(exponent);

        const double result = factor_1 * factor_2 * factor_3;

        return result;
    }
    
};

class WeibullDistributionKernelHandleImpl : public WeibullDistributionKernelHandle {
/**
* Implementation of the handle for the cpu that controls the gpu object
*/

public:
     /**
     * @brief Constructs the WeibullDistributionKernelHandle Implementation
     * @param _dev_ptr The pointer to the WeibullDistributionKernel object on the GPU
     */
    WeibullDistributionKernelHandleImpl(WeibullDistributionKernel* _dev_ptr);

    /**
    * @brief Init function called by the constructor, has to be public in order to be allowed to use device lamdas in it, do not call from outside
    */
    void _init();

    /**
    * @brief Sets the k kernel parameter used in calculation
    * @param k The k kernel parameter used in calculation
    */
    void set_k(const double k) override;

    /**
    * @brief Sets the b kernel parameter used in calculation
    * @param b The b kernel parameter used in calculation
    */
    void set_b(const double b) override;

    /**
    * @brief Returns the pointer to the WeibullDistributionKernel stored on the GPU
    * @return The pointer to the WeibullDistributionKernel stored on the GPU
    */
    [[nodiscard]] void* get_device_pointer() override;

private:
    WeibullDistributionKernel* device_ptr;

    double* handle_k;
    double* handle_b;
};
};