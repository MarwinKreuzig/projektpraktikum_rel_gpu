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

struct GammaDistributionKernel {
    /**
    * Struct representing GammaDistributionKernel on the gpu. Contains most of the data contained by the original cpu class
    */

    double k;
    double theta;

    // These depend on k and theta
    double gamma_divisor_inv;
    double theta_divisor;
    
    /**
     * @brief Constructs a GammaDistributionKernel on the GPU 
     * @param k The k algorithm parameter
     * @param theta The theta algorithm parameter
     */
    __device__ GammaDistributionKernel(double k, double theta)
        : k(k), theta(theta) {
        gamma_divisor_inv = 1.0 / (tgamma(k) * pow(theta, k));
        theta_divisor = -1.0 / theta;
    }
    
    /**
     * @brief Calculates the attractiveness to connect on the basis of the gamma distribution
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

        const double factor_1 = number_free_elements * gamma_divisor_inv;

        double3 diff_vector = (source_position - target_position).to_double3();
        const double x = CudaMath::calculate_2_norm(diff_vector);

        const double factor_2 = pow(x, k - 1);
        const double factor_3 = exp(x * theta_divisor);

        const double result = factor_1 * factor_2 * factor_3;

        return result;
    }
    
};

class GammaDistributionKernelHandleImpl : public GammaDistributionKernelHandle {
/**
* Implementation of the handle for the cpu that controls the gpu object
*/

public:
     /**
     * @brief Constructs the GammaDistributionKernelHandle Implementation
     * @param _dev_ptr The pointer to the GammaDistributionKernel object on the GPU
     */
    GammaDistributionKernelHandleImpl(GammaDistributionKernel* _dev_ptr);

    /**
    * @brief Init function called by the constructor, has to be public in order to be allowed to use device lamdas in it, do not call from outside
    */
    void _init();
    
    /**
    * @brief Sets the k and theta kernel parameters used in calculation
    * @param k The k kernel parameter used in calculation
    * @param theta The theta kernel parameter used in calculation
    */
    void set_k_theta(const double k, const double theta) override;

    /**
    * @brief Returns the pointer to the GammaDistributionKernel stored on the GPU
    * @return The pointer to the GammaDistributionKernel stored on the GPU
    */
    [[nodiscard]] void* get_device_pointer() override;

private:
    GammaDistributionKernel* device_ptr;

    double* handle_k;
    double* handle_theta;

    double* handle_gamma_divisor_inv;
    double* handle_theta_divisor;
};
};