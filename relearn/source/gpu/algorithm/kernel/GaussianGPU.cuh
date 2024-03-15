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

struct GaussianDistributionKernel {
/**
* Struct representing GaussianDistributionKernel on the gpu. Contains most of the data contained by the original cpu class
*/

    double mu;
    double sigma;

    // Depends on sigma
    double squared_sigma_inv;

    /**
     * @brief Constructs a GaussianDistributionKernel on the GPU 
     * @param mu The mu algorithm parameter
     * @param sigma The theta algorithm parameter
     */
    __device__ GaussianDistributionKernel(double mu, double sigma)
        : mu(mu), sigma(sigma) {
        squared_sigma_inv = 1.0 / (sigma * sigma);
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of the gaussian distribution
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

        double3 diff_vector = (target_position - source_position).to_double3();
        const double x = CudaMath::calculate_2_norm(diff_vector);
        const double numerator = (x - mu) * (x - mu);

        const double exponent = -numerator * squared_sigma_inv;

        // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
        const double exp_val = exp(exponent);
        const double ret_val = number_free_elements * exp_val;

        return ret_val;
    }
    
};

class GaussianDistributionKernelHandleImpl : public GaussianDistributionKernelHandle {
/**
* Implementation of the handle for the cpu that controls the gpu object
*/

public:
     /**
     * @brief Constructs the GaussianDistributionKernelHandle Implementation
     * @param _dev_ptr The pointer to the GaussianDistributionKernel object on the GPU
     */
    GaussianDistributionKernelHandleImpl(GaussianDistributionKernel* _dev_ptr);

    /**
    * @brief Init function called by the constructor, has to be public in order to be allowed to use device lamdas in it, do not call from outside
    */
    void _init();

    /**
    * @brief Sets the mu kernel parameter used in calculation
    * @param mu The mu kernel parameter used in calculation
    */
    void set_mu(const double mu) override;

    /**
    * @brief Sets the sigma kernel parameter used in calculation
    * @param sigma The sigma kernel parameter used in calculation
    */
    void set_sigma(const double sigma) override;

    /**
    * @brief Returns the pointer to the GaussianDistributionKernel stored on the GPU
    * @return The pointer to the GaussianDistributionKernel stored on the GPU
    */
    [[nodiscard]] void* get_device_pointer() override;

private:
    GaussianDistributionKernel* device_ptr;

    double* handle_mu;
    double* handle_sigma;

    double* handle_squared_sigma_inv;
};
};