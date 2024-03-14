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
#include "../../structure/Octree.cuh"
#include "GaussianGPU.cuh"
#include "GammaGPU.cuh"
#include "LinearGPU.cuh"
#include "WeibullGPU.cuh"

#include <iostream>
#include <cuda.h>

namespace gpu::kernel {

struct Kernel {
/**
* Struct representing Kernel on the gpu. Contains most of the data contained by the original cpu class
*/

    KernelType currently_used_kernel{ KernelType::Gaussian };
    GammaDistributionKernel* gamma;
    GaussianDistributionKernel* gaussian;
    LinearDistributionKernel* linear;
    WeibullDistributionKernel* weibull;
    
    /**
     * @brief Constructs the Kernel on the GPU 
     * @param gamma The pointer to the gamma kernel on the GPU
     * @param gaussian The pointer to the gaussian kernel on the GPU
     * @param linear The pointer to the linear kernel on the GPU
     * @param weibull TThe pointer to the weibull kernel on the GPU
     */
    __device__ Kernel(GammaDistributionKernel* gamma, GaussianDistributionKernel* gaussian, LinearDistributionKernel* linear, WeibullDistributionKernel* weibull)
        : gamma(gamma), gaussian(gaussian), linear(linear), weibull(weibull) {}

    /**
     * @brief Calculates the attractiveness to connect on the basis of the set kernel type.
     *      Performs all necessary checks and passes the values to the actual kernel.
     * @param source_index The source neuron index
     * @param source_position The source position
     * @param target_index The target index
     * @param number_elements The number free elements of the relevant type in the target
     * @param target_position The position of the target node
     * @exception Fails if the currently_used_kernel is unknown
     * @return The calculated attractiveness, might be 0.0 to avoid autapses
     */
    __device__ double calculate_attractiveness_to_connect(const RelearnGPUTypes::neuron_index_type& source_index, const gpu::Vector::CudaDouble3& source_position,
        const RelearnGPUTypes::neuron_index_type& target_index, const RelearnGPUTypes::number_elements_type& number_elements, const gpu::Vector::CudaDouble3& target_position) {
        // A neuron must not form an autapse, i.e., a synapse to itself
        if (source_index == target_index) {
            return 0.0;
        }

        switch (currently_used_kernel) {
        case KernelType::Gamma:
            return gamma->calculate_attractiveness_to_connect(source_position, target_position, number_elements);
        case KernelType::Gaussian:
            return gaussian->calculate_attractiveness_to_connect(source_position, target_position, number_elements);
        case KernelType::Linear:
            return linear->calculate_attractiveness_to_connect(source_position, target_position, number_elements);
        case KernelType::Weibull:
            return weibull->calculate_attractiveness_to_connect(source_position, target_position, number_elements);
        }

        RelearnGPUException::fail_device("gpu::kernel::Kernel::calculate_attractiveness_to_connect: {} is an unknown kernel type!", currently_used_kernel);

        return 0.0;
    }
};

class KernelHandleImpl : public KernelHandle {
/**
* Implementation of the handle for the cpu that controls the gpu object
*/

public:
     /**
     * @brief Constructs the KernelHandle Implementation
     * @param _dev_ptr The pointer to the Kernel object on the GPU
     */
    KernelHandleImpl(Kernel* _dev_ptr);

    /**
    * @brief Init function called by the constructor, has to be public in order to be allowed to use device lamdas in it, do not call from outside
    */
    void _init();

    /**
    * @brief Sets the kernel type to be used during the Barnes Hut algorithm
    * @param kernel_type The kernel type to be used during the Barnes Hut algorithm
    */
    void set_kernel_type(const KernelType kernel_type) override;

    /**
    * @brief Returns the pointer to the Kernel stored on the GPU
    * @return The pointer to the Kernel stored on the GPU
    */
    [[nodiscard]] void* get_device_pointer() override;

private:
    Kernel* device_ptr;

    KernelType* handle_currently_used_kernel;
};
};