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

#include "../../utils/Macros.h"
#include "../../utils/GpuTypes.h"

#include <memory>
#include <vector>

// Macros to define methods as virtual/implemented somewhere else when compiled with cuda
#ifndef CUDA_DEFINITION
#define CUDA_DEFINITION ;
#define CUDA_PTR_DEFINITION CUDA_DEFINITION
#endif

namespace gpu::kernel {

    /**
     * KernelType equivalent enum for the GPU
     */
    enum class KernelType : uint16_t {
        Gaussian = 0,
        Linear = 1,
        Gamma = 2,
        Weibull = 3
    };

    class KernelHandle {
    public:
        /**
        * @brief Sets the kernel type to be used during the Barnes Hut algorithm
        * @param kernel_type The kernel type to be used during the Barnes Hut algorithm
        */
        virtual void set_kernel_type(const KernelType kernel_type) = 0;

        /**
        * @brief Returns the pointer to the Kernel stored on the GPU
        * @return The pointer to the Kernel stored on the GPU
        */
        [[nodiscard]] virtual void* get_device_pointer() = 0;
    };

    /**
    * @brief Creates a new KernelHandle with the given pointers to the different kernel types on the GPU
    * @param gamma_dev_ptr Pointer to the gamma kernel on the GPU
    * @param gaussian_dev_ptr Pointer to the gaussian kernel on the GPU
    * @param linear_dev_ptr Pointer to the linear kernel on the GPU
    * @param weibull_dev_ptr Pointer to the weibull kernel on the GPU
    * @return A shared pointer to the created KernelHandle
    */
    std::shared_ptr<KernelHandle> create_kernel(void* gamma_dev_ptr, void* gaussian_dev_ptr, void* linear_dev_ptr, void* weibull_dev_ptr) CUDA_PTR_DEFINITION

    class GaussianDistributionKernelHandle {
    public:
        /**
        * @brief Sets the mu kernel parameter used in calculation
        * @param mu The mu kernel parameter used in calculation
        */
        virtual void set_mu(const double mu) = 0;

        /**
        * @brief Sets the sigma kernel parameter used in calculation
        * @param sigma The sigma kernel parameter used in calculation
        */
        virtual void set_sigma(const double sigma) = 0;

        /**
        * @brief Returns the pointer to the GaussianDistributionKernel stored on the GPU
        * @return The pointer to the GaussianDistributionKernel stored on the GPU
        */
        [[nodiscard]] virtual void* get_device_pointer() = 0;
    };

    /**
    * @brief Creates a new GaussianDistributionKernelHandle with the given kernel parameters
    * @param mu The mu kernel parameter used in calculation
    * @param sigma The sigma kernel parameter used in calculation
    * @return A shared pointer to the created GaussianDistributionKernelHandle
    */
    std::shared_ptr<GaussianDistributionKernelHandle> create_gaussian(double mu, double sigma) CUDA_PTR_DEFINITION

    class GammaDistributionKernelHandle {
    public:
        /**
        * @brief Sets the k and theta kernel parameters used in calculation
        * @param k The k kernel parameter used in calculation
        * @param theta The theta kernel parameter used in calculation
        */
        virtual void set_k_theta(const double k, const double theta) = 0;

        /**
        * @brief Returns the pointer to the GammaDistributionKernel stored on the GPU
        * @return The pointer to the GammaDistributionKernel stored on the GPU
        */
        [[nodiscard]] virtual void* get_device_pointer() = 0;
    };

    /**
    * @brief Creates a new GammaDistributionKernelHandle with the given kernel parameters
    * @param k The k kernel parameter used in calculation
    * @param theta The theta kernel parameter used in calculation
    * @return A shared pointer to the created GammaDistributionKernelHandle
    */
    std::shared_ptr<GammaDistributionKernelHandle> create_gamma(double k, double theta) CUDA_PTR_DEFINITION

    class LinearDistributionKernelHandle {
    public:
        /**
        * @brief Sets the cutoff_point kernel parameter used in calculation
        * @param cutoff_point The cutoff_point kernel parameter used in calculation
        */
        virtual void set_cutoff(const double cutoff_point) = 0;

        /**
        * @brief Returns the pointer to the LinearDistributionKernel stored on the GPU
        * @return The pointer to the LinearDistributionKernel stored on the GPU
        */
        [[nodiscard]] virtual void* get_device_pointer() = 0;
    };

    /**
    * @brief Creates a new LinearDistributionKernelHandle with the given kernel parameters
    * @param cutoff_point The cutoff_point kernel parameter used in calculation
    * @return A shared pointer to the created LinearDistributionKernelHandle
    */
    std::shared_ptr<LinearDistributionKernelHandle> create_linear(double cutoff_point) CUDA_PTR_DEFINITION

    class WeibullDistributionKernelHandle {
    public:
        /**
        * @brief Sets the k kernel parameter used in calculation
        * @param k The k kernel parameter used in calculation
        */
        virtual void set_k(const double k) = 0;

        /**
        * @brief Sets the b kernel parameter used in calculation
        * @param b The b kernel parameter used in calculation
        */
        virtual void set_b(const double b) = 0;

        /**
        * @brief Returns the pointer to the WeibullDistributionKernel stored on the GPU
        * @return The pointer to the WeibullDistributionKernel stored on the GPU
        */
        [[nodiscard]] virtual void* get_device_pointer() = 0;
    };

    /**
    * @brief Creates a new WeibullDistributionKernelHandle with the given kernel parameters
    * @param k The k kernel parameter used in calculation
    * @param b The b kernel parameter used in calculation
    * @return A shared pointer to the created WeibullDistributionKernelHandle
    */
    std::shared_ptr<WeibullDistributionKernelHandle> create_weibull(double k, double b) CUDA_PTR_DEFINITION
};