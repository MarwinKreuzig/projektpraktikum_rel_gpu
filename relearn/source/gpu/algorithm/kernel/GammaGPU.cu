/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "GammaGPU.cuh"

namespace gpu::kernel {

    GammaDistributionKernelHandleImpl::GammaDistributionKernelHandleImpl(GammaDistributionKernel* _dev_ptr) 
        : device_ptr(_dev_ptr) {
        _init();
    }

    void GammaDistributionKernelHandleImpl::_init() {
        handle_k = execute_and_copy<double*>([=] __device__(GammaDistributionKernel* kernel) { return &kernel->k; }, device_ptr);
        handle_theta = execute_and_copy<double*>([=] __device__(GammaDistributionKernel* kernel) { return &kernel->theta; }, device_ptr);
        handle_gamma_divisor_inv = execute_and_copy<double*>([=] __device__(GammaDistributionKernel* kernel) { return &kernel->gamma_divisor_inv; }, device_ptr);
        handle_theta_divisor = execute_and_copy<double*>([=] __device__(GammaDistributionKernel* kernel) { return &kernel->theta_divisor; }, device_ptr);
    }

    void GammaDistributionKernelHandleImpl::set_k_theta(const double k, const double theta) {
        cuda_memcpy_to_device((void*)handle_k, (void*)&k, sizeof(double), 1);
        cuda_memcpy_to_device((void*)handle_theta, (void*)&theta, sizeof(double), 1);
        double gamma_divisor_inv = 1.0 / (std::tgamma(k) * std::pow(theta, k));
        double theta_divisor = -1.0 / theta;
        cuda_memcpy_to_device((void*)handle_gamma_divisor_inv, (void*)&gamma_divisor_inv, sizeof(double), 1);
        cuda_memcpy_to_device((void*)handle_theta_divisor, (void*)&theta_divisor, sizeof(double), 1);
    }

    [[nodiscard]] void* GammaDistributionKernelHandleImpl::get_device_pointer() {
        return device_ptr;
    }

    std::shared_ptr<GammaDistributionKernelHandle> create_gamma(double k, double theta) {
        GammaDistributionKernel* kernel_dev_ptr = init_class_on_device<GammaDistributionKernel>(k, theta);

        cudaDeviceSynchronize();
        gpu_check_last_error();

        auto a = std::make_shared<GammaDistributionKernelHandleImpl>(kernel_dev_ptr);
        return std::move(a);
    }
};