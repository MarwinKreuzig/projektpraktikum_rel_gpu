/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "GaussianGPU.cuh"

namespace gpu::kernel {

GaussianDistributionKernelHandleImpl::GaussianDistributionKernelHandleImpl(GaussianDistributionKernel* _dev_ptr)
    : device_ptr(_dev_ptr) {
    _init();
}

void GaussianDistributionKernelHandleImpl::_init() {
    handle_mu = execute_and_copy<double*>([=] __device__(GaussianDistributionKernel * kernel) { return &kernel->mu; }, device_ptr);
    handle_sigma = execute_and_copy<double*>([=] __device__(GaussianDistributionKernel * kernel) { return &kernel->sigma; }, device_ptr);
    handle_squared_sigma_inv = execute_and_copy<double*>([=] __device__(GaussianDistributionKernel * kernel) { return &kernel->squared_sigma_inv; }, device_ptr);
}

void GaussianDistributionKernelHandleImpl::set_mu(const double mu) {
    cuda_memcpy_to_device((void*)handle_mu, (void*)&mu, sizeof(double), 1);
}

void GaussianDistributionKernelHandleImpl::set_sigma(const double sigma) {
    cuda_memcpy_to_device((void*)handle_sigma, (void*)&sigma, sizeof(double), 1);
    double squared_sigma_inv = 1.0 / (sigma * sigma);
    cuda_memcpy_to_device((void*)handle_squared_sigma_inv, (void*)&squared_sigma_inv, sizeof(double), 1);
}

[[nodiscard]] void* GaussianDistributionKernelHandleImpl::get_device_pointer() {
    return device_ptr;
}

std::shared_ptr<GaussianDistributionKernelHandle> create_gaussian(double mu, double sigma) {
    GaussianDistributionKernel* kernel_dev_ptr = init_class_on_device<GaussianDistributionKernel>(mu, sigma);

    cudaDeviceSynchronize();
    gpu_check_last_error();

    auto a = std::make_shared<GaussianDistributionKernelHandleImpl>(kernel_dev_ptr);
    return std::move(a);
}
};