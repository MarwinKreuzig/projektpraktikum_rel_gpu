/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "WeibullGPU.cuh"

namespace gpu::kernel {

    WeibullDistributionKernelHandleImpl::WeibullDistributionKernelHandleImpl(WeibullDistributionKernel* _dev_ptr) 
        : device_ptr(_dev_ptr) {
        _init();
    }

    void WeibullDistributionKernelHandleImpl::_init() {
        handle_k = execute_and_copy<double*>([=] __device__(WeibullDistributionKernel* kernel) { return &kernel->k; }, device_ptr);
        handle_b = execute_and_copy<double*>([=] __device__(WeibullDistributionKernel* kernel) { return &kernel->b; }, device_ptr);
    }

    void WeibullDistributionKernelHandleImpl::set_k(const double k) {
        cuda_memcpy_to_device((void*)handle_k, (void*)&k, sizeof(double), 1);
    }

    void WeibullDistributionKernelHandleImpl::set_b(const double b) {
        cuda_memcpy_to_device((void*)handle_b, (void*)&b, sizeof(double), 1);
    }

    [[nodiscard]] void* WeibullDistributionKernelHandleImpl::get_device_pointer() {
        return device_ptr;
    }

    std::shared_ptr<WeibullDistributionKernelHandle> create_weibull(double k, double b) {
        WeibullDistributionKernel* kernel_dev_ptr = init_class_on_device<WeibullDistributionKernel>(k, b);

        cudaDeviceSynchronize();
        gpu_check_last_error();

        auto a = std::make_shared<WeibullDistributionKernelHandleImpl>(kernel_dev_ptr);
        return std::move(a);
    }
};