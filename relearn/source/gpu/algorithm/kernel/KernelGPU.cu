/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "KernelGPU.cuh"

namespace gpu::kernel {

KernelHandleImpl::KernelHandleImpl(Kernel* _dev_ptr) 
    : device_ptr(_dev_ptr) {
    _init();
}

void KernelHandleImpl::_init() {
    handle_currently_used_kernel = execute_and_copy<KernelType*>([=] __device__(Kernel* kernel) { return &kernel->currently_used_kernel; }, device_ptr);
}

void KernelHandleImpl::set_kernel_type(const KernelType kernel_type) {
    cuda_memcpy_to_device((void*)handle_currently_used_kernel, (void*)&kernel_type, sizeof(KernelType), 1);
}

[[nodiscard]] void* KernelHandleImpl::get_device_pointer() {
    return device_ptr;
}

std::shared_ptr<KernelHandle> create_kernel(void* gamma_dev_ptr, void* gaussian_dev_ptr, void* linear_dev_ptr, void* weibull_dev_ptr) {
    Kernel* kernel_dev_ptr = init_class_on_device<Kernel>((GammaDistributionKernel*)gamma_dev_ptr, (GaussianDistributionKernel*)gaussian_dev_ptr, (LinearDistributionKernel*)linear_dev_ptr, (WeibullDistributionKernel*)weibull_dev_ptr);

    cudaDeviceSynchronize();
    gpu_check_last_error();

    auto a = std::make_shared<KernelHandleImpl>(kernel_dev_ptr);
    return std::move(a);
}
};