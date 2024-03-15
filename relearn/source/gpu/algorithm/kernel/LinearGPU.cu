/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "LinearGPU.cuh"

namespace gpu::kernel {

LinearDistributionKernelHandleImpl::LinearDistributionKernelHandleImpl(LinearDistributionKernel* _dev_ptr)
    : device_ptr(_dev_ptr) {
    _init();
}

void LinearDistributionKernelHandleImpl::_init() {
    handle_cutoff_point = execute_and_copy<double*>([=] __device__(LinearDistributionKernel * kernel) { return &kernel->cutoff_point; }, device_ptr);
}

void LinearDistributionKernelHandleImpl::set_cutoff(const double cutoff_point) {
    cuda_memcpy_to_device((void*)handle_cutoff_point, (void*)&cutoff_point, sizeof(double), 1);
}

[[nodiscard]] void* LinearDistributionKernelHandleImpl::get_device_pointer() {
    return device_ptr;
}

std::shared_ptr<LinearDistributionKernelHandle> create_linear(double cutoff_point) {
    LinearDistributionKernel* kernel_dev_ptr = init_class_on_device<LinearDistributionKernel>(cutoff_point);

    cudaDeviceSynchronize();
    gpu_check_last_error();

    auto a = std::make_shared<LinearDistributionKernelHandleImpl>(kernel_dev_ptr);
    return std::move(a);
}
};