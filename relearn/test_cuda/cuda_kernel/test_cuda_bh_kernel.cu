/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_cuda_bh_kernel.cuh"

#include "../harness/adapter/random/RandomAdapter.h"

#include "../../source/gpu/algorithm/kernel/KernelGPU.cuh"
#include "RelearnGPUException.h"

static constexpr double default_k = 1.0;
static constexpr double default_theta = 1.0;

static constexpr double default_mu = 0.0;
static constexpr double default_sigma = 750.0;

static constexpr double default_cutoff = std::numeric_limits<double>::infinity();

static constexpr double default_b = 1.0;

__global__ void deploy_kernel(uint64_t source_index, gpu::Vector::CudaDouble3 source_position, uint64_t target_index, int number_elements,
    gpu::Vector::CudaDouble3 target_position, gpu::kernel::Kernel* kernel, double* result) {

    result[0] = kernel->calculate_attractiveness_to_connect(source_index, source_position, target_index, number_elements, target_position);
}

double precompute_gamma(double k, double theta, int number_elements, gpu::Vector::CudaDouble3 source_position, gpu::Vector::CudaDouble3 target_position) {
    const auto factor_1 = number_elements * (1.0 / (std::tgamma(k) * std::pow(theta, k)));
    const auto vec_x = (source_position - target_position).to_double3();
    const auto xx = vec_x.x * vec_x.x;
    const auto yy = vec_x.y * vec_x.y;
    const auto zz = vec_x.z * vec_x.z;

    const auto sum = xx + yy + zz;
    const auto x = sqrt(sum);

    const auto factor_2 = std::pow(x, k - 1);
    const auto factor_3 = std::exp(x * (-1.0 / theta));

    const auto precomp_result = factor_1 * factor_2 * factor_3;
    return precomp_result;
}

double precompute_gaussian(double mu, double sigma, int number_elements, gpu::Vector::CudaDouble3 source_position, gpu::Vector::CudaDouble3 target_position) {

    const auto position_diff = (target_position - source_position).to_double3();
    const auto xx = position_diff.x * position_diff.x;
    const auto yy = position_diff.y * position_diff.y;
    const auto zz = position_diff.z * position_diff.z;

    const auto sum = xx + yy + zz;
    const auto x = sqrt(sum);

    const auto numerator = (x - mu) * (x - mu);

    const auto exponent = -numerator * (1.0 / (sigma * sigma));

    const auto exp_val = std::exp(exponent);
    const auto precomp_result = number_elements * exp_val;

    return precomp_result;
}

double precompute_linear(double cutoff_point, int number_elements, gpu::Vector::CudaDouble3 source_position, gpu::Vector::CudaDouble3 target_position) {

    const auto cast_number_elements = static_cast<double>(number_elements);

    if (std::isinf(cutoff_point)) {
        return cast_number_elements;
    }

    const auto vec_x = (source_position - target_position).to_double3();
    const auto xx = vec_x.x * vec_x.x;
    const auto yy = vec_x.y * vec_x.y;
    const auto zz = vec_x.z * vec_x.z;

    const auto sum = xx + yy + zz;
    const auto x = sqrt(sum);

    if (x > cutoff_point) {
        return 0.0;
    }

    const auto factor = x / cutoff_point;

    return (1 - factor) * cast_number_elements;
}

double precompute_weibull(double k, double b, int number_elements, gpu::Vector::CudaDouble3 source_position, gpu::Vector::CudaDouble3 target_position) {
    const auto factor_1 = number_elements * b * k;

    const auto vec_x = (source_position - target_position).to_double3();
    const auto xx = vec_x.x * vec_x.x;
    const auto yy = vec_x.y * vec_x.y;
    const auto zz = vec_x.z * vec_x.z;

    const auto sum = xx + yy + zz;
    const auto x = sqrt(sum);

    const auto factor_2 = std::pow(x, k - 1);
    const auto exponent = -b * factor_2 * x;
    const auto factor_3 = std::exp(exponent);

    const auto result = factor_1 * factor_2 * factor_3;

    return result;
}

// These tests check whether the computed results on the GPU are correct for different scenarios

TEST_F(CudaKernelTest, cudaKernelTestGamma) {
    auto gamma = gpu::kernel::create_gamma(default_k, default_theta);
    auto gaussian = gpu::kernel::create_gaussian(default_mu, default_sigma);
    auto linear = gpu::kernel::create_linear(default_cutoff);
    auto weibull = gpu::kernel::create_weibull(default_k, default_b);

    auto kernel = gpu::kernel::create_kernel(gamma->get_device_pointer(), gaussian->get_device_pointer(), linear->get_device_pointer(), weibull->get_device_pointer());

    double* result = (double*)cuda_malloc(1 * sizeof(double));
    double result_cpu[1];
    kernel->set_kernel_type(gpu::kernel::KernelType::Gamma);

    // Default values
    uint64_t source_index = 0;
    gpu::Vector::CudaDouble3 source_position(1.0, 3.0, 7.0);
    uint64_t target_index = 1;
    int number_elements = 5;
    gpu::Vector::CudaDouble3 target_position(-2.0, -3.9, 0.0);

    // Basic case
    auto precomp_result = precompute_gamma(default_k, default_theta, number_elements, source_position, target_position);
    deploy_kernel<<<1, 1>>>(source_index, source_position, target_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_NEAR(result_cpu[0], precomp_result, eps);

    // For autopsy
    deploy_kernel<<<1, 1>>>(source_index, source_position, source_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_EQ(result_cpu[0], 0.0);

    // For setter
    gamma->set_k_theta(2.0, 3.0);
    precomp_result = precompute_gamma(2.0, 3.0, number_elements, source_position, target_position);
    deploy_kernel<<<1, 1>>>(source_index, source_position, target_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_NEAR(result_cpu[0], precomp_result, eps);
}

TEST_F(CudaKernelTest, cudaKernelTestGaussian) {
    auto gamma = gpu::kernel::create_gamma(default_k, default_theta);
    auto gaussian = gpu::kernel::create_gaussian(default_mu, default_sigma);
    auto linear = gpu::kernel::create_linear(default_cutoff);
    auto weibull = gpu::kernel::create_weibull(default_k, default_b);

    auto kernel = gpu::kernel::create_kernel(gamma->get_device_pointer(), gaussian->get_device_pointer(), linear->get_device_pointer(), weibull->get_device_pointer());

    double* result = (double*)cuda_malloc(1 * sizeof(double));
    double result_cpu[1];
    kernel->set_kernel_type(gpu::kernel::KernelType::Gaussian);

    // Default values
    uint64_t source_index = 0;
    gpu::Vector::CudaDouble3 source_position(1.0, 3.0, 7.0);
    uint64_t target_index = 1;
    int number_elements = 5;
    gpu::Vector::CudaDouble3 target_position(-2.0, -3.9, 0.0);

    // Basic case
    auto precomp_result = precompute_gaussian(default_mu, default_sigma, number_elements, source_position, target_position);
    deploy_kernel<<<1, 1>>>(source_index, source_position, target_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_NEAR(result_cpu[0], precomp_result, eps);

    // For autopsy
    deploy_kernel<<<1, 1>>>(source_index, source_position, source_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_EQ(result_cpu[0], 0.0);

    // For setter
    gaussian->set_mu(0.1);
    gaussian->set_sigma(850.0);
    precomp_result = precompute_gaussian(0.1, 850.0, number_elements, source_position, target_position);
    deploy_kernel<<<1, 1>>>(source_index, source_position, target_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_NEAR(result_cpu[0], precomp_result, eps);
}

TEST_F(CudaKernelTest, cudaKernelTestLinear) {
    auto gamma = gpu::kernel::create_gamma(default_k, default_theta);
    auto gaussian = gpu::kernel::create_gaussian(default_mu, default_sigma);
    auto linear = gpu::kernel::create_linear(default_cutoff);
    auto weibull = gpu::kernel::create_weibull(default_k, default_b);

    auto kernel = gpu::kernel::create_kernel(gamma->get_device_pointer(), gaussian->get_device_pointer(), linear->get_device_pointer(), weibull->get_device_pointer());

    double* result = (double*)cuda_malloc(1 * sizeof(double));
    double result_cpu[1];
    kernel->set_kernel_type(gpu::kernel::KernelType::Linear);

    // Default values
    uint64_t source_index = 0;
    gpu::Vector::CudaDouble3 source_position(1.0, 3.0, 7.0);
    uint64_t target_index = 1;
    int number_elements = 5;
    gpu::Vector::CudaDouble3 target_position(-2.0, -3.9, 0.0);

    // Basic case
    auto precomp_result = precompute_linear(default_cutoff, number_elements, source_position, target_position);
    deploy_kernel<<<1, 1>>>(source_index, source_position, target_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_NEAR(result_cpu[0], precomp_result, eps);

    // For autopsy
    deploy_kernel<<<1, 1>>>(source_index, source_position, source_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_EQ(result_cpu[0], 0.0);

    // Case 1
    linear->set_cutoff(5.0);
    precomp_result = precompute_linear(5.0, number_elements, source_position, target_position);
    deploy_kernel<<<1, 1>>>(source_index, source_position, target_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_NEAR(result_cpu[0], precomp_result, eps);

    // Case 2
    linear->set_cutoff(20.0);
    precomp_result = precompute_linear(20.0, number_elements, source_position, target_position);
    deploy_kernel<<<1, 1>>>(source_index, source_position, target_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_NEAR(result_cpu[0], precomp_result, eps);
}

TEST_F(CudaKernelTest, cudaKernelTestWeibull) {
    auto gamma = gpu::kernel::create_gamma(default_k, default_theta);
    auto gaussian = gpu::kernel::create_gaussian(default_mu, default_sigma);
    auto linear = gpu::kernel::create_linear(default_cutoff);
    auto weibull = gpu::kernel::create_weibull(default_k, default_b);

    auto kernel = gpu::kernel::create_kernel(gamma->get_device_pointer(), gaussian->get_device_pointer(), linear->get_device_pointer(), weibull->get_device_pointer());

    double* result = (double*)cuda_malloc(1 * sizeof(double));
    double result_cpu[1];
    kernel->set_kernel_type(gpu::kernel::KernelType::Weibull);

    // Default values
    uint64_t source_index = 0;
    gpu::Vector::CudaDouble3 source_position(1.0, 3.0, 7.0);
    uint64_t target_index = 1;
    int number_elements = 5;
    gpu::Vector::CudaDouble3 target_position(-2.0, -3.9, 0.0);

    // Basic case
    auto precomp_result = precompute_weibull(default_k, default_b, number_elements, source_position, target_position);
    deploy_kernel<<<1, 1>>>(source_index, source_position, target_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_NEAR(result_cpu[0], precomp_result, eps);

    // For autopsy
    deploy_kernel<<<1, 1>>>(source_index, source_position, source_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_EQ(result_cpu[0], 0.0);

    // For setter
    weibull->set_k(5.0);
    weibull->set_b(10.0);
    precomp_result = precompute_weibull(5.0, 10.0, number_elements, source_position, target_position);
    deploy_kernel<<<1, 1>>>(source_index, source_position, target_index, number_elements, target_position, (gpu::kernel::Kernel*)kernel->get_device_pointer(), result);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(result), (void*)&result_cpu[0], sizeof(double), 1);
    ASSERT_NEAR(result_cpu[0], precomp_result, eps);
}