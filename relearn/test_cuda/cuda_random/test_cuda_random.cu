/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_cuda_random.cuh"

#include "../harness/adapter/random/RandomAdapter.h"
#include "../../source/gpu/utils/RandomNew.cuh"

#include "RelearnGPUException.h"

static constexpr int grid_size = 2;
static constexpr int block_size = 2;

__global__ void do_percentage(gpu::random::RandomStateData* random_state_data, double* _result) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    _result[thread_id] = random_state_data->get_percentage(gpu::random::RandomKeyHolder::BARNES_HUT, thread_id);
}

// Checks wheter the percentage given back is actually a percentage
TEST_F(CudaRandomTest, cudaRandomTestPercentage) {
    gpu::random::RandomHolder::get_instance().init_allocation(gpu::random::RandomKeyHolder::BARNES_HUT, block_size, grid_size);
    double* gpu_result = (double*)cuda_malloc(sizeof(double) * block_size * grid_size);
    double cpu_result[block_size * grid_size];

    do_percentage<<<2, 2>>>(gpu::random::RandomHolder::get_instance().get_device_pointer(), gpu_result);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host(gpu_result, cpu_result, sizeof(double), block_size * grid_size);

    for (int i = 0; i < block_size * grid_size; i++) {
        ASSERT_LE(cpu_result[i], 1.0);
        ASSERT_GE(cpu_result[i], 0.0);
    }
}
