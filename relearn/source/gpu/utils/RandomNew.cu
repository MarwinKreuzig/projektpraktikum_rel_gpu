/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RandomNew.cuh"

namespace gpu::random {

__global__ void init_state(RandomKeyHolder kernel, RandomStateData* random_data) {
    uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= random_data->random_states[kernel].size) {
        return;
    }

    random_data->init_state(kernel, thread_id);
}

RandomHolder::RandomHolder() {
    init();
}

void RandomHolder::init() {
    device_ptr = init_class_on_device<RandomStateData>();

    cudaDeviceSynchronize();
    gpu_check_last_error();

    for (int i = 0; i < RandomKeyHolder::COUNT; i++) {
        void* random_states_ptr = execute_and_copy<void*>([=] __device__(RandomStateData* random_state_data, int index) { return (void*)(&(random_state_data->random_states[index])); }, device_ptr, i);
        handle_random_states[i] = gpu::Vector::CudaArrayDeviceHandle<random_state_type>(random_states_ptr);
    }

    cudaDeviceSynchronize();
    gpu_check_last_error();
}

void RandomHolder::init_allocation(RandomKeyHolder kernel, size_t block_size, size_t grid_size) {
    handle_random_states[kernel].resize(block_size * grid_size);

    auto [_grid_size, _block_size] = get_number_blocks_and_threads(init_state, block_size * grid_size);

    init_state<<<_grid_size,_block_size>>>(kernel, device_ptr);
    cudaDeviceSynchronize();
    gpu_check_last_error();
}
};