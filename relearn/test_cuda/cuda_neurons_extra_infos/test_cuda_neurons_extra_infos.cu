/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_cuda_neurons_extra_infos.cuh"

#include "../harness/adapter/random/RandomAdapter.h"

#include "../../source/gpu/neurons/NeuronsExtraInfos.cuh"
#include "RelearnGPUException.h"

static constexpr RelearnGPUTypes::number_neurons_type num_neurons = 50;

__global__ void get_positions(gpu::neurons::NeuronsExtraInfos* extra_infos, double3* _return) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    _return[thread_id] = extra_infos->positions[thread_id];
}

__global__ void get_disable_flags(gpu::neurons::NeuronsExtraInfos* extra_infos, UpdateStatus* _return) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    _return[thread_id] = extra_infos->disable_flags[thread_id];
}

TEST_F(CudaNeuronsExtraInfosTest, cudaNeuronsExtraInfosTest) {
    auto neurons_extra_infos = gpu::neurons::create();

    neurons_extra_infos->init(num_neurons);

    // position check
    std::vector<gpu::Vec3d> positions;
    for (int i = 0; i < num_neurons; i++) {
        double rand = RandomAdapter::get_random_double(-5.0, 5.0, this->mt);
        positions.push_back(gpu::Vec3d(rand, rand, rand));
    }

    neurons_extra_infos->set_positions(positions);
    double3* gpu_pos = (double3*)cuda_malloc(sizeof(double3) * num_neurons);
    get_positions<<<1,num_neurons>>>((gpu::neurons::NeuronsExtraInfos*)neurons_extra_infos->get_device_pointer(), gpu_pos);
    double3 cpu_pos[num_neurons];
    cuda_memcpy_to_host(gpu_pos, &cpu_pos, sizeof(double3), num_neurons);

    for (int i = 0; i < num_neurons; i++) {
        ASSERT_EQ(cpu_pos[i].x, positions[i].x);
        ASSERT_EQ(cpu_pos[i].y, positions[i].y);
        ASSERT_EQ(cpu_pos[i].z, positions[i].z);
    }

    // disable flag check
    UpdateStatus* gpu_disable_flags = (UpdateStatus*)cuda_malloc(sizeof(UpdateStatus) * num_neurons);
    get_disable_flags<<<1,num_neurons>>>((gpu::neurons::NeuronsExtraInfos*)neurons_extra_infos->get_device_pointer(), gpu_disable_flags);
    UpdateStatus cpu_disable_flags[num_neurons];
    cuda_memcpy_to_host(gpu_disable_flags, &cpu_disable_flags, sizeof(UpdateStatus), num_neurons);

    for (int i = 0; i < num_neurons; i++) {
        ASSERT_EQ(cpu_disable_flags[i], UpdateStatus::Enabled);
    }

    RelearnGPUTypes::neuron_id_type neuron_to_disable = RandomAdapter::get_random_integer(0, (int)num_neurons - 1, this->mt);
    RelearnGPUTypes::neuron_id_type neuron_to_disable2 = RandomAdapter::get_random_integer(0, (int)num_neurons - 1, this->mt);
    neurons_extra_infos->disable_neurons({neuron_to_disable, neuron_to_disable2});
    neurons_extra_infos->enable_neurons({neuron_to_disable2});

    get_disable_flags<<<1,num_neurons>>>((gpu::neurons::NeuronsExtraInfos*)neurons_extra_infos->get_device_pointer(), gpu_disable_flags);
    cuda_memcpy_to_host(gpu_disable_flags, &cpu_disable_flags, sizeof(UpdateStatus), num_neurons);

    ASSERT_EQ(cpu_disable_flags[neuron_to_disable], UpdateStatus::Disabled);
    ASSERT_EQ(cpu_disable_flags[neuron_to_disable2], UpdateStatus::Enabled);
}