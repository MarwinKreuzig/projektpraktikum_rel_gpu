/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_cuda_synaptic_elements.cuh"

#include "../harness/adapter/random/RandomAdapter.h"

#include "../../source/gpu/neurons/models/SynapticElements.cuh"
#include "RelearnGPUException.h"

static constexpr RelearnGPUTypes::number_neurons_type num_neurons = 50;

__global__ void do_get_free_elements(gpu::models::SynapticElements* syn, unsigned int* _return) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    _return[thread_id] = syn->get_free_elements(thread_id);
}

TEST_F(CudaSynapticElementsTest, cudaSynapticElementsTest) {
    auto axons = gpu::models::create_synaptic_elements(ElementType::Axon);

    std::vector<double> grown_elements;
    for (int i = 0; i < num_neurons; i++) {
        grown_elements.push_back(RandomAdapter::get_random_double(1.0, 5.0, this->mt));
    }

    axons->init(num_neurons, grown_elements);
    unsigned int* free_elements_gpu = (unsigned int*)cuda_malloc(sizeof(unsigned int) * num_neurons);
    do_get_free_elements<<<1, num_neurons>>>((gpu::models::SynapticElements*)axons->get_device_pointer(), free_elements_gpu);
    unsigned int free_elements_cpu[num_neurons];
    cuda_memcpy_to_host(free_elements_gpu, &free_elements_cpu, sizeof(unsigned int), num_neurons);

    for (int i = 0; i < num_neurons; i++) {
        ASSERT_EQ(free_elements_cpu[i], static_cast<unsigned int>(grown_elements[i]));
    }

    // Post update test
    int neuron_to_connect = RandomAdapter::get_random_integer(0, (int)num_neurons - 1, this->mt);

    axons->update_connected_elements(neuron_to_connect, 1);
    axons->update_grown_elements(neuron_to_connect, -1);

    do_get_free_elements<<<1, num_neurons>>>((gpu::models::SynapticElements*)axons->get_device_pointer(), free_elements_gpu);
    cuda_memcpy_to_host(free_elements_gpu, &free_elements_cpu, sizeof(unsigned int), num_neurons);
    grown_elements[neuron_to_connect] = grown_elements[neuron_to_connect] - 2;
    for (int i = 0; i < num_neurons; i++) {
        ASSERT_EQ(free_elements_cpu[i], static_cast<unsigned int>(grown_elements[i]));
    }

    // Create Neurons test
    std::vector<double> new_grown_elements;
    for (int i = 0; i < num_neurons + 50; i++) {
        new_grown_elements.push_back(RandomAdapter::get_random_double(1.0, 5.0, this->mt));
    }

    axons->create_neurons(num_neurons + 50, new_grown_elements);
    unsigned int* new_free_elements_gpu = (unsigned int*)cuda_malloc(sizeof(unsigned int) * (num_neurons + 50));
    do_get_free_elements<<<1, num_neurons + 50>>>((gpu::models::SynapticElements*)axons->get_device_pointer(), new_free_elements_gpu);
    unsigned int new_free_elements_cpu[num_neurons + 50];
    cuda_memcpy_to_host(new_free_elements_gpu, &new_free_elements_cpu, sizeof(unsigned int), num_neurons + 50);
    new_grown_elements[neuron_to_connect] = new_grown_elements[neuron_to_connect] - 1;

    for (int i = 0; i < num_neurons + 50; i++) {
        ASSERT_EQ(new_free_elements_cpu[i], static_cast<unsigned int>(new_grown_elements[i])) << " i:" << i;
    }
}