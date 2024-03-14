/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cuda_barnes_hut/test_cuda_barnes_hut.cuh"

#include "../harness/adapter/random/RandomAdapter.h"
#include "../harness/adapter/gpu/OctreeGPUAdapter.h"

#include "../../source/gpu/algorithm/BarnesHutKernel.cuh"
#include "RelearnGPUException.h"
#include "../../source/gpu/utils/RandomNew.cuh"

static constexpr int number_neurons_for_parallel = 64;
static constexpr int number_neurons_for_one_neuron = 1000;

static constexpr int subtract_for_smaller_gather_for_parallel = 30;
static constexpr int subtract_for_smaller_gather_for_one_neuron = 400;


__global__ void deploy_pick_target(gpu::algorithm::Octree* octree, gpu::algorithm::BarnesHutData* barnes_hut_data, gpu::random::RandomStateData* random_state_data,
    gpu::kernel::Kernel* kernel, uint64_t* picks_gpu) {

    unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= octree->number_neurons) {
        return;
    }

    uint64_t root_index = octree->number_neurons + octree->number_virtual_neurons - 1;
    extern __shared__ gpu::Vector::CudaDouble3 shared_data[];
    gpu::Vector::CudaDouble3* ptr_shared_target_positions = shared_data;
    int* ptr_shared_number_elements = (int*)&shared_data[number_neurons_for_parallel];
    

    picks_gpu[thread_id] = gpu::kernel::pick_target(thread_id, 
                                    octree->get_position_for_signal(SignalType::Excitatory, thread_id),
                                    root_index,
                                    barnes_hut_data, 
                                    SignalType::Excitatory, 
                                    octree,
                                    kernel,
                                    random_state_data,
                                    ptr_shared_number_elements,
                                    ptr_shared_target_positions);
}

__global__ void deploy_acceptance_criterion(double acceptance_criterion, gpu::algorithm::Octree* octree, int number_elements, gpu::kernel::AcceptanceStatus* result_gpu,
    uint64_t source_index, uint64_t target_index) {
    
    auto source_position = octree->position_excitatory_element[source_index];
    auto target_position = octree->position_excitatory_element[target_index];

    result_gpu[0] = gpu::kernel::test_acceptance_criterion(source_position, target_index, acceptance_criterion, octree, number_elements, target_position);
}

// Checks whether an autopsy happened in for the picked nodes on a parallel gather and pick with a full gather before pick
TEST_F(CudaBarnesHutTest, cudaPickTargetTestNoAutopsyFull) {

    auto [max_level, octree] = OctreeGPUAdapter::construct_random_octree(this->mt, number_neurons_for_parallel);

    auto barnes_hut_data = gpu::algorithm::create_barnes_hut_data(number_neurons_for_parallel * 2, number_neurons_for_parallel, 1);
    barnes_hut_data->update_kernel_allocation_sizes(number_neurons_for_parallel, max_level);

    auto kernel = OctreeGPUAdapter::get_kernel(); 

    uint64_t* picks_gpu = (uint64_t*)cuda_malloc(64 * sizeof(uint64_t));
    uint64_t picks_cpu[64];

    deploy_pick_target<<<1, 64, number_neurons_for_parallel * sizeof(gpu::Vector::CudaDouble3) + number_neurons_for_parallel * sizeof(int)>>>((gpu::algorithm::Octree*)octree->get_device_pointer(),
                                    (gpu::algorithm::BarnesHutData*)barnes_hut_data->get_device_pointer(),
                                    gpu::random::RandomHolder::get_instance().get_device_pointer(),
                                    (gpu::kernel::Kernel*)kernel->get_device_pointer(),
                                    picks_gpu);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(picks_gpu), (void*)&picks_cpu[0], sizeof(uint64_t), 64);

    for (int i = 0; i < 64; i++) {
        ASSERT_NE(i, picks_cpu[i]);
    }
}

// Checks whether an autopsy happened in for the picked nodes on a parallel gather and pick with a partial gather before pick
TEST_F(CudaBarnesHutTest, cudaPickTargetTestNoAutopsyPartial) {
    auto [max_level, octree] = OctreeGPUAdapter::construct_random_octree(this->mt, number_neurons_for_parallel);

    auto barnes_hut_data = gpu::algorithm::create_barnes_hut_data(number_neurons_for_parallel * 2, number_neurons_for_parallel - subtract_for_smaller_gather_for_parallel, 1);
    barnes_hut_data->update_kernel_allocation_sizes(number_neurons_for_parallel, max_level);

    auto kernel = OctreeGPUAdapter::get_kernel(); 

    uint64_t* picks_gpu = (uint64_t*)cuda_malloc(64 * sizeof(uint64_t));
    uint64_t picks_cpu[64];

    deploy_pick_target<<<1, 64, number_neurons_for_parallel * sizeof(gpu::Vector::CudaDouble3) + number_neurons_for_parallel * sizeof(int)>>>((gpu::algorithm::Octree*)octree->get_device_pointer(),
                                    (gpu::algorithm::BarnesHutData*)barnes_hut_data->get_device_pointer(),
                                    gpu::random::RandomHolder::get_instance().get_device_pointer(),
                                    (gpu::kernel::Kernel*)kernel->get_device_pointer(),
                                    picks_gpu);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(picks_gpu), (void*)&picks_cpu[0], sizeof(uint64_t), 64);

    for (int i = 0; i < 64; i++) {
        ASSERT_NE(i, picks_cpu[i]);
    }
}

// Picks a node on the GPU and checks whether it is part of the nodes which were gathered through the CPU Algorithm, using full gather before pick
TEST_F(CudaBarnesHutTest, cudaPickTargetTestAgainstCPUFull) {
    auto barnes_hut_data = gpu::algorithm::create_barnes_hut_data(number_neurons_for_one_neuron * 2, number_neurons_for_one_neuron, 1);

    auto [max_level, octree, gathered_device_indices] = OctreeGPUAdapter::construct_random_octree_and_gather(this->mt, number_neurons_for_one_neuron, 0, SignalType::Excitatory, barnes_hut_data->get_acceptance_criterion());

    barnes_hut_data->update_kernel_allocation_sizes(number_neurons_for_one_neuron, max_level);

    auto kernel = OctreeGPUAdapter::get_kernel(); 

    uint64_t* pick_gpu = (uint64_t*)cuda_malloc(1 * sizeof(uint64_t));
    uint64_t pick_cpu;

    deploy_pick_target<<<1, 1, number_neurons_for_parallel * sizeof(gpu::Vector::CudaDouble3) + number_neurons_for_parallel * sizeof(int)>>>((gpu::algorithm::Octree*)octree->get_device_pointer(),
                                    (gpu::algorithm::BarnesHutData*)barnes_hut_data->get_device_pointer(),
                                    gpu::random::RandomHolder::get_instance().get_device_pointer(),
                                    (gpu::kernel::Kernel*)kernel->get_device_pointer(),
                                    pick_gpu);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(pick_gpu), (void*)&pick_cpu, sizeof(uint64_t), 1);

    ASSERT_NE(std::find(gathered_device_indices.begin(), gathered_device_indices.end(), pick_cpu), gathered_device_indices.end());
}

// Picks a node on the GPU and checks whether it is part of the nodes which were gathered through the CPU Algorithm, using a partial gather before pick
TEST_F(CudaBarnesHutTest, cudaPickTargetTestAgainstCPUPartial) {
    auto barnes_hut_data = gpu::algorithm::create_barnes_hut_data(number_neurons_for_one_neuron * 2, number_neurons_for_one_neuron - subtract_for_smaller_gather_for_one_neuron, 1);

    auto [max_level, octree, gathered_device_indices] = OctreeGPUAdapter::construct_random_octree_and_gather(this->mt, number_neurons_for_one_neuron, 0, SignalType::Excitatory, barnes_hut_data->get_acceptance_criterion());

    barnes_hut_data->update_kernel_allocation_sizes(number_neurons_for_one_neuron, max_level);

    auto kernel = OctreeGPUAdapter::get_kernel(); 

    uint64_t* pick_gpu = (uint64_t*)cuda_malloc(1 * sizeof(uint64_t));
    uint64_t pick_cpu;

    deploy_pick_target<<<1, 1, number_neurons_for_parallel * sizeof(gpu::Vector::CudaDouble3) + number_neurons_for_parallel * sizeof(int)>>>((gpu::algorithm::Octree*)octree->get_device_pointer(),
                                    (gpu::algorithm::BarnesHutData*)barnes_hut_data->get_device_pointer(),
                                    gpu::random::RandomHolder::get_instance().get_device_pointer(),
                                    (gpu::kernel::Kernel*)kernel->get_device_pointer(),
                                    pick_gpu);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    cuda_memcpy_to_host((void*)(pick_gpu), (void*)&pick_cpu, sizeof(uint64_t), 1);

    ASSERT_NE(std::find(gathered_device_indices.begin(), gathered_device_indices.end(), pick_cpu), gathered_device_indices.end());
}

TEST_F(CudaBarnesHutTest, cudaTestAcceptanceCriterion) {
    auto [max_level, octree] = OctreeGPUAdapter::construct_random_octree(this->mt, number_neurons_for_parallel);

    gpu::kernel::AcceptanceStatus* result_gpu = (gpu::kernel::AcceptanceStatus*)cuda_malloc(1 * sizeof(gpu::kernel::AcceptanceStatus));
    gpu::kernel::AcceptanceStatus result_cpu;

    uint64_t target_index = number_neurons_for_parallel;
    uint64_t source_index = 0;
    const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, 0.3, this->mt);
    int number_elements = 5;

    // Case test virtual node
    deploy_acceptance_criterion<<<1, 1>>>(acceptance_criterion, (gpu::algorithm::Octree*)octree->get_device_pointer(), number_elements, result_gpu,
        source_index, target_index);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host((void*)(result_gpu), (void*)&result_cpu, sizeof(gpu::kernel::AcceptanceStatus), 1);

    gpu::Vec3d source_pos_vec = octree->get_node_position(source_index, SignalType::Excitatory);
    gpu::Vec3d target_pos_vec = octree->get_node_position(target_index, SignalType::Excitatory);
    gpu::Vector::CudaDouble3 source_pos(source_pos_vec.x, source_pos_vec.y, source_pos_vec.z);
    gpu::Vector::CudaDouble3 target_pos(target_pos_vec.x, target_pos_vec.y, target_pos_vec.z);

    auto [min_vec, max_vec] = octree->get_bounding_box(target_index);
    gpu::Vector::CudaDouble3 min(min_vec.x, min_vec.y, min_vec.z);
    gpu::Vector::CudaDouble3 max(max_vec.x, max_vec.y, max_vec.z);
    gpu::Vector::CudaDouble3 diff_vector = max - min;
    auto maximum_cell_dimension = diff_vector.max();

    const auto distance = CudaMath::calculate_2_norm((target_pos - source_pos).to_double3());
    const auto quotient = maximum_cell_dimension / distance;

    if (acceptance_criterion > quotient) {
        ASSERT_EQ(result_cpu, gpu::kernel::AcceptanceStatus::Accept);
    } else {
        ASSERT_EQ(result_cpu, gpu::kernel::AcceptanceStatus::Expand);
    }

    // case test neuron target
    deploy_acceptance_criterion<<<1, 1>>>(acceptance_criterion, (gpu::algorithm::Octree*)octree->get_device_pointer(), number_elements, result_gpu,
        source_index, 1);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host((void*)(result_gpu), (void*)&result_cpu, sizeof(gpu::kernel::AcceptanceStatus), 1);
    ASSERT_EQ(result_cpu, gpu::kernel::AcceptanceStatus::Accept);

    // case test autopsy
    deploy_acceptance_criterion<<<1, 1>>>(acceptance_criterion, (gpu::algorithm::Octree*)octree->get_device_pointer(), number_elements, result_gpu,
        source_index, source_index);
    cudaDeviceSynchronize();
    gpu_check_last_error();
    cuda_memcpy_to_host((void*)(result_gpu), (void*)&result_cpu, sizeof(gpu::kernel::AcceptanceStatus), 1);
    ASSERT_EQ(result_cpu, gpu::kernel::AcceptanceStatus::Discard);
}