#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHutKernel.h"
#include "BarnesHutData.cuh"
#include "../structure/Octree.cuh"
#include "../neurons/NeuronsExtraInfos.cuh"
#include "../neurons/models/SynapticElements.cuh"
#include "../utils/RandomNew.cuh"
#include "kernel/KernelGPU.cuh"

namespace gpu::kernel {

/**
 * @brief Does the Barnes Hut Algorithm on the GPU and creates new synapses
 * @param barnes_hut_data The pointer to the BarnesHutData on the GPU
 * @param octree The pointer to the Octree on the GPU
 * @param synaptic_elements The pointer to the SynapticElements on the GPU
 * @param neurons_extra_infos The pointer to the NeuronsExtraInfos on the GPU
 * @param kernel The pointer to the Kernel on the GPU
 * @param random_state_data The pointer to the RandomStateData on the GPU
 */
__global__ void update_connectivity_kernel(gpu::algorithm::BarnesHutData* barnes_hut_data,
    gpu::algorithm::Octree* octree,
    gpu::models::SynapticElements* synaptic_elements,
    gpu::neurons::NeuronsExtraInfos* neurons_extra_infos,
    gpu::kernel::Kernel* kernel,
    gpu::random::RandomStateData* random_state_data);

/**
 * @brief Tests the acceptance criterion for a given node to determine wether the target node should be expanded, discarded or accepted
 * @param source_position The position of the source neuron
 * @param target_index The index of the target node
 * @param acceptance_criterion The acceptance criterion
 * @param octree The pointer to the Octree on the GPU
 * @param number_elements The number of elements of the desired type of the target
 * @param target_pos The position of the desired signal type of the target node
 * @return The acceptance status defining wether the target node should be expanded, discarded or accepted
 */
__device__ AcceptanceStatus test_acceptance_criterion(const gpu::Vector::CudaDouble3& source_position, const RelearnGPUTypes::neuron_index_type& target_index, const double acceptance_criterion, gpu::algorithm::Octree* octree, const RelearnGPUTypes::number_elements_type& number_elements,
    const gpu::Vector::CudaDouble3& target_pos);

/**
 * @brief When the normal attractiveness to connect kernel return 0 for everything, this calcualtes the attractiveness in a different way
 * @param target_index The index to the target node
 * @param source_index The index of the source neuron
 * @param source_position The position of the source neuron
 * @param number_elements The number of free elements in the target
 * @param target_position The position of the target
 * @return The probability to connect to the target neuron
 */
__device__ inline double zero_prob_sum_case_kernel(const RelearnGPUTypes::neuron_index_type& target_index, const RelearnGPUTypes::neuron_index_type& source_index, const gpu::Vector::CudaDouble3& source_position,
    const RelearnGPUTypes::number_elements_type& number_elements, const gpu::Vector::CudaDouble3& target_position);

/**
 * @brief Randomly picks a target from the nodes gathered given the probability distribution
 * @param random_state_data The pointer to the RandomStateData on the GPU
 * @param prob_sum The sum of all probabilities in the probability interval
 * @param barnes_hut_data The pointer to the BarnesHutData on the GPU
 * @param num_nodes_gathered The number of nodes gathered during prefix traversal
 * @param thread_id The id of the calling thread
 * @param number_threads The number of threads used to call this kernel
 * @return The index of the randomly picked target neuron from the gathered nodes
 */
__device__ inline RelearnGPUTypes::neuron_index_type pick_target_randomly_from_dist(gpu::random::RandomStateData* random_state_data, const double& prob_sum, gpu::algorithm::BarnesHutData* barnes_hut_data,
    const uint64_t& num_nodes_gathered, const int& thread_id, const int& number_threads);

/**
 * @brief Picks a target to try to generate a synpase with for a given source neuron
 * @param octree The pointer to the Octree on the GPU
 * @param barnes_hut_data The pointer to the BarnesHutData on the GPU
 * @param thread_id The id of the calling thread
 * @param root_index The root which is used for prefix traversal
 * @param prob_sum The sum of all probabilities to connect to the gathered nodes
 * @param shared_number_elements The pointer to the shared memory where threads save the number of vacant elements of their target
 * @param shared_target_positions The pointer to the shared memory where threads save the positions of their target
 * @param signal_type_needed The signal type which the source neuron needs
 * @param source_position The position of the source neuron
 * @param use_kernel Determines wether to use the kernel or the zero_prob_sum_case_kernel function
 * @param source_index The index of the source neuron
 * @param number_threads The number of threads executing the current kernel
 * @param random_state_data The pointer to the RandomStateData on the GPU
 * @param kernel The pointer to the Kernel on the GPU
 * @return The index of the picked target neuron, returns number_neurons + number_virtual_neurons when no node was picked
 */
__device__ inline RelearnGPUTypes::neuron_index_type traverse_and_pick_target(gpu::algorithm::Octree* octree, gpu::algorithm::BarnesHutData* barnes_hut_data, const int& thread_id,
    const RelearnGPUTypes::neuron_index_type& root_index, double& prob_sum, RelearnGPUTypes::number_elements_type* shared_number_elements, gpu::Vector::CudaDouble3* shared_target_positions, const SignalType& signal_type_needed,
    const gpu::Vector::CudaDouble3& source_position, bool use_kernel, const RelearnGPUTypes::neuron_index_type& source_index, const int& number_threads, gpu::random::RandomStateData* random_state_data,
    gpu::kernel::Kernel* kernel);

/**
 * @brief Picks a target to try to generate a synpase with for a given source neuron
 * @param source_index The index of the source neuron
 * @param source_position The position of the source neuron
 * @param root_index The root which is used for prefix traversal
 * @param barnes_hut_data The pointer to the BarnesHutData on the GPU
 * @param signal_type_needed The signal type which the source neuron needs
 * @param num_nodes_gathered The number of nodes gathered during prefix traversal
 * @param octree The pointer to the Octree on the GPU
 * @param kernel The pointer to the Kernel on the GPU
 * @param random_state_data The pointer to the RandomStateData on the GPU
 * @param shared_number_elements The pointer to the shared memory where the number elements of the target are stored per thread
 * @param shared_target_positions The pointer to the shared memory where the position of the target is stored per thread
 * @return The index of the picked target neuron, returns number_neurons + number_virtual_neurons when no node was picked
 */
__device__ RelearnGPUTypes::neuron_index_type pick_target(const RelearnGPUTypes::neuron_index_type& source_index,
    const gpu::Vector::CudaDouble3& source_position,
    const RelearnGPUTypes::neuron_index_type& root_index,
    gpu::algorithm::BarnesHutData* barnes_hut_data,
    const SignalType& signal_type_needed,
    gpu::algorithm::Octree* octree,
    gpu::kernel::Kernel* kernel,
    gpu::random::RandomStateData* random_state_data,
    RelearnGPUTypes::number_elements_type* shared_number_elements,
    gpu::Vector::CudaDouble3* shared_target_positions);

/**
 * @brief Tries to find a target neuron for the given source neuron and then tries to generate a synapse with it
 * @param source_index The index of the source neuron
 * @param source_position The position of the source neuron
 * @param octree The pointer to the Octree on the GPU
 * @param root_index The index of the root node used for traversal
 * @param signal_type_needed The signal type which the source neuron needs
 * @param barnes_hut_data The pointer to the BarnesHutData on the GPU
 * @param kernel The pointer to the Kernel on the GPU
 * @param random_state_data The pointer to the RandomStateData on the GPU
 * @param shared_number_elements The pointer to the shared memory where the number elements of the target are stored per thread
 * @param shared_target_positions The pointer to the shared memory where the position of the target is stored per thread
 * @return True if a target was found and a synapse could be generated, false if not
 */
__device__ bool find_target_neuron(const RelearnGPUTypes::neuron_index_type& source_index,
    const gpu::Vector::CudaDouble3& source_position,
    gpu::algorithm::Octree* octree,
    const RelearnGPUTypes::neuron_index_type& root_index,
    const SignalType& signal_type_needed,
    gpu::algorithm::BarnesHutData* barnes_hut_data,
    gpu::kernel::Kernel* kernel,
    gpu::random::RandomStateData* random_state_data,
    RelearnGPUTypes::number_elements_type* shared_number_elements,
    gpu::Vector::CudaDouble3* shared_target_positions);

/**
 * @brief Tries to generate target neurons for the given source neuron, as many as there are vacant elements in the source neuron
 * @param source_index The index of the source neuron
 * @param source_position The position of the source neuron
 * @param number_vacant_elements The number vacant elements in the source neuron
 * @param octree The pointer to the Octree on the GPU
 * @param signal_type_needed The signal type which the source neuron needs
 * @param barnes_hut_data The pointer to the BarnesHutData on the GPU
 * @param kernel The pointer to the Kernel on the GPU
 * @param random_state_data The pointer to the RandomStateData on the GPU
 * @param shared_number_elements The pointer to the shared memory where the number elements of the target are stored per thread
 * @param shared_target_positions The pointer to the shared memory where the position of the target is stored per thread
 */
__device__ void find_target_neurons(const RelearnGPUTypes::neuron_index_type& source_index,
    const gpu::Vector::CudaDouble3& source_position,
    const uint64_t& number_vacant_elements,
    gpu::algorithm::Octree* octree,
    const SignalType& signal_type_needed,
    gpu::algorithm::BarnesHutData* barnes_hut_data,
    gpu::kernel::Kernel* kernel,
    gpu::random::RandomStateData* random_state_data,
    RelearnGPUTypes::number_elements_type* shared_number_elements,
    gpu::Vector::CudaDouble3* shared_target_positions);

/**
 * @brief Updates the leaf nodes in the octree to have the new number of vacant elements which are stored in the SynapticElements on the GPU
 * @param octree The pointer to the Octree on the GPU
 * @param axons The pointer to the Axons on the GPU
 * @param excitatory_dendrites The pointer to the Excitatory Dendrites on the GPU
 * @param inhibitory_dendrites The pointer to the Inhibitory Dendrites on the GPU
 * @param extra_infos The pointer to the NeuronsExtraInfos on the GPU
 */
__global__ void update_leaf_nodes_kernel(gpu::algorithm::Octree* octree,
    gpu::models::SynapticElements* axons,
    gpu::models::SynapticElements* excitatory_dendrites,
    gpu::models::SynapticElements* inhibitory_dendrites,
    gpu::neurons::NeuronsExtraInfos* extra_infos);
};