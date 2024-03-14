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
#include "../Commons.cuh"
#include "../structure/CudaArray.cuh"
#include "BarnesHutData.cuh"
#include "../structure/Octree.cuh"
#include "../neurons/NeuronsExtraInfos.cuh"
#include "../neurons/models/SynapticElements.cuh"
#include "../utils/CudaMath.cuh"
#include "../utils/RandomNew.cuh"
#include "kernel/KernelGPU.cuh"
#include "BarnesHutKernel.cuh"

namespace gpu::kernel {

    __device__ AcceptanceStatus test_acceptance_criterion(const gpu::Vector::CudaDouble3& source_position, const RelearnGPUTypes::neuron_index_type& target_index, const double acceptance_criterion, gpu::algorithm::Octree* octree, const RelearnGPUTypes::number_elements_type& number_elements,
        const gpu::Vector::CudaDouble3& target_pos) {
        
        RelearnGPUException::device_check(acceptance_criterion > 0.0,
            "gpu::kernel::test_acceptance_criterion: The acceptance criterion was not positive: ({})", acceptance_criterion);
        RelearnGPUException::device_check(target_index < octree->number_neurons + octree->number_virtual_neurons,
            "gpu::kernel::test_acceptance_criterion: target_index was invalid");
        
        // Never accept a node with zero vacant elements
        if (number_elements <= 0) {
            return AcceptanceStatus::Discard;
        }

        auto distance_vector = (target_pos - source_position).to_double3();;
        const auto distance = CudaMath::calculate_2_norm(distance_vector);

        // No autapse
        if (distance == 0.0) {
            return AcceptanceStatus::Discard;
        }

        // Always accept a leaf node
        if (target_index < octree->number_neurons) {
            return AcceptanceStatus::Accept;
        }

        const auto length = octree->get_maximal_dimension_difference(target_index);

        // Original Barnes-Hut acceptance criterion
        // const auto ret_val = (length / distance) < acceptance_criterion;
        const auto ret_val = length < (acceptance_criterion * distance);
        return ret_val ? AcceptanceStatus::Accept : AcceptanceStatus::Expand;
    }

    __device__ inline double zero_prob_sum_case_kernel(const RelearnGPUTypes::neuron_index_type& source_index, const gpu::Vector::CudaDouble3& source_position, const RelearnGPUTypes::neuron_index_type& target_index, 
        const RelearnGPUTypes::number_elements_type& number_elements, const gpu::Vector::CudaDouble3& target_position) {

        if (target_index == source_index) {
            return 0.0;
        }

        auto diff_vector = (source_position - target_position).to_double3();;
        return static_cast<double>(number_elements) / (CudaMath::calculate_2_norm(diff_vector));
    }

    __device__ inline RelearnGPUTypes::neuron_index_type pick_target_randomly_from_dist(gpu::random::RandomStateData* random_state_data, const double &prob_sum, gpu::algorithm::BarnesHutData* barnes_hut_data,
        const uint64_t &num_nodes_gathered, const int &thread_id, const int &number_threads) {

        const auto random_percent = random_state_data->get_percentage(gpu::random::RandomKeyHolder::BARNES_HUT, thread_id);
        const auto random_number = random_percent * prob_sum;

        RelearnGPUException::device_check(random_number >= 0.0, "gpu::kernel::randomly_pick_target: random_number was smaller than 0.0");

        size_t counter = 0;
        auto sum_probabilities = 0.0;

        for (; counter < num_nodes_gathered && sum_probabilities < random_number; counter++) {
            sum_probabilities += barnes_hut_data->probability_intervals[thread_id + counter * number_threads];
        }

        RelearnGPUException::device_check(sum_probabilities > 0.0, "gpu::kernel::randomly_pick_target: The sum of probabilities was <= 0.0");

        while (barnes_hut_data->probability_intervals[thread_id + (counter - 1) * number_threads] <= 0.0) {
            // Ignore all probabilities that are <= 0.0
            counter--;
        }

        return barnes_hut_data->nodes_to_consider[thread_id + number_threads * (counter - 1)];
    }

    __device__ inline RelearnGPUTypes::neuron_index_type traverse_and_pick_target(gpu::algorithm::Octree* octree, gpu::algorithm::BarnesHutData* barnes_hut_data, const int& thread_id,
        const RelearnGPUTypes::neuron_index_type& root_index, double& prob_sum, RelearnGPUTypes::number_elements_type* shared_number_elements, gpu::Vector::CudaDouble3* shared_target_positions, const SignalType& signal_type_needed, 
        const gpu::Vector::CudaDouble3& source_position, bool use_kernel, const RelearnGPUTypes::neuron_index_type& source_index, const int &number_threads, gpu::random::RandomStateData* random_state_data, 
        gpu::kernel::Kernel* kernel) {

        const auto add_children = [&](RelearnGPUTypes::neuron_index_type virtual_neuron_index) {
            auto smaller_index = virtual_neuron_index - octree->number_neurons;
            for (int i = 0; i < octree->num_children[smaller_index]; i++) {
                barnes_hut_data->coalesced_prefix_traversal_stack.push(octree->child_indices[smaller_index + i * octree->number_virtual_neurons], thread_id);
            }
        };

        barnes_hut_data->coalesced_prefix_traversal_stack.reset(thread_id);
        // The algorithm expects that root is not considered directly, rather its children
        add_children(root_index);

        prob_sum = 0.0;
        RelearnGPUTypes::neuron_index_type current_pick = octree->number_neurons + octree->number_virtual_neurons;
        uint64_t current_num_nodes_gathered = 0;
        while (!barnes_hut_data->coalesced_prefix_traversal_stack.empty(thread_id)) {
            // Get top-of-stack node and remove it
            auto current_node_index = barnes_hut_data->coalesced_prefix_traversal_stack.top(thread_id);
            barnes_hut_data->coalesced_prefix_traversal_stack.pop(thread_id);
            
            shared_number_elements[threadIdx.x] = octree->get_num_free_elements_for_signal(signal_type_needed, current_node_index);
            shared_target_positions[threadIdx.x] = octree->get_position_for_signal(signal_type_needed, current_node_index);

            /**
             * Should node be used for probability interval?
             * Only take those that have the required elements
             */
            const auto status = test_acceptance_criterion(source_position, current_node_index, 
                barnes_hut_data->acceptance_criterion, octree, shared_number_elements[threadIdx.x], shared_target_positions[threadIdx.x]);

            if (status == AcceptanceStatus::Expand) {
                // Need to expand
                add_children(current_node_index);
                continue;
            }

            if (status == AcceptanceStatus::Accept) {
                double prob;
                const auto& target_position = shared_target_positions[threadIdx.x];
                const auto& number_elements = shared_number_elements[threadIdx.x];
                if (use_kernel) {
                    prob = kernel->calculate_attractiveness_to_connect(source_index, source_position, current_node_index, number_elements, target_position);
                } else {
                    prob = zero_prob_sum_case_kernel(source_index, source_position, current_node_index, number_elements, target_position);
                }
                

                prob_sum += prob;

                // add to the arrays
                barnes_hut_data->nodes_to_consider[thread_id + number_threads * current_num_nodes_gathered] = current_node_index;
                barnes_hut_data->probability_intervals[thread_id + current_num_nodes_gathered * number_threads] = prob;
                
                if (++current_num_nodes_gathered == barnes_hut_data->num_nodes_gathered_before_pick && prob_sum != 0.0) {
                    current_pick = pick_target_randomly_from_dist(random_state_data, prob_sum, barnes_hut_data, current_num_nodes_gathered, thread_id, number_threads);

                    current_num_nodes_gathered = 1;

                    barnes_hut_data->nodes_to_consider[thread_id] = current_pick;
                    barnes_hut_data->probability_intervals[thread_id] = prob_sum;
                }
            }
        }

        // Now check if there are any nodes left to be calculated on
        if (current_num_nodes_gathered > 1 && prob_sum != 0.0) {
            current_pick = pick_target_randomly_from_dist(random_state_data, prob_sum, barnes_hut_data, current_num_nodes_gathered, thread_id, number_threads);
        }

        return current_pick;
    }

    __device__ RelearnGPUTypes::neuron_index_type pick_target(const RelearnGPUTypes::neuron_index_type& source_index, 
                                    const gpu::Vector::CudaDouble3& source_position,
                                    const RelearnGPUTypes::neuron_index_type& root_index,
                                    gpu::algorithm::BarnesHutData* barnes_hut_data, 
                                    const SignalType& signal_type_needed, 
                                    gpu::algorithm::Octree* octree,
                                    gpu::kernel::Kernel* kernel,
                                    gpu::random::RandomStateData* random_state_data,
                                    RelearnGPUTypes::number_elements_type* shared_number_elements,
                                    gpu::Vector::CudaDouble3* shared_target_positions) {

        RelearnGPUException::device_check(barnes_hut_data->acceptance_criterion > 0.0,
            "gpu::kernel::gather_pick_target: The acceptance criterion was not positive: ({})", barnes_hut_data->acceptance_criterion);
        RelearnGPUException::device_check(root_index < octree->number_neurons + octree->number_virtual_neurons,
            "gpu::kernel::gather_pick_target: root_index was invalid");


        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        int number_threads = blockDim.x * gridDim.x;

        shared_number_elements[threadIdx.x] = octree->get_num_free_elements_for_signal(signal_type_needed, root_index);
        if (shared_number_elements[threadIdx.x] <= 0) {
            return octree->number_neurons + octree->number_virtual_neurons;
        }
        
        if (root_index < octree->number_neurons) {
            shared_target_positions[threadIdx.x] = octree->get_position_for_signal(signal_type_needed, root_index);

            const auto status = test_acceptance_criterion(source_position, root_index, barnes_hut_data->acceptance_criterion, octree,
                shared_number_elements[threadIdx.x], shared_target_positions[threadIdx.x]);

            if (status != AcceptanceStatus::Discard) {
                return root_index;
            }

            return octree->number_neurons + octree->number_virtual_neurons;
        }

        double prob_sum = 0.0;
        uint64_t current_pick = traverse_and_pick_target(octree, barnes_hut_data, thread_id, root_index, prob_sum, shared_number_elements, 
            shared_target_positions, signal_type_needed, source_position, true, source_index, number_threads, random_state_data, kernel);

        if (prob_sum == 0.0) {
            current_pick = traverse_and_pick_target(octree, barnes_hut_data, thread_id, root_index, prob_sum, shared_number_elements, shared_target_positions, 
                signal_type_needed, source_position, false, source_index, number_threads, random_state_data, kernel);
        }

        if (prob_sum == 0.0) {
            // If the vector still contains only the same node, return nothing
            return octree->number_neurons + octree->number_virtual_neurons;
        }

        RelearnGPUException::device_check(current_pick != source_index, "gpu::kernel::pick_target: The picked neuron and source are the same!");
        return current_pick;
    }

    __device__ bool find_target_neuron(const RelearnGPUTypes::neuron_index_type& source_index, 
                                       const gpu::Vector::CudaDouble3& source_position, 
                                       gpu::algorithm::Octree* octree,
                                       const RelearnGPUTypes::neuron_index_type& root_index,
                                       const SignalType& signal_type_needed, 
                                       gpu::algorithm::BarnesHutData* barnes_hut_data,
                                       gpu::kernel::Kernel* kernel,
                                       gpu::random::RandomStateData* random_state_data,
                                       RelearnGPUTypes::number_elements_type* shared_number_elements,
                                       gpu::Vector::CudaDouble3* shared_target_positions) {
        
        RelearnGPUException::device_check(barnes_hut_data->acceptance_criterion > 0.0,
            "gpu::kernel::find_target_neuron: The acceptance criterion was not positive: ({})", barnes_hut_data->acceptance_criterion);
        RelearnGPUException::device_check( root_index < octree->number_neurons + octree->number_virtual_neurons, "gpu::kernel::find_target_neuron: root invalid");

        if (root_index == source_index) {
            return false;
        }

        const auto try_creating_synapse = [&](const RelearnGPUTypes::neuron_index_type target_index) {
	        int num_free_elements;
            if (signal_type_needed == SignalType::Excitatory) {
                num_free_elements = atomicSub(&(octree->num_free_elements_excitatory[target_index]), 1);
            } else {
                num_free_elements = atomicSub(&(octree->num_free_elements_inhibitory[target_index]), 1);
            }

            if (num_free_elements > 0) {
                // Insert
                int synapse_index = atomicAdd(&(barnes_hut_data->number_synapses), (uint64_t)1);
                barnes_hut_data->source_ids[synapse_index] = octree->neuron_ids[source_index];
                barnes_hut_data->target_ids[synapse_index] = octree->neuron_ids[target_index];
                const int weight = (SignalType::Inhibitory == signal_type_needed) ? -1 : 1;
                barnes_hut_data->weights[synapse_index] = weight;
            }
	    };

        if (root_index < octree->number_neurons) {
            try_creating_synapse(root_index);

            return true;
        }

        for (auto root_of_subtree_index = root_index; true;) {
            RelearnGPUTypes::neuron_index_type node_selected_index = pick_target(source_index, source_position, root_of_subtree_index, barnes_hut_data, 
                signal_type_needed, octree, kernel, random_state_data, shared_number_elements, shared_target_positions);
            if (node_selected_index >= octree->number_neurons + octree->number_virtual_neurons) {
                return false;
            }

            // A chosen child is a valid target
            if (node_selected_index < octree->number_neurons) {
                try_creating_synapse(node_selected_index);

                return true;
            }

            // We need to choose again, starting from the chosen virtual neuron
            root_of_subtree_index = node_selected_index;
        }
    }

    __device__ void find_target_neurons(const RelearnGPUTypes::neuron_index_type& source_index, 
                                        const gpu::Vector::CudaDouble3& source_position, 
                                        const uint64_t& number_vacant_elements, 
                                        gpu::algorithm::Octree* octree, 
                                        const SignalType& signal_type_needed, 
                                        gpu::algorithm::BarnesHutData* barnes_hut_data,
                                        gpu::kernel::Kernel* kernel,
                                        gpu::random::RandomStateData* random_state_data,
                                        RelearnGPUTypes::number_elements_type* shared_number_elements,
                                        gpu::Vector::CudaDouble3* shared_target_positions) {
                       
        RelearnGPUException::device_check(barnes_hut_data->acceptance_criterion > 0.0,
            "BarnesHutBase::find_target_neurons: The acceptance criterion was not positive: ({})", barnes_hut_data->acceptance_criterion);
        
        for (uint64_t j = 0; j < number_vacant_elements; j++) {
            // Find one target at the time
            // Returns false if find neuron failed, but not because another neurons was faster at connecting to the target neuron
            uint64_t root_index = octree->number_neurons + octree->number_virtual_neurons - 1;
            bool result = find_target_neuron(source_index, 
                                             source_position, 
                                             octree,
                                             root_index,
                                             signal_type_needed, 
                                             barnes_hut_data,
                                             kernel,
                                             random_state_data,
                                             shared_number_elements,
                                             shared_target_positions);

            if (!result) {
                break;
            }
        }
    }

    __global__ void update_connectivity_kernel(gpu::algorithm::BarnesHutData* barnes_hut_data, 
                                               gpu::algorithm::Octree* octree,
                                               gpu::models::SynapticElements* synaptic_elements,
                                               gpu::neurons::NeuronsExtraInfos* neurons_extra_infos,
                                               gpu::kernel::Kernel* kernel,
                                               gpu::random::RandomStateData* random_state_data) {

        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        int warp_id = thread_id / warpSize;
        int thread_lane = thread_id % warpSize;

        // We want to coalesce the neuron indices for every warp, and minimize the stride of the warps (stride becomes 32), this results in this pattern
        unsigned int octree_node_index = blockIdx.x * blockDim.x * barnes_hut_data->neurons_per_thread + thread_lane + barnes_hut_data->neurons_per_thread * warpSize * warp_id;

        if (octree_node_index >= octree->number_neurons) {
            return;
        }

        if (thread_id == 0) {
            barnes_hut_data->number_synapses = 0;
        }
        __threadfence();

        // The shared memory as it is used now does not increase performance by much, if anything
        // Other techniques to try to make use of it failed though as well, so one might have to think about it a bit more
        extern __shared__ gpu::Vector::CudaDouble3 shared_memory_data[];

        gpu::Vector::CudaDouble3* shared_target_positions = shared_memory_data; 
        RelearnGPUTypes::number_elements_type* shared_number_elements = (RelearnGPUTypes::number_elements_type*)&shared_target_positions[blockDim.x];

        for (int i = 0; i < barnes_hut_data->neurons_per_thread; ++i) {
            RelearnGPUTypes::neuron_index_type source_index = octree_node_index + i * warpSize;
            if (source_index >= octree->number_neurons) {
                return;
            }

            auto neuron_id = octree->neuron_ids[source_index];

            if (neurons_extra_infos->disable_flags[neuron_id] != UpdateStatus::Enabled) {
                continue;
            }

            const auto number_vacant_axons = synaptic_elements->get_free_elements(neuron_id);
            if (number_vacant_axons == 0) {
                continue;
            }

            find_target_neurons(source_index, 
                                neurons_extra_infos->positions[neuron_id], 
                                number_vacant_axons, 
                                octree, 
                                synaptic_elements->signal_types[neuron_id], 
                                barnes_hut_data,
                                kernel,
                                random_state_data,
                                shared_number_elements,
                                shared_target_positions);
        }

        // Reset num_free_elements in octree
        for (int i = 0; i < barnes_hut_data->neurons_per_thread; ++i) {
            RelearnGPUTypes::neuron_index_type source_index = octree_node_index + i * blockDim.x;
            octree->num_free_elements_excitatory[source_index] = max(0, octree->num_free_elements_excitatory[source_index]);
            octree->num_free_elements_inhibitory[source_index] = max(0, octree->num_free_elements_inhibitory[source_index]);
        }
    }

    std::vector<gpu::Synapse> update_connectivity_gpu(gpu::algorithm::BarnesHutDataHandle* gpu_handle, 
                                                      void* octree_device_ptr,
                                                      void* axons_device_ptr,
                                                      void* neurons_extra_infos_device_ptr,
                                                      void* kernel_device_ptr) {

        auto grid_size = gpu_handle->get_grid_size();
        auto block_size = gpu_handle->get_block_size();
        unsigned int number_warps_per_block = (unsigned int)std::ceil((double)(block_size) / 32.);

        cudaDeviceSynchronize();
        gpu_check_last_error();
        update_connectivity_kernel<<<grid_size,block_size,block_size*sizeof(gpu::Vector::CudaDouble3) + block_size*sizeof(int)>>>(static_cast<gpu::algorithm::BarnesHutData*>(gpu_handle->get_device_pointer()),
                                            static_cast<gpu::algorithm::Octree*>(octree_device_ptr),
                                            static_cast<gpu::models::SynapticElements*>(axons_device_ptr),
                                            static_cast<gpu::neurons::NeuronsExtraInfos*>(neurons_extra_infos_device_ptr),
                                            static_cast<gpu::kernel::Kernel*>(kernel_device_ptr),
                                            gpu::random::RandomHolder::get_instance().get_device_pointer());
        cudaDeviceSynchronize();
        gpu_check_last_error();

        unsigned int number_synapses = gpu_handle->get_number_synapses();
        std::vector<RelearnGPUTypes::neuron_id_type> source_ids = gpu_handle->get_source_ids();
        std::vector<RelearnGPUTypes::neuron_id_type> target_ids = gpu_handle->get_target_ids();
        std::vector<int> weights = gpu_handle->get_weights();

        std::vector<gpu::Synapse> output;
        output.reserve(number_synapses);
        for (int i = 0; i < number_synapses; i++) {
            output.emplace_back(target_ids[i], source_ids[i], weights[i]);
        }

        return output;
    }

    // number of threads in the grid when calling this has to be equal to number_neurons
    // this kernel also currently assumes that we are dealing either with barnes hut (both dendrites exist) or barnes hut inverse (both axons exist)
    __global__ void update_leaf_nodes_kernel(gpu::algorithm::Octree* octree,
                                             gpu::models::SynapticElements* axons,
                                             gpu::models::SynapticElements* excitatory_dendrites,
                                             gpu::models::SynapticElements* inhibitory_dendrites,
                                             gpu::neurons::NeuronsExtraInfos* extra_infos) {

        uint64_t octree_node_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (octree_node_index >= octree->number_neurons) {
            return;
        }
        
        auto neuron_id = octree->neuron_ids[octree_node_index];

        if (extra_infos->disable_flags[neuron_id] != UpdateStatus::Enabled) {
            octree->num_free_elements_excitatory[octree_node_index] = 0;
            octree->num_free_elements_inhibitory[octree_node_index] = 0;

            return;
        }

        if (octree->stored_element_type == ElementType::Axon) {
            if (axons->signal_types[neuron_id] == SignalType::Excitatory) {
                octree->num_free_elements_excitatory[octree_node_index] = axons->get_free_elements(neuron_id);
                octree->num_free_elements_inhibitory[octree_node_index] = 0;
            } else {
                octree->num_free_elements_excitatory[octree_node_index] = 0;
                octree->num_free_elements_inhibitory[octree_node_index] = axons->get_free_elements(neuron_id);
            }
        } else {
            octree->num_free_elements_excitatory[octree_node_index] = excitatory_dendrites->get_free_elements(neuron_id);
            octree->num_free_elements_inhibitory[octree_node_index] = inhibitory_dendrites->get_free_elements(neuron_id);
        }
    }

    void update_leaf_nodes(void* octree_device_ptr,
                           void* axons_device_ptr,
                           void* ex_dendrites_ptr,
                           void* in_dendrites_ptr,
                           void* neurons_extra_infos_device_ptr,
                           uint64_t number_neurons,
                           int& grid_size,
                           int& block_size) {

        if (block_size <= 0) {
           auto [new_grid_size, new_block_size] = get_number_blocks_and_threads(update_leaf_nodes_kernel, number_neurons);
           grid_size = new_grid_size;
           block_size = new_block_size;
        }
        
        cudaDeviceSynchronize();
        gpu_check_last_error();
        update_leaf_nodes_kernel<<<grid_size,block_size>>>((gpu::algorithm::Octree*)octree_device_ptr,
                                          (gpu::models::SynapticElements*)axons_device_ptr,
                                          (gpu::models::SynapticElements*)ex_dendrites_ptr,
                                          (gpu::models::SynapticElements*)in_dendrites_ptr,
                                          (gpu::neurons::NeuronsExtraInfos*)neurons_extra_infos_device_ptr);
        cudaDeviceSynchronize();
        gpu_check_last_error();
    }
};