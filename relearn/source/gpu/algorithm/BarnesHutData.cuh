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
#include "../Commons.cuh"
#include "../structure/CudaArray.cuh"
#include "../structure/CudaStack.cuh"
#include "../structure/CudaCoalescedStack.cuh"

namespace gpu::algorithm {
struct BarnesHutData {
    /**
     * Struct which holds data needed by the GPU for the Barnes Hut algorithm
     */

    gpu::Vector::CudaArray<RelearnGPUTypes::neuron_id_type> source_ids;
    gpu::Vector::CudaArray<RelearnGPUTypes::neuron_id_type> target_ids;
    gpu::Vector::CudaArray<int> weights;
    unsigned int number_synapses{ 0 };

    RelearnGPUTypes::number_neurons_type neurons_per_thread;

    double acceptance_criterion{ Constants::bh_default_theta };

    // size: num_threads * num_nodes_gathered_before_pick
    gpu::Vector::CudaArray<double> probability_intervals;

    // size of num_threads * num_nodes_gathered_before_pick
    gpu::Vector::CudaArray<RelearnGPUTypes::neuron_index_type> nodes_to_consider;
    uint64_t num_nodes_gathered_before_pick;

    // coalesced stack, size: num_threads * max_level_of_octree * 8
    gpu::Vector::CudaCoalescedStack<RelearnGPUTypes::neuron_index_type> coalesced_prefix_traversal_stack;
};

class BarnesHutDataHandleImpl : public BarnesHutDataHandle {
public:
    /**
     * @brief Constructs new BarnesHutDataHandleImpl given the pointer to the device data and synapse space to allocate
     * @param _dev_ptr The pointer to the device struct
     * @param synapse_space The number of synapses which is an upper bound for the synapses created by barnes hut, in order to allocate enough space on the GPU
     * @param _neurons_per_thread The number of neurons which will be worked on per thread
     * @param _num_nodes_gathered_before_pick The number of nodes which will be gathered before a first pick based on the probability distribution will be done;
     * mainly defines the tradeoff between memory allocated on the GPU and the random number generation overhead
     */
    BarnesHutDataHandleImpl(BarnesHutData* _dev_ptr, unsigned int synapse_space, RelearnGPUTypes::number_neurons_type _neurons_per_thread, uint64_t _num_nodes_gathered_before_pick);

    /**
     * @brief Init function called by the constructor, has to be public in order to be allowed to use device lamdas in it, do not call from outside
     * @param synapse_space The number of synapses which is an upper bound for the synapses created by barnes hut, in order to allocate enough space on the GPU
     */
    void _init(unsigned int synapse_space);

    /**
     * @brief Returns the device pointer to the data on the GPU
     * @return The device pointer to the data on the GPU
     */
    [[nodiscard]] void* get_device_pointer() override;

    /**
     * @brief Returns the number of synapses for which is is currently allocated space for on the GPU
     * @return The number of synapses for which is is currently allocated space for on the GPU
     */
    [[nodiscard]] unsigned int get_number_synapses() override;

    /**
     * @brief Returns the number of neurons assigned to each GPU thread
     * @return The number of neurons assigned to each GPU thread
     */
    [[nodiscard]] RelearnGPUTypes::number_neurons_type get_neurons_per_thread() override;

    /**
     * @brief Returns the source ids of the synapses created by the Barnes Hut algorithm on the GPU
     * @return The source ids of the synapses created by the Barnes Hut algorithm on the GPU
     */
    [[nodiscard]] std::vector<RelearnGPUTypes::neuron_id_type> get_source_ids() override;

    /**
     * @brief Returns the target ids of the synapses created by the Barnes Hut algorithm on the GPU
     * @return The target ids of the synapses created by the Barnes Hut algorithm on the GPU
     */
    [[nodiscard]] std::vector<RelearnGPUTypes::neuron_id_type> get_target_ids() override;

    /**
     * @brief Returns the weights of the synapses created by the Barnes Hut algorithm on the GPU
     * @return The weights of the synapses created by the Barnes Hut algorithm on the GPU
     */
    [[nodiscard]] std::vector<int> get_weights() override;

    /**
     * @brief Returns the block size used to call the Barnes Hut kernel
     * @return The block size used to call the Barnes Hut kernel
     */
    [[nodiscard]] unsigned int get_block_size() override;

    /**
     * @brief Returns the grid size used to call the Barnes Hut kernel
     * @return The grid size used to call the Barnes Hut kernel
     */
    [[nodiscard]] unsigned int get_grid_size() override;

    /**
     * @brief Updates the number of synapses allocated for on the GPU
     * @param synapse_space The new number of synapses which should be allocated
     */
    void update_synapse_allocation(unsigned int synapse_space) override;

    /**
     * @brief Updates various allocation sizes of barnes hut data needed on the GPU which depend on the number of neurons and the maximum depth of the tree,
     * also updates the grid and block size, should be called once after the octree was created and before the kernel is used for the first time
     * @param number_neurons The number of neurons in the simulation
     * @param max_level The maximum depth of the octree
     */
    void update_kernel_allocation_sizes(RelearnGPUTypes::number_neurons_type number_neurons, uint16_t max_level /*, unsigned int num_nodes_to_consider_shared*/) override;

    /**
     * @brief Sets the acceptance criterion to a new value, stored on the GPU
     * @param acceptance_criterion The acceptance criterion to store on the GPU
     */
    void set_acceptance_criterion(double acceptance_criterion) override;

    /**
     * @brief Gets the acceptance criterion stored on the GPU
     * @return The acceptance criterion stored on the GPU
     */
    double get_acceptance_criterion() override;

private:
    BarnesHutData* device_ptr;

    gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::neuron_id_type> handle_source_ids;
    gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::neuron_id_type> handle_target_ids;
    gpu::Vector::CudaArrayDeviceHandle<int> handle_weights;
    unsigned int* handle_number_synapses;

    RelearnGPUTypes::number_neurons_type* handle_neurons_per_thread;
    RelearnGPUTypes::number_neurons_type neurons_per_thread{ 2 };

    double* handle_acceptance_criterion;

    uint64_t* handle_num_nodes_gathered_before_pick;
    uint64_t num_nodes_gathered_before_pick;

    gpu::Vector::CudaArrayDeviceHandle<double> handle_probability_intervals;
    gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::neuron_index_type> handle_nodes_to_consider;

    unsigned int block_size{ 1 };
    unsigned int grid_size{ 1 };

    gpu::Vector::CudaCoalescedStackDeviceHandle<RelearnGPUTypes::neuron_index_type> handle_coalesced_prefix_traversal_stack;
};
};