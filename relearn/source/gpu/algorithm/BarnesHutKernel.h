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

#include "../utils/Interface.h"

namespace gpu::algorithm {
    class BarnesHutDataHandle {
        /**
        * Class which acts as a handle to the GPU data needed for the Barnes Hut algorithm
        */
    public:
        /**
         * @brief Returns the device pointer to the data on the GPU
         * @return The device pointer to the data on the GPU
         */
        [[nodiscard]] virtual void* get_device_pointer() = 0;

        /**
         * @brief Returns the number of synapses for which is is currently allocated space for on the GPU
         * @return The number of synapses for which is is currently allocated space for on the GPU
         */
        [[nodiscard]] virtual unsigned int get_number_synapses() = 0;

        /**
         * @brief Returns the number of neurons assigned to each GPU thread
         * @return The number of neurons assigned to each GPU thread
         */
        [[nodiscard]] virtual RelearnGPUTypes::number_neurons_type get_neurons_per_thread() = 0;

        /**
         * @brief Returns the source ids of the synapses created by the Barnes Hut algorithm on the GPU
         * @return The source ids of the synapses created by the Barnes Hut algorithm on the GPU
         */
        [[nodiscard]] virtual std::vector<RelearnGPUTypes::neuron_id_type> get_source_ids() = 0;

        /**
         * @brief Returns the target ids of the synapses created by the Barnes Hut algorithm on the GPU
         * @return The target ids of the synapses created by the Barnes Hut algorithm on the GPU
         */
        [[nodiscard]] virtual std::vector<RelearnGPUTypes::neuron_id_type> get_target_ids() = 0;

         /**
         * @brief Returns the weights of the synapses created by the Barnes Hut algorithm on the GPU
         * @return The weights of the synapses created by the Barnes Hut algorithm on the GPU
         */
        [[nodiscard]] virtual std::vector<int> get_weights() = 0;

        /**
         * @brief Returns the block size used to call the Barnes Hut kernel
         * @return The block size used to call the Barnes Hut kernel
         */
        [[nodiscard]] virtual unsigned int get_block_size() = 0;

        /**
         * @brief Returns the grid size used to call the Barnes Hut kernel
         * @return The grid size used to call the Barnes Hut kernel
         */
        [[nodiscard]] virtual unsigned int get_grid_size() = 0;

        /**
         * @brief Updates the number of synapses allocated for on the GPU
         * @param synapse_space The new number of synapses which should be allocated
         */
        virtual void update_synapse_allocation(unsigned int synapse_space) = 0;

        /**
         * @brief Updates various allocation sizes of barnes hut data needed on the GPU which depend on the number of neurons and the maximum depth of the tree, 
         * also updates the grid and block size, should be called once after the octree was created and before the kernel is used for the first time
         * @param number_neurons The number of neurons in the simulation
         * @param max_level The maximum depth of the octree
         */
        virtual void update_kernel_allocation_sizes(RelearnGPUTypes::number_neurons_type number_neurons, uint16_t max_level) = 0;

        /**
         * @brief Sets the acceptance criterion to a new value, stored on the GPU
         * @param acceptance_criterion The acceptance criterion to store on the GPU
         */
        virtual void set_acceptance_criterion(double acceptance_criterion) = 0;

        /**
         * @brief Gets the acceptance criterion stored on the GPU
         * @return The acceptance criterion stored on the GPU
         */
        virtual double get_acceptance_criterion() = 0;
    };

    /**
    * @brief Creates a new BarnesHutDataHandle
    * @param synapse_space The number of synpses that should be allocated on the GPU
    * @param num_nodes_gathered_before_pick The number of nodes gathered before first node is picked as potential target;
    * essentially defines memory vs random number generation tradeoff
    * @param neurons_per_thread The number of neurons which will be worked on by a single thread
    * @return A unique pointer to a new BarnesHutDataHandle
    */
    std::unique_ptr<BarnesHutDataHandle> create_barnes_hut_data(unsigned int synapse_space, uint64_t num_nodes_gathered_before_pick, RelearnGPUTypes::number_neurons_type neurons_per_thread = 2);
}

namespace gpu::kernel {

    enum class AcceptanceStatus : char {
        /**
        * Enum which determines whether a node should be discarded, expanded or accepted during prefix traversal
        */
        Discard = 0,
        Expand = 1,
        Accept = 2,
    };

    /**
    * @brief Does the Barnes Hut algorithm on the GPU
    * @param gpu_handle The pointer to the GPU handle of the Barnes Hut data
    * @param octree_device_ptr The pointer to the Octree on the GPU
    * @param axons_device_ptr The pointer to the Axons on the GPU
    * @param neurons_extra_infos_device_ptr The pointer to the NeuronsExtraInfos on the GPU
    * @param kernel_device_ptr The pointer to the Kernel on the GPU
    * @return The synapses which were created
    */
    [[nodiscard]] std::vector<gpu::Synapse> update_connectivity_gpu(gpu::algorithm::BarnesHutDataHandle* gpu_handle,
                                                                    void* octree_device_ptr,
                                                                    void* axons_device_ptr,
                                                                    void* neurons_extra_infos_device_ptr,
                                                                    void* kernel_device_ptr);

    /**
    * @brief Updates the leaf nodes to the number of vacant elements in the SynapticElements on the GPU
    * @param octree_device_ptr The pointer to the Octree on the GPU
    * @param axons_device_ptr The pointer to the Axons on the GPU
    * @param ex_dendrites_ptr The pointer to the Excitatory Dendrites on the GPU
    * @param in_dendrites_ptr The pointer to the Inhibitory Dendrites on the GPU
    * @param neurons_extra_infos_device_ptr The pointer to the NeuronsExtraInfos on the GPU
    * @param number_neurons The number of neurons in the simulation
    */
    void update_leaf_nodes(void* octree_device_ptr,
                           void* axons_device_ptr,
                           void* ex_dendrites_ptr,
                           void* in_dendrites_ptr,
                           void* neurons_extra_infos_device_ptr,
                           RelearnGPUTypes::number_neurons_type number_neurons,
                           int& grid_size,
                           int& block_size);
}