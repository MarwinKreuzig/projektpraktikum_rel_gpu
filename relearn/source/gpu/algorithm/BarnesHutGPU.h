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

#include "../../algorithm/BarnesHutInternal/BarnesHut.h"
#include "../structure/GpuDataStructures.h"
#include "BarnesHutKernel.h"

class BarnesHutGPU : public BarnesHut {
    /**
    * Class which represents the BarnesHut algorithm on the GPU. Overrides specific functionality to now run on the GPU
    */
public:
    /**
    * @brief Constructs the Barnes Hut algorithm for the GPU
    * @param octree The octree, assumes that all leaf nodes were already initialized
    */
    explicit BarnesHutGPU(const std::shared_ptr<OctreeImplementation<BarnesHutCell>>& octree);

    /**
    * @brief Constructs the Barnes Hut algorithm for the GPU
    * @param octree The octree, assumes that all leaf nodes were already initialized
    * @param num_nodes_gathered_before_pick The number of nodes gathered before first node is picked as potential target;
    * essentially defines memory vs random number generation tradeoff
    * @param neurons_per_thread The number of neurons which will be worked on by a single thread
    */
    explicit BarnesHutGPU(const std::shared_ptr<OctreeImplementation<BarnesHutCell>>& octree, uint64_t num_nodes_gathered_before_pick, RelearnGPUTypes::number_neurons_type neurons_per_thread);

    /**
    * @brief Updates the number of free elements in the leaf nodes and virtual neurons on the GPU in a bottom up fashion, also adjusts virtual neuron position
    */
    void update_octree() override;

    /**
    * @brief Does the Barnes Hut Algorithm on the GPU and creates new synapses
    * @param number_neurons The number of neurons in the simulation
    * @return Currently returns the created synapses to the CPU
    */
    [[nodiscard]] std::tuple<PlasticLocalSynapses, PlasticDistantInSynapses, PlasticDistantOutSynapses> update_connectivity(number_neurons_type number_neurons) override;

    /**
    * @brief Sets the acceptance criterion on the GPU and the underlying base BarnesHut class
    * @param acceptance_criterion The acceptance criterion
    */
    void set_acceptance_criterion(double acceptance_criterion) override;

protected:

    /**
    * @brief Updates the number of free elements in the leaf nodes from the SynapticElements data on the GPU
    */
    void update_leaf_nodes() override;

private:
    std::unique_ptr<gpu::algorithm::BarnesHutDataHandle> gpu_handle;

    unsigned int synapse_space{0};

    number_neurons_type number_neurons{0};

    int update_leaf_nodes_kernel_block_size{0};
    int update_leaf_nodes_kernel_grid_size{0};
};
