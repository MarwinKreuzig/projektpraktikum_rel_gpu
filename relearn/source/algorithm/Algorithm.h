/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include "../neurons/helper/RankNeuronId.h"
#include "../neurons/helper/SynapseCreationRequests.h"
#include "../neurons/SignalType.h"
#include "../util/Vec3.h"

#include <optional>
#include <vector>

class NeuronsExtraInfo;
class SynapticElements;

/**
 * This is a virtual interface for all algorithms that can be used to find target neurons with vacant dendrites.
 * It provides Algorithm::find_target_neuron, Algorithm::update_leaf_nodes and every derived class must also implement 
 * static void update_functor(OctreeNode<Cell>* node)
 * with Cell being exposed publicly via
 * using AdditionalCellAttributes = Cell;
 */
class Algorithm {
public:    
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons
     * @param num_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so (== 0), the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @param axons The axon model that is used
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    virtual [[nodiscard]] MapSynapseCreationRequests find_target_neurons(size_t num_neurons, const std::vector<char>& disable_flags,
        const std::unique_ptr<NeuronsExtraInfo>& extra_infos, const std::unique_ptr<SynapticElements>& axons)
        = 0;

    /**
     * @brief Updates all leaf nodes in the octree by the algorithm
     * @param disable_flags Flags that indicate if a neuron id disabled (0) or enabled (otherwise)
     * @param dendrites_excitatory_counts The number of total excitatory dendrites, accessed via operator[] with the neuron ids
     * @param dendrites_excitatory_connected_counts The number of connected excitatory dendrites, accessed via operator[] with the neuron ids
     * @param dendrites_inhibitory_counts The number of total inhibitory dendrites, accessed via operator[] with the neuron ids
     * @param dendrites_inhibitory_connected_counts The number of connected inhibitory dendrites, accessed via operator[] with the neuron ids
     * @exception Can throw a RelearnException
     */
    virtual void update_leaf_nodes(const std::vector<char>& disable_flags,
        const std::vector<double>& dendrites_excitatory_counts, const std::vector<unsigned int>& dendrites_excitatory_connected_counts,
        const std::vector<double>& dendrites_inhibitory_counts, const std::vector<unsigned int>& dendrites_inhibitory_connected_counts)
        = 0;
};