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
#include "../neurons/SignalType.h"
#include "../util/Vec3.h"

#include <optional>
#include <vector>

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
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication
     * @param src_neuron_id The neuron's id that wants to connect. Is used to disallow autapses (connections to itself)
     * @param axon_pos_xyz The neuron's position that wants to connect. Is used in probability computations
     * @param dendrite_type_needed The signal type that is searched.
     * @exception Can throw a RelearnException
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorihtm found a matching neuron, it's id and MPI rank are returned.
     */
    virtual [[nodiscard]] std::optional<RankNeuronId> find_target_neuron(size_t src_neuron_id, const Vec3d& axon_pos_xyz, SignalType dendrite_type_needed) = 0;
    
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