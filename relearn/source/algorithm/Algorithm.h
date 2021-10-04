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

#include "../neurons/helper/SynapseCreationRequests.h"
#include "../util/RelearnException.h"

#include <memory>
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
     * @brief Sets probability parameter used to determine the probability for a cell of being selected
     * @param sigma The probability parameter, >= 0.0
     * @exception Throws a RelearnExeption if sigma < 0.0
     */
    void set_probability_parameter(const double sigma) {
        RelearnException::check(sigma > 0.0, "In Algorithm::set_probability_parameter, sigma was not greater than 0");
        this->sigma = sigma;
    }

    /**
     * @brief Returns the currently used probability parameter
     * @return The currently used probability parameter
     */
    [[nodiscard]] double get_probabilty_parameter() const noexcept {
        return sigma;
    }

    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons
     * @param num_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so (== 0), the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @param axons The axon model that is used
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    [[nodiscard]] virtual MapSynapseCreationRequests find_target_neurons(size_t num_neurons, const std::vector<char>& disable_flags,
        const std::unique_ptr<NeuronsExtraInfo>& extra_infos, const std::unique_ptr<SynapticElements>& axons)
        = 0;

    /**
     * @brief Updates all leaf nodes in the octree by the algorithm
     * @param disable_flags Flags that indicate if a neuron id disabled (0) or enabled (otherwise)
     * @param axons The model for the axons
     * @param excitatory_dendrites The model for the excitatory dendrites
     * @param inhibitory_dendrites The model for the inhibitory dendrites
     * @exception Throws a RelearnException if the vectors have different sizes or the leaf nodes are not in order of their neuron id
     */
    virtual void update_leaf_nodes(const std::vector<char>& disable_flags, const std::unique_ptr<SynapticElements>& axons,
        const std::unique_ptr<SynapticElements>& excitatory_dendrites, const std::unique_ptr<SynapticElements>& inhibitory_dendrites)
        = 0;

private:
    double sigma{ default_sigma };

public:
    constexpr static double default_sigma{ 750.0 };
};