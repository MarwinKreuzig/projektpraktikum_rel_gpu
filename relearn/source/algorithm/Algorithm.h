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

#include "Types.h"
#include "neurons/UpdateStatus.h"
#include "util/RelearnException.h"

#include <memory>
#include <tuple>
#include <vector>

class NeuronsExtraInfo;
class SynapticElements;

/**
 * This is a virtual interface for all algorithms that can be used to create new synapses.
 * It provides Algorithm::update_connectivity and Algorithm::update_octree.
 */
class Algorithm {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Registers the synaptic elements with the algorithm
     * @param axons The model for the axons
     * @param excitatory_dendrites The model for the excitatory dendrites
     * @param inhibitory_dendrites The model for the inhibitory dendrites
     * @exception Throws a RelearnException if one of the pointers is empty
     */
    void set_synaptic_elements(std::shared_ptr<SynapticElements> axons,
        std::shared_ptr<SynapticElements> excitatory_dendrites, std::shared_ptr<SynapticElements> inhibitory_dendrites) {
        const bool axons_full = axons.operator bool();
        const bool excitatory_dendrites_full = excitatory_dendrites.operator bool();
        const bool inhibitory_dendrites_full = inhibitory_dendrites.operator bool();

        RelearnException::check(axons_full, "Algorithm::set_synaptic_elements: axons was empty");
        RelearnException::check(excitatory_dendrites_full, "Algorithm::set_synaptic_elements: excitatory_dendrites was empty");
        RelearnException::check(inhibitory_dendrites_full, "Algorithm::set_synaptic_elements: inhibitory_dendrites was empty");

        this->axons = std::move(axons);
        this->excitatory_dendrites = std::move(excitatory_dendrites);
        this->inhibitory_dendrites = std::move(inhibitory_dendrites);
    }

    /**
     * @brief Updates the connectivity with the algorithm. Already updates the synaptic elements, i.e., the axons and dendrites (both excitatory and inhibitory).
     *      Does not update the network graph. Performs communication with MPI
     * @param number_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so, the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @exception Can throw a RelearnException
     * @return A tuple with the created synapses that must be committed to the network graph
     */
    [[nodiscard]] virtual std::tuple<LocalSynapses, DistantInSynapses, DistantOutSynapses> update_connectivity(number_neurons_type number_neurons,
        const std::vector<UpdateStatus>& disable_flags, const std::unique_ptr<NeuronsExtraInfo>& extra_infos)
        = 0;

    /**
     * @brief Updates the octree according to the necessities of the algorithm.
     *      Performs communication via MPI
     * @param disable_flags Flags that indicate if a neuron id disabled or enabled. If disabled, it is ignored for all purposes
     * @exception Can throw a RelearnException
     */
    virtual void update_octree(const std::vector<UpdateStatus>& disable_flags) = 0;

protected:
    std::shared_ptr<SynapticElements> axons{}; // NOLINT
    std::shared_ptr<SynapticElements> excitatory_dendrites{}; // NOLINT
    std::shared_ptr<SynapticElements> inhibitory_dendrites{}; // NOLINT
};
