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

#include "algorithm/Internal/ExchangingAlgorithm.h"

#include "Config.h"
#include "Types.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "mpi/CommunicationMap.h"
#include "neurons/helper/DistantNeuronRequests.h"

#include <memory>
#include <utility>
#include <vector>

class NeuronsExtraInfo;
template <typename T>
class OctreeImplementation;
template <typename T>
class OctreeNode;

/**
 * This class represents the implementation and adaptation of the Barnes�Hut algorithm. The parameters can be set on the fly.
 * It is strongly tied to Octree, and performs MPI communication
 */
class BarnesHutLocationAware : public ForwardAlgorithm<DistantNeuronRequest, DistantNeuronResponse, BarnesHutCell> {
public:
    using AdditionalCellAttributes = BarnesHutCell;
    using position_type = typename RelearnTypes::position_type;
    using counter_type = typename RelearnTypes::counter_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit BarnesHutLocationAware(const std::shared_ptr<OctreeImplementation<BarnesHutCell>>& octree)
        : ForwardAlgorithm(octree) { }

    /**
     * @brief Sets acceptance criterion for cells in the tree
     * @param acceptance_criterion The acceptance criterion, > 0.0
     * @exception Throws a RelearnException if acceptance_criterion <= 0.0
     */
    void set_acceptance_criterion(double acceptance_criterion);

    /**
     * @brief Returns the currently used acceptance criterion
     * @return The currently used acceptance criterion
     */
    [[nodiscard]] double get_acceptance_criterion() const noexcept {
        return acceptance_criterion;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(ForwardAlgorithm<DistantNeuronRequest, DistantNeuronResponse, BarnesHutCell>);
        footprint->emplace("BarnesHutLocationAware", my_footprint);

        ForwardAlgorithm<DistantNeuronRequest, DistantNeuronResponse, BarnesHutCell>::record_memory_footprint(footprint);
    }

protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    [[nodiscard]] CommunicationMap<DistantNeuronRequest> find_target_neurons(number_neurons_type number_neurons) override;

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] std::pair<CommunicationMap<DistantNeuronResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>>
    process_requests(const CommunicationMap<DistantNeuronRequest>& neuron_requests) override;

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] PlasticDistantOutSynapses process_responses(const CommunicationMap<DistantNeuronRequest>& neuron_requests,
        const CommunicationMap<DistantNeuronResponse>& neuron_responses) override;

private:
    double acceptance_criterion{ Constants::bh_default_theta };
};
