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
#include "neurons/enums/ElementType.h"
#include "neurons/enums/SignalType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/DistantNeuronRequests.h"

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

class NeuronsExtraInfo;
template <typename T>
class OctreeImplementation;
template <typename T>
class OctreeNode;

/**
 * This class represents the implementation and adaptation of the Barnes–Hut algorithm. The parameters can be set on the fly.
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

protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    [[nodiscard]] CommunicationMap<DistantNeuronRequest> find_target_neurons(number_neurons_type number_neurons) override;

    /**
     * @brief Finds target neurons for a specified source neuron
     * @param source_neuron_id The source neuron's id
     * @param source_position The source neuron's position
     * @param number_vacant_elements The number of vacant elements of the source neuron
     * @param root Where the source neuron should start to search for targets. It is not const because the children might be changed if the node is
     * @param element_type The element type the source neuron searches
     * @param signal_type The signal type the source neuron searches
     * @return A vector of pairs with (a) the target mpi rank and (b) the request for that rank
     */
    [[nodiscard]] std::vector<std::tuple<MPIRank, DistantNeuronRequest>> find_target_neurons(const NeuronID& source_neuron_id, const position_type& source_position,
        const counter_type& number_vacant_elements, OctreeNode<AdditionalCellAttributes>* root, ElementType element_type, SignalType signal_type);

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] std::pair<CommunicationMap<DistantNeuronResponse>, std::pair<LocalSynapses, DistantInSynapses>>
    process_requests(const CommunicationMap<DistantNeuronRequest>& neuron_requests) override;

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] DistantOutSynapses process_responses(const CommunicationMap<DistantNeuronRequest>& neuron_requests,
        const CommunicationMap<DistantNeuronResponse>& neuron_responses) override;

private:
    double acceptance_criterion{ Constants::bh_default_theta };
};
