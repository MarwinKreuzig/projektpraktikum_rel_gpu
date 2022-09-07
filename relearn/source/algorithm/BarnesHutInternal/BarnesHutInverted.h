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

#include "BarnesHutBase.h"
#include "BarnesHutInvertedCell.h"
#include "Types.h"
#include "algorithm/Connector.h"
#include "algorithm/ExchangingAlgorithm.h"
#include "mpi/CommunicationMap.h"
#include "neurons/ElementType.h"
#include "neurons/SignalType.h"
#include "neurons/UpdateStatus.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

class NeuronsExtraInfo;
template <typename T>
class OctreeImplementation;
class SynapticElements;

/**
 * This class represents the implementation and adaptation of the Barnes Hut algorithm. The parameters can be set on the fly.
 * In this instance, dendrites search for axons.
 * It is strongly tied to Octree, which might perform MPI communication via NodeCache::download_children()
 */
class BarnesHutInverted : public BarnesHutBase<BarnesHutInvertedCell>, public BackwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse> {
public:
    using AdditionalCellAttributes = BarnesHutInvertedCell;
    using position_type = typename RelearnTypes::position_type;
    using counter_type = typename RelearnTypes::counter_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit BarnesHutInverted(const std::shared_ptr<OctreeImplementation<BarnesHutInvertedCell>>& octree)
        : global_tree(octree) {
        RelearnException::check(octree != nullptr, "BarnesHutInverted::BarnesHutInverted: octree was null");
    }

    /**
     * @brief Updates all leaf nodes in the octree by the algorithm
     * @param disable_flags Flags that indicate if a neuron is disabled or enabled. If disabled, it won't be updated
     * @exception Throws a RelearnException if the number of flags is different than the number of leaf nodes, or if there is an internal error
     */
    void update_leaf_nodes(const std::vector<UpdateStatus>& disable_flags) override;

    void update_octree(const std::vector<UpdateStatus>& disable_flags) override;

protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons
     * @param number_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so, the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    [[nodiscard]] CommunicationMap<SynapseCreationRequest> find_target_neurons(size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
        const std::unique_ptr<NeuronsExtraInfo>& extra_infos) override;

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
    [[nodiscard]] std::vector<std::tuple<int, SynapseCreationRequest>> find_target_neurons(const NeuronID& source_neuron_id, const position_type& source_position, const counter_type& number_vacant_elements,
        OctreeNode<AdditionalCellAttributes>* root, ElementType element_type, SignalType signal_type);

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses from the local rank
     */
    [[nodiscard]] std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<LocalSynapses, DistantOutSynapses>>
    process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests) override {
        return BackwardConnector::process_requests(creation_requests, axons);
    }

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses to this MPI rank from other MPI ranks
     */
    [[nodiscard]] DistantInSynapses process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
        const CommunicationMap<SynapseCreationResponse>& creation_responses) override {
        return BackwardConnector::process_responses(creation_requests, creation_responses, excitatory_dendrites, inhibitory_dendrites);
    }

private:
    std::shared_ptr<OctreeImplementation<BarnesHutInvertedCell>> global_tree{};
};
