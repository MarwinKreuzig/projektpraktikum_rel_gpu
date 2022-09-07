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

#include "NaiveCell.h"
#include "Types.h"
#include "algorithm/Connector.h"
#include "algorithm/Internal/ExchangingAlgorithm.h"
#include "neurons/SignalType.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

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
 * It is strongly tied to Octree, and might perform MPI communication via NodeCache::download_children()
 */
class Naive : public ForwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse> {
public:
    using AdditionalCellAttributes = NaiveCell;
    using position_type = typename RelearnTypes::position_type;
    using counter_type = typename RelearnTypes::counter_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit Naive(const std::shared_ptr<OctreeImplementation<NaiveCell>>& octree)
        : global_tree(octree) {
        RelearnException::check(octree != nullptr, "Naive::Naive: octree was null");
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
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param number_neurons The number of local neurons
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<LocalSynapses, DistantInSynapses>>
    process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests) override {
        return ForwardConnector::process_requests(creation_requests, excitatory_dendrites, inhibitory_dendrites);
    }

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] DistantOutSynapses process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
        const CommunicationMap<SynapseCreationResponse>& creation_responses) override {
        return ForwardConnector::process_responses(creation_requests, creation_responses, axons);
    }

private:
    /**
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::download_children()
     * @param src_neuron_id The neuron's id that wants to connect. Is used to disallow autapses (connections to itself)
     * @param axon_position The neuron's position that wants to connect. Is used in probability computations
     * @param dendrite_type_needed The signal type that is searched.
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorihtm found a matching neuron, it's id and MPI rank are returned.
     */
    [[nodiscard]] std::optional<RankNeuronId> find_target_neuron(const NeuronID& src_neuron_id, const position_type& axon_position, SignalType dendrite_type_needed);

    [[nodiscard]] static double
    calc_attractiveness_to_connect(
        const NeuronID& src_neuron_id,
        const position_type& axon_position,
        const OctreeNode<NaiveCell>& node_with_dendrite,
        SignalType dendrite_type_needed);

    [[nodiscard]] std::vector<double> create_interval(
        const NeuronID& src_neuron_id,
        const position_type& axon_position,
        SignalType dendrite_type_needed,
        const std::vector<OctreeNode<NaiveCell>*>& vector) const;

    [[nodiscard]] static std::tuple<bool, bool> acceptance_criterion_test(
        const position_type& axon_position,
        const OctreeNode<NaiveCell>* node_with_dendrite,
        SignalType dendrite_type_needed);

    [[nodiscard]] static std::vector<OctreeNode<NaiveCell>*> get_nodes_for_interval(
        const position_type& axon_position,
        OctreeNode<NaiveCell>* root,
        SignalType dendrite_type_needed);

    std::shared_ptr<OctreeImplementation<NaiveCell>> global_tree{};
};
