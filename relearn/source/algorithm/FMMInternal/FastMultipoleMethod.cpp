/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FastMultipoleMethod.h"

#include "algorithm/Connector.h"
#include "algorithm/FMMInternal/FastMultipoleMethodBase.h"
#include "structure/NodeCache.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

CommunicationMap<SynapseCreationRequest> FastMultipoleMethod::find_target_neurons(const number_neurons_type number_neurons) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = MPIWrapper::get_num_ranks();

    const auto size_hint = std::min(number_neurons_type(number_ranks), number_neurons);
    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks, size_hint);

    auto* root = get_octree_root();
    RelearnException::check(root != nullptr, "FastMultipoleMethod::find_target_neurons: root was nullptr");

    // Get number of dendrites
    const auto total_number_dendrites_ex = root->get_cell().get_number_excitatory_dendrites();
    const auto total_number_dendrites_in = root->get_cell().get_number_inhibitory_dendrites();

    const auto& local_branch_nodes = get_octree()->get_local_branch_nodes();
    const auto branch_level = get_level_of_branch_nodes();

    if (total_number_dendrites_ex > 0) {
        FastMultipoleMethodBase::make_creation_request_for(root, local_branch_nodes, branch_level, ElementType::Axon, SignalType::Excitatory, synapse_creation_requests_outgoing);
    }
    if (total_number_dendrites_in > 0) {
        FastMultipoleMethodBase::make_creation_request_for(root, local_branch_nodes, branch_level, ElementType::Axon, SignalType::Inhibitory, synapse_creation_requests_outgoing);
    }

    // Stop Timer and make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache<FastMultipoleMethodCell>::clear();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>> FastMultipoleMethod::process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests) {
    return ForwardConnector::process_requests(creation_requests, excitatory_dendrites, inhibitory_dendrites);
}

PlasticDistantOutSynapses FastMultipoleMethod::process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
    const CommunicationMap<SynapseCreationResponse>& creation_responses) {
    return ForwardConnector::process_responses(creation_requests, creation_responses, axons);
}
