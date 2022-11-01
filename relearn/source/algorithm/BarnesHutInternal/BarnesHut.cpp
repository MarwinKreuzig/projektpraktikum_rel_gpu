/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHut.h"

#include "algorithm/Connector.h"
#include "neurons/NeuronsExtraInfo.h"
#include "structure/NodeCache.h"
#include "structure/OctreeNode.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <algorithm>

CommunicationMap<SynapseCreationRequest> BarnesHut::find_target_neurons(const number_neurons_type number_neurons, const std::vector<UpdateStatus>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    const auto number_ranks = MPIWrapper::get_num_ranks();

    const auto size_hint = std::min(number_neurons, number_neurons_type(number_ranks));
    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks, size_hint);

    auto* const root = get_octree_root();

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(root, number_neurons, extra_infos, disable_flags, synapse_creation_requests_outgoing)
    for (auto neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const NeuronID id{ neuron_id };

        const auto number_vacant_axons = axons->get_free_elements(id);
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto& axon_position = extra_infos->get_position(id);
        const auto dendrite_type_needed = axons->get_signal_type(id);

        const auto& requests = BarnesHutBase::find_target_neurons(id, axon_position, number_vacant_axons, root, ElementType::Dendrite, dendrite_type_needed);
        for (const auto& [target_rank, creation_request] : requests) {
#pragma omp critical(BHrequests)
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache<BarnesHutCell>::clear();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

 std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<LocalSynapses, DistantInSynapses>>
BarnesHut::process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests) {
    return ForwardConnector::process_requests(creation_requests, excitatory_dendrites, inhibitory_dendrites);
}

 DistantOutSynapses BarnesHut::process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
    const CommunicationMap<SynapseCreationResponse>& creation_responses) {
    return ForwardConnector::process_responses(creation_requests, creation_responses, axons);
}
