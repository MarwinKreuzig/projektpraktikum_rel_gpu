/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHutInverted.h"

#include "algorithm/Connector.h"
#include "neurons/NeuronsExtraInfo.h"
#include "structure/NodeCache.h"
#include "structure/OctreeNode.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <algorithm>

CommunicationMap<SynapseCreationRequest> BarnesHutInverted::find_target_neurons(const size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    const auto number_ranks = MPIWrapper::get_num_ranks();

    const auto size_hint = std::min(number_neurons, size_t(number_ranks));
    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks, size_hint);

    auto* const root = get_octree_root();

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(root, number_neurons, extra_infos, disable_flags, synapse_creation_requests_outgoing)
    for (auto neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] != UpdateStatus::Enabled) {
            continue;
        }

        const NeuronID id{ neuron_id };

        const auto number_vacant_excitatory_dendrites = excitatory_dendrites->get_free_elements(id);
        const auto number_vacant_inhibitory_dendrites = inhibitory_dendrites->get_free_elements(id);

        if (number_vacant_excitatory_dendrites + number_vacant_inhibitory_dendrites == 0) {
            continue;
        }

        const auto& dendrite_position = extra_infos->get_position(id);

        const auto& excitatory_requests = BarnesHutBase::find_target_neurons(id, dendrite_position, number_vacant_excitatory_dendrites, root, ElementType::Axon, SignalType::Excitatory);
        for (const auto& [target_rank, creation_request] : excitatory_requests) {
#pragma omp critical(BHIrequests)
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }

        const auto& inhibitory_requests = BarnesHutBase::find_target_neurons(id, dendrite_position, number_vacant_inhibitory_dendrites, root, ElementType::Axon, SignalType::Inhibitory);
        for (const auto& [target_rank, creation_request] : inhibitory_requests) {
#pragma omp critical(BHIrequests)
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache<BarnesHutInvertedCell>::clear();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<LocalSynapses, DistantOutSynapses>>
BarnesHutInverted::process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests) {
    return BackwardConnector::process_requests(creation_requests, axons);
}

DistantInSynapses BarnesHutInverted::process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
    const CommunicationMap<SynapseCreationResponse>& creation_responses) {
    return BackwardConnector::process_responses(creation_requests, creation_responses, excitatory_dendrites, inhibitory_dendrites);
}
