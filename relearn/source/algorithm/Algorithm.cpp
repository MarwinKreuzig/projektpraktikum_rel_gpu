/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Algorithm.h"

#include "mpi/MPIWrapper.h"
#include "util/Timers.h"

std::tuple<LocalSynapses, DistantInSynapses, DistantOutSynapses> Algorithm::update_connectivity(size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    const auto synapse_creation_requests_outgoing = find_target_neurons(number_neurons, disable_flags, extra_infos);

    Timers::start(TimerRegion::CREATE_SYNAPSES);

    Timers::start(TimerRegion::CREATE_SYNAPSES_EXCHANGE_REQUESTS);
    const auto& synapse_creation_requests_incoming = MPIWrapper::exchange_requests(synapse_creation_requests_outgoing);
    Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES_EXCHANGE_REQUESTS);

    Timers::start(TimerRegion::CREATE_SYNAPSES_PROCESS_REQUESTS);
    auto [responses_outgoing, synapses] = create_synapses_process_requests(number_neurons, synapse_creation_requests_incoming);
    auto& [local_synapses, distant_in_synapses] = synapses;
    Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES_PROCESS_REQUESTS);

    Timers::start(TimerRegion::CREATE_SYNAPSES_EXCHANGE_RESPONSES);
    const auto& responses_incoming = MPIWrapper::exchange_requests(responses_outgoing);
    Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES_EXCHANGE_RESPONSES);

    Timers::start(TimerRegion::CREATE_SYNAPSES_PROCESS_RESPONSES);
    auto out_synapses = create_synapses_process_responses(synapse_creation_requests_outgoing, responses_incoming);
    Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES_PROCESS_RESPONSES);

    Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES);

    return {
        std::move(local_synapses), std::move(distant_in_synapses), std::move(out_synapses)
    };
}

std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<LocalSynapses, DistantInSynapses>>
Algorithm::create_synapses_process_requests(size_t number_neurons, const CommunicationMap<SynapseCreationRequest>& synapse_creation_requests_incoming) {

    const auto my_rank = MPIWrapper::get_my_rank();
    const auto number_ranks = MPIWrapper::get_num_ranks();

    CommunicationMap<SynapseCreationResponse> responses(number_ranks);

    if (synapse_creation_requests_incoming.empty()) {
        return { responses, {} };
    }

    responses.resize(synapse_creation_requests_incoming.get_request_sizes());

    LocalSynapses local_synapses{};
    local_synapses.reserve(number_neurons);

    DistantInSynapses distant_synapses{};
    distant_synapses.reserve(number_neurons);

    std::vector<std::pair<int, unsigned int>> indices{};
    indices.reserve(synapse_creation_requests_incoming.get_total_number_requests());

    for (const auto& [source_rank, requests] : synapse_creation_requests_incoming) {
        for (unsigned int request_index = 0; request_index < requests.size(); request_index++) {
            indices.emplace_back(source_rank, request_index);
        }
    }

    // We need to shuffle the request indices so we do not prefer those from smaller MPI ranks and lower neuron ids
    RandomHolder::shuffle(RandomHolderKey::Algorithm, indices.begin(), indices.end());

    for (const auto& [source_rank, request_index] : indices) {
        const auto& [target_neuron_id, source_neuron_id, dendrite_type_needed] = synapse_creation_requests_incoming.get_request(source_rank, request_index);

        RelearnException::check(target_neuron_id.get_local_id() < number_neurons, "Neurons::create_synapses_process_requests: Target_neuron_id exceeds my neurons");

        const auto& dendrites = (SignalType::Inhibitory == dendrite_type_needed) ? inhibitory_dendrites : excitatory_dendrites;

        const auto weight = (SignalType::Inhibitory == dendrite_type_needed) ? -1 : 1;
        const auto number_free_elements = dendrites->get_free_elements(target_neuron_id);

        if (number_free_elements == 0) {
            // Other axons were faster and came first
            responses.set_request(source_rank, request_index, SynapseCreationResponse::Failed);
            //responses.append(source_rank, SynapseCreationResponse::Failed);
            continue;
        }

        // Increment number of connected dendrites
        dendrites->update_connected_elements(target_neuron_id, 1);

        // Set response to "connected" (success)
        //responses.append(source_rank, SynapseCreationResponse::Succeeded);
        responses.set_request(source_rank, request_index, SynapseCreationResponse::Succeeded);

        if (source_rank == my_rank) {
            local_synapses.emplace_back(target_neuron_id, source_neuron_id, weight);
            continue;
        }

        distant_synapses.emplace_back(target_neuron_id, RankNeuronId{ source_rank, source_neuron_id }, weight);
    }

    return { responses, { local_synapses, distant_synapses } };
}

DistantOutSynapses Algorithm::create_synapses_process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests, const CommunicationMap<SynapseCreationResponse>& creation_responses) {
    const auto my_rank = MPIWrapper::get_my_rank();
    DistantOutSynapses synapses{};

    // Process the responses of all mpi ranks
    for (const auto& [target_rank, requests] : creation_responses) {
        const auto num_requests = requests.size();

        // All responses from a rank
        for (auto request_index = 0; request_index < num_requests; request_index++) {
            const auto connected = requests[request_index];
            if (connected == SynapseCreationResponse::Failed) {
                continue;
            }

            const auto& [target_neuron_id, source_neuron_id, dendrite_type_needed] = creation_requests.get_request(target_rank, request_index);

            // Increment number of connected axons
            axons->update_connected_elements(source_neuron_id, 1);

            if (target_rank == my_rank) {
                // I have already created the synapse in the network if the response comes from myself
                continue;
            }

            // Mark this synapse for later use (must be added to the network graph)
            const auto weight = (SignalType::Inhibitory == dendrite_type_needed) ? -1 : +1;
            synapses.emplace_back(RankNeuronId{ target_rank, target_neuron_id }, source_neuron_id, weight);
        }
    }

    return synapses;
}
