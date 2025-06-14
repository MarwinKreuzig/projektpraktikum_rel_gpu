/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHutLocationAware.h"

#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/Connector.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/NodeCache.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"
#include "util/NeuronID.h"
#include "util/Timers.h"

#include <algorithm>

void BarnesHutLocationAware::set_acceptance_criterion(const double acceptance_criterion) {
    RelearnException::check(acceptance_criterion > 0.0, "BarnesHut::set_acceptance_criterion: acceptance_criterion was less than or equal to 0 ({})", acceptance_criterion);
    this->acceptance_criterion = acceptance_criterion;
}

CommunicationMap<DistantNeuronRequest> BarnesHutLocationAware::find_target_neurons(const number_neurons_type number_neurons) {
    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = MPIWrapper::get_number_ranks();
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto size_hint = std::min(number_neurons, number_neurons_type(number_ranks));
    CommunicationMap<DistantNeuronRequest> neuron_requests_outgoing(number_ranks, size_hint);

    auto* const root = get_octree_root();
    const auto level_of_branch_nodes = get_level_of_branch_nodes();

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(root, number_neurons, disable_flags, neuron_requests_outgoing, level_of_branch_nodes, my_rank)
    for (auto neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] != UpdateStatus::Enabled) {
            continue;
        }

        const NeuronID id(neuron_id);

        const auto number_vacant_axons = axons->get_free_elements(id);
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto& axon_position = extra_infos->get_position(id);
        const auto dendrite_type_needed = axons->get_signal_type(id);

        const auto& requests = BarnesHutBase<BarnesHutCell>::find_target_neurons_location_aware({ my_rank, id }, axon_position, number_vacant_axons,
            root, ElementType::Dendrite, dendrite_type_needed, level_of_branch_nodes, acceptance_criterion);

        for (const auto& [target_rank, creation_request] : requests) {
#pragma omp critical(BHrequests)
            neuron_requests_outgoing.append(target_rank, creation_request);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache<BarnesHutCell>::clear();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return neuron_requests_outgoing;
}

std::pair<CommunicationMap<DistantNeuronResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>>
BarnesHutLocationAware::process_requests(const CommunicationMap<DistantNeuronRequest>& neuron_requests) {
    const auto number_ranks = neuron_requests.get_number_ranks();

    const auto size_hint = neuron_requests.size();
    CommunicationMap<DistantNeuronResponse> neuron_responses(number_ranks, size_hint);

    if (neuron_requests.empty()) {
        return { neuron_responses, {} };
    }

    CommunicationMap<SynapseCreationRequest> creation_requests(number_ranks, size_hint);
    creation_requests.resize(neuron_requests.get_request_sizes());

    for (const auto& [source_rank, requests] : neuron_requests) {
        const auto num_requests = requests.size();

        // All requests from a rank
        for (auto request_index = 0; request_index < num_requests; request_index++) {
            const auto& current_request = requests[request_index];

            const auto source_neuron_id = current_request.get_source_id();
            const auto signal_type = current_request.get_signal_type();
            const auto target_neuron_type = current_request.get_target_neuron_type();

            if (target_neuron_type == DistantNeuronRequest::TargetNeuronType::Leaf) {
                const auto target_id = current_request.get_leaf_node_id();
                const NeuronID target_neuron_id{ target_id };
                creation_requests.set_request(source_rank, request_index, SynapseCreationRequest{ target_neuron_id, source_neuron_id, signal_type });
                continue;
            }

            const auto rma_offset = current_request.get_rma_offset();
            const auto idx_offset = rma_offset / sizeof(OctreeNode<BarnesHutCell>);

            OctreeNode<AdditionalCellAttributes>* chosen_target = MemoryHolder<AdditionalCellAttributes>::get_parent_from_offset(idx_offset);

            // Otherwise get target through local barnes hut
            const auto source_position = current_request.get_source_position();

            // If the local search is successful, create a SynapseCreationRequest
            if (const auto& local_search = BarnesHutBase<BarnesHutCell>::find_target_neuron({ MPIRank(request_index), source_neuron_id }, source_position, chosen_target, ElementType::Dendrite, signal_type, acceptance_criterion); local_search.has_value()) {
                const auto& [target_rank, target_neuron_id] = local_search.value();

                creation_requests.set_request(source_rank, request_index, SynapseCreationRequest{ target_neuron_id, source_neuron_id, signal_type });
            } else {
                creation_requests.set_request(source_rank, request_index, SynapseCreationRequest{ source_neuron_id, source_neuron_id, signal_type });
            }
        }
    }

    // Pass the translated requests to the forward connector
    auto [creation_responses, synapses] = ForwardConnector::process_requests(creation_requests, excitatory_dendrites, inhibitory_dendrites);

    // Translate the responses back by adding the found neuron id
    neuron_responses.resize(creation_responses.get_request_sizes());

    for (const auto& [source_rank, responses] : creation_responses) {
        const auto num_responses = responses.size();

        // All responses for a rank
        for (auto response_index = 0; response_index < num_responses; response_index++) {
            const auto [target_neuron_id, source_neuron_id, dendrite_type_needed] = creation_requests.get_request(source_rank, response_index);
            const auto response = responses[response_index];

            neuron_responses.set_request(source_rank, response_index, DistantNeuronResponse{ target_neuron_id, response });
        }
    }

    return std::make_pair(neuron_responses, synapses);
}

PlasticDistantOutSynapses BarnesHutLocationAware::process_responses(const CommunicationMap<DistantNeuronRequest>& neuron_requests,
    const CommunicationMap<DistantNeuronResponse>& neuron_responses) {

    RelearnException::check(neuron_requests.size() == neuron_responses.size(), "BarnesHutLocationAware::process_responses: Requests and Responses had different sizes");

    const auto number_ranks = neuron_requests.get_number_ranks();

    const auto size_hint = neuron_requests.size();
    CommunicationMap<SynapseCreationRequest> creation_requests(number_ranks, size_hint);
    creation_requests.resize(neuron_requests.get_request_sizes());

    CommunicationMap<SynapseCreationResponse> creation_responses(number_ranks, size_hint);
    creation_responses.resize(neuron_responses.get_request_sizes());

    for (const auto& [rank, requests] : neuron_requests) {
        const auto& responses = neuron_responses.get_requests(rank);

        for (auto index = 0; index < requests.size(); index++) {
            const auto source_neuron_id = requests[index].get_source_id();
            const auto signal_type = requests[index].get_signal_type();
            const auto target_neuron_id = responses[index].get_source_id();
            const auto creation_response = responses[index].get_creation_response();

            if (creation_response == SynapseCreationResponse::Succeeded) {
                // If the creation succeeded set the corresponding target neuron
                creation_requests.set_request(rank, index, SynapseCreationRequest{ target_neuron_id, source_neuron_id, signal_type });
            } else {
                // Otherwise set the source as the target
                creation_requests.set_request(rank, index, SynapseCreationRequest{ source_neuron_id, source_neuron_id, signal_type });
            }

            creation_responses.set_request(rank, index, creation_response);
        }
    }

    return ForwardConnector::process_responses(creation_requests, creation_responses, axons);
}
