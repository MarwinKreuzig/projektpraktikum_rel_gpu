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

#include "algorithm/Connector.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/NodeCache.h"
#include "structure/OctreeNode.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <algorithm>

CommunicationMap<DistantNeuronRequest> BarnesHutLocationAware::find_target_neurons(const size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    const auto number_ranks = MPIWrapper::get_num_ranks();

    const auto size_hint = std::min(number_neurons, size_t(number_ranks));
    CommunicationMap<DistantNeuronRequest> neuron_requests_outgoing(number_ranks, size_hint);

    auto* const root = get_octree_root();

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(root, number_neurons, extra_infos, disable_flags, neuron_requests_outgoing)
    for (auto neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] != UpdateStatus::Enabled) {
            continue;
        }

        const NeuronID id{ neuron_id };

        const auto number_vacant_axons = axons->get_free_elements(id);
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto& axon_position = extra_infos->get_position(id);
        const auto dendrite_type_needed = axons->get_signal_type(id);

        const auto& requests = find_target_neurons(id, axon_position, number_vacant_axons, root, ElementType::Dendrite, dendrite_type_needed);
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

std::vector<std::tuple<int, DistantNeuronRequest>> BarnesHutLocationAware::find_target_neurons(const NeuronID& source_neuron_id, const position_type& source_position,
    const counter_type& number_vacant_elements, OctreeNode<AdditionalCellAttributes>* root, const ElementType element_type, const SignalType signal_type) {

    std::vector<std::tuple<int, DistantNeuronRequest>> requests{};
    requests.reserve(number_vacant_elements);

    const auto level_of_branch_nodes = get_level_of_branch_nodes();

    for (counter_type j = 0; j < number_vacant_elements; j++) {
        // Find one target at the time
        const auto& neuron_request = find_target_neuron(source_neuron_id, source_position, root, element_type, signal_type, level_of_branch_nodes);
        if (!neuron_request.has_value()) {
            // If finding failed, it won't succeed in later iterations
            break;
        }

        const auto& [target_rank, request] = neuron_request.value();

        requests.emplace_back(target_rank, request);
    }

    return requests;
}

[[nodiscard]] std::pair<CommunicationMap<DistantNeuronResponse>, std::pair<LocalSynapses, DistantInSynapses>>
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

            OctreeNode<AdditionalCellAttributes>* chosen_target = nullptr;

            if (target_neuron_type == DistantNeuronRequest::TargetNeuronType::BranchNode) {
                const auto branch_node_id = current_request.get_branch_node_id();
                chosen_target = get_octree()->get_branch_node_pointer(branch_node_id);
            } else {
                const auto rma_offset = current_request.get_rma_offset();
                chosen_target = MemoryHolder<AdditionalCellAttributes>::get_node_from_offset(rma_offset);
            }

            // Otherwise get target through local barnes hut
            const auto source_position = current_request.get_source_position();

            // If the local search is successful, create a SynapseCreationRequest
            if (const auto& local_search = BarnesHutBase<BarnesHutCell>::find_target_neuron(source_neuron_id, source_position, chosen_target, ElementType::Dendrite, signal_type); local_search.has_value()) {
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

DistantOutSynapses BarnesHutLocationAware::process_responses(const CommunicationMap<DistantNeuronRequest>& neuron_requests,
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
