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

#include "io/LogFiles.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/models/SynapticElements.h"
#include "structure/NodeCache.h"
#include "structure/Octree.h"
#include "structure/OctreeNode.h"
#include "util/Random.h"
#include "util/Timers.h"

#include <algorithm>
#include <array>
#include <iostream>

void BarnesHutLocationAware::update_leaf_nodes(const std::vector<UpdateStatus>& disable_flags) {
    RelearnException::check(global_tree != nullptr, "BarnesHutLocationAware::update_leaf_nodes: global_tree was nullptr");

    const auto& leaf_nodes = global_tree->get_leaf_nodes();
    const auto num_leaf_nodes = leaf_nodes.size();
    const auto num_disable_flags = disable_flags.size();

    RelearnException::check(num_leaf_nodes == num_disable_flags, "BarnesHutLocationAware::update_leaf_nodes: The vectors were of different sizes");

    using counter_type = BarnesHutCell::counter_type;

    for (const auto& neuron_id : NeuronID::range(num_leaf_nodes)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        auto* node = leaf_nodes[local_neuron_id];
        RelearnException::check(node != nullptr, "BarnesHutLocationAware::update_leaf_nodes: node was nullptr: {}", neuron_id);

        if (disable_flags[local_neuron_id] == UpdateStatus::Disabled) {
            node->set_cell_number_dendrites(0, 0);
            continue;
        }

        const auto& cell = node->get_cell();
        const auto other_neuron_id = cell.get_neuron_id();

        RelearnException::check(neuron_id == other_neuron_id, "BarnesHutLocationAware::update_leaf_nodes: The nodes are not in order");

        const auto& [cell_xyz_min, cell_xyz_max] = cell.get_size();
        const auto& opt_excitatory_position = cell.get_excitatory_dendrites_position();
        const auto& opt_inhibitory_position = cell.get_inhibitory_dendrites_position();

        RelearnException::check(opt_excitatory_position.has_value(), "BarnesHutLocationAware::update_leaf_nodes: Neuron {} does not have an excitatory position", neuron_id);
        RelearnException::check(opt_inhibitory_position.has_value(), "BarnesHutLocationAware::update_leaf_nodes: Neuron {} does not have an inhibitory position", neuron_id);

        const auto& excitatory_position = opt_excitatory_position.value();
        const auto& inhibitory_position = opt_inhibitory_position.value();

        const auto excitatory_position_in_box = excitatory_position.check_in_box(cell_xyz_min, cell_xyz_max);
        const auto inhibitory_position_in_box = inhibitory_position.check_in_box(cell_xyz_min, cell_xyz_max);

        RelearnException::check(excitatory_position_in_box, "BarnesHutLocationAware::update_leaf_nodes: Excitatory position ({}) is not in cell: [({}), ({})]", excitatory_position, cell_xyz_min, cell_xyz_max);
        RelearnException::check(inhibitory_position_in_box, "BarnesHutLocationAware::update_leaf_nodes: Inhibitory position ({}) is not in cell: [({}), ({})]", inhibitory_position, cell_xyz_min, cell_xyz_max);

        const auto number_vacant_dendrites_excitatory = excitatory_dendrites->get_free_elements(neuron_id);
        const auto number_vacant_dendrites_inhibitory = inhibitory_dendrites->get_free_elements(neuron_id);

        node->set_cell_number_dendrites(number_vacant_dendrites_excitatory, number_vacant_dendrites_inhibitory);
    }
}

CommunicationMap<DistantNeuronRequest> BarnesHutLocationAware::find_target_neurons(const size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    std::vector<double> lengths{};

    const auto number_ranks = MPIWrapper::get_num_ranks();

    const auto size_hint = std::min(number_neurons, size_t(number_ranks));
    CommunicationMap<DistantNeuronRequest> neuron_requests_outgoing(number_ranks, size_hint);

    auto* const root = global_tree->get_root();

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(root, number_neurons, extra_infos, disable_flags, neuron_requests_outgoing, lengths)
    for (auto neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const NeuronID id{ neuron_id };
        const auto& axon_position = extra_infos->get_position(id);
        const auto dendrite_type_needed = axons->get_signal_type(id);

        const auto number_vacant_axons = axons->get_free_elements(id);
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto& requests = find_target_neurons(id, axon_position, number_vacant_axons, root, ElementType::Dendrite, dendrite_type_needed);
        for (const auto& [target_rank, creation_request, length] : requests) {
#pragma omp critical(BHrequests)
            neuron_requests_outgoing.append(target_rank, creation_request);
#pragma omp critical(BHlengths)
            lengths.emplace_back(length);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache<BarnesHutCell>::empty();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return neuron_requests_outgoing;
}

std::vector<std::tuple<int, DistantNeuronRequest, double>> BarnesHutLocationAware::find_target_neurons(const NeuronID& source_neuron_id, const position_type& source_position,
    const counter_type& number_vacant_elements, OctreeNode<AdditionalCellAttributes>* root, const ElementType element_type, const SignalType signal_type) {

    std::vector<std::tuple<int, DistantNeuronRequest, double>> requests{};
    requests.reserve(number_vacant_elements);

    for (unsigned int j = 0; j < number_vacant_elements; j++) {
        // Find one target at the time
        const auto& neuron_request = find_target_neuron(source_neuron_id, source_position, root, element_type, signal_type, global_tree->get_level_of_branch_nodes());
        if (!neuron_request.has_value()) {
            // If finding failed, it won't succeed in later iterations
            break;
        }

        const auto& [target_rank, request] = neuron_request.value();

        requests.emplace_back(target_rank, request, 0);
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
                chosen_target = static_cast<OctreeNode<AdditionalCellAttributes>*>(global_tree->get_branch_node_pointer(branch_node_id));
            } else {
                const auto rma_offset = current_request.get_rma_offset();
                chosen_target = MemoryHolder<AdditionalCellAttributes>::get_node_from_offset(rma_offset);
            }

            // Otherwise get target through local barnes hut
            const auto source_position = current_request.get_source_position();

            // If the local search is successful, create a SynapseCreationRequest
            if (const auto& local_search = BarnesHutBase<BarnesHutCell>::find_target_neuron(source_neuron_id, source_position, chosen_target, ElementType::Dendrite, signal_type); local_search.has_value()) {
                const auto& [targer_rank, target_neuron_id] = local_search.value();

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
