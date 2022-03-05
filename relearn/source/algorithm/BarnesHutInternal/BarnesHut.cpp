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

void BarnesHut::update_leaf_nodes(const std::vector<UpdateStatus>& disable_flags) {
    RelearnException::check(global_tree != nullptr, "BarnesHut::update_leaf_nodes: global_tree was nullptr");

    const auto& leaf_nodes = global_tree->get_leaf_nodes();
    const auto num_leaf_nodes = leaf_nodes.size();
    const auto num_disable_flags = disable_flags.size();

    RelearnException::check(num_leaf_nodes == num_disable_flags, "BarnesHut::update_leaf_nodes: The vectors were of different sizes");

    using counter_type = BarnesHutCell::counter_type;

    for (const auto neuron_id : NeuronID::range(num_leaf_nodes)) {
        const auto local_neuron_id = neuron_id.get_local_id();

        auto* node = leaf_nodes[local_neuron_id];
        RelearnException::check(node != nullptr, "BarnesHut::update_leaf_nodes: node was nullptr: {}", neuron_id);

        if (disable_flags[local_neuron_id] == UpdateStatus::Disabled) {
            node->set_cell_number_dendrites(0, 0);
            continue;
        }

        const auto& cell = node->get_cell();
        const auto other_neuron_id = cell.get_neuron_id();

        RelearnException::check(neuron_id == other_neuron_id, "BarnesHut::update_leaf_nodes: The nodes are not in order");

        const auto& [cell_xyz_min, cell_xyz_max] = cell.get_size();
        const auto& opt_excitatory_position = cell.get_excitatory_dendrites_position();
        const auto& opt_inhibitory_position = cell.get_inhibitory_dendrites_position();

        RelearnException::check(opt_excitatory_position.has_value(), "BarnesHut::update_leaf_nodes: Neuron {} does not have an excitatory position", neuron_id);
        RelearnException::check(opt_inhibitory_position.has_value(), "BarnesHut::update_leaf_nodes: Neuron {} does not have an inhibitory position", neuron_id);

        const auto& excitatory_position = opt_excitatory_position.value();
        const auto& inhibitory_position = opt_inhibitory_position.value();

        const auto excitatory_position_in_box = excitatory_position.check_in_box(cell_xyz_min, cell_xyz_max);
        const auto inhibitory_position_in_box = inhibitory_position.check_in_box(cell_xyz_min, cell_xyz_max);

        RelearnException::check(excitatory_position_in_box, "BarnesHut::update_leaf_nodes: Excitatory position ({}) is not in cell: [({}), ({})]", excitatory_position, cell_xyz_min, cell_xyz_max);
        RelearnException::check(inhibitory_position_in_box, "BarnesHut::update_leaf_nodes: Inhibitory position ({}) is not in cell: [({}), ({})]", inhibitory_position, cell_xyz_min, cell_xyz_max);

        const auto number_vacant_dendrites_excitatory = excitatory_dendrites->get_free_elements(neuron_id);
        const auto number_vacant_dendrites_inhibitory = inhibitory_dendrites->get_free_elements(neuron_id);

        node->set_cell_number_dendrites(number_vacant_dendrites_excitatory, number_vacant_dendrites_inhibitory);
    }
}

CommunicationMap<SynapseCreationRequest> BarnesHut::find_target_neurons(const size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    const auto number_ranks = MPIWrapper::get_num_ranks();

    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks);

    const auto root = global_tree->get_root();
    const auto sigma = get_probabilty_parameter();

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(root, sigma, number_neurons, extra_infos, disable_flags, synapse_creation_requests_outgoing)
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

        const auto& requests = find_target_neurons(id, axon_position, number_vacant_axons, root, ElementType::Dendrite, dendrite_type_needed, sigma);
        for (const auto& [target_rank, creation_request] : requests) {
#pragma omp critical
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }
    }

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache::empty<BarnesHutCell>();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    return synapse_creation_requests_outgoing;
}

std::vector<std::pair<int, SynapseCreationRequest>> BarnesHut::find_target_neurons(const NeuronID& source_neuron_id, const position_type& source_position,
    const counter_type& number_vacant_elements, OctreeNode<AdditionalCellAttributes>* root, const ElementType element_type, const SignalType signal_type, const double sigma) {

    std::vector<std::pair<int, SynapseCreationRequest>> requests{};
    requests.reserve(number_vacant_elements);

    for (unsigned int j = 0; j < number_vacant_elements; j++) {
        // Find one target at the time
        std::optional<RankNeuronId> rank_neuron_id = find_target_neuron(source_neuron_id, source_position, root, element_type, signal_type, sigma);
        if (!rank_neuron_id.has_value()) {
            // If finding failed, it won't succeed in later iterations
            break;
        }

        const auto& [target_rank, target_id] = rank_neuron_id.value();
        const SynapseCreationRequest creation_request(target_id, source_neuron_id, signal_type);

        requests.emplace_back(target_rank, creation_request);
    }

    return requests;
}

std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<LocalSynapses, DistantInSynapses>>
BarnesHut::create_synapses_process_requests(size_t number_neurons, const CommunicationMap<SynapseCreationRequest>& synapse_creation_requests_incoming) {

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

        RelearnException::check(target_neuron_id.get_local_id() < number_neurons, "BarnesHut::create_synapses_process_requests: Target_neuron_id exceeds my neurons");

        const auto& dendrites = (SignalType::Inhibitory == dendrite_type_needed) ? inhibitory_dendrites : excitatory_dendrites;

        const auto weight = (SignalType::Inhibitory == dendrite_type_needed) ? -1 : 1;
        const auto number_free_elements = dendrites->get_free_elements(target_neuron_id);

        if (number_free_elements == 0) {
            // Other axons were faster and came first
            responses.set_request(source_rank, request_index, SynapseCreationResponse::Failed);
            // responses.append(source_rank, SynapseCreationResponse::Failed);
            continue;
        }

        // Increment number of connected dendrites
        dendrites->update_connected_elements(target_neuron_id, 1);

        // Set response to "connected" (success)
        // responses.append(source_rank, SynapseCreationResponse::Succeeded);
        responses.set_request(source_rank, request_index, SynapseCreationResponse::Succeeded);

        if (source_rank == my_rank) {
            local_synapses.emplace_back(target_neuron_id, source_neuron_id, weight);
            continue;
        }

        distant_synapses.emplace_back(target_neuron_id, RankNeuronId{ source_rank, source_neuron_id }, weight);
    }

    return { responses, { local_synapses, distant_synapses } };
}

DistantOutSynapses BarnesHut::create_synapses_process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests, const CommunicationMap<SynapseCreationResponse>& creation_responses) {
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
