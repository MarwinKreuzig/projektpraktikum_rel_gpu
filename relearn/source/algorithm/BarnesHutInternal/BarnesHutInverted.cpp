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

CommunicationMap<SynapseCreationRequest> BarnesHutInverted::find_target_neurons(
    const size_t number_neurons,
    const std::vector<UpdateStatus>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    const auto number_ranks = MPIWrapper::get_num_ranks();

    auto work = []() -> std::vector<std::pair<int, SynapseCreationRequest>> {
        return {};
    };

    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks);
    Timers::start(TimerRegion::FIND_TARGET_NEURONS);

    const auto root = global_tree->get_root();

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(number_neurons, extra_infos, disable_flags, synapse_creation_requests_outgoing)
    for (auto neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::DISABLED) {
            continue;
        }

        const NeuronID id{ neuron_id };

        const auto number_vacant_axons = axons->get_free_elements(id);
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto axon_type_needed = axons->get_signal_type(id);
        const auto& axon_position = extra_infos->get_position(id);

        std::vector<std::pair<int, SynapseCreationRequest>> resquests{};
        resquests.reserve(number_vacant_axons);

        // For all vacant axons of neuron "neuron_id"
        for (unsigned int j = 0; j < number_vacant_axons; j++) {
            /**
             * Find target neuron for connecting and
             * connect if target neuron has still axon available.
             *
             * The target neuron might not have any axons left
             * as other axons might already have connected to them.
             * Right now, those collisions are handled in a first-come-first-served fashion.
             */
            std::optional<RankNeuronId> rank_neuron_id = find_target_neuron(id, axon_position, root, ElementType::AXON, axon_type_needed);
            if (!rank_neuron_id.has_value()) {
                // If finding failed, it won't succeed in later iterations
                break;
            }

            const auto& [target_rank, target_id] = rank_neuron_id.value();
            const SynapseCreationRequest creation_request(target_id, id, axon_type_needed);

            resquests.emplace_back(target_rank, creation_request);
        }

        for (const auto& [target_rank, creation_request] : resquests) {
            /**
             * Append request for synapse creation to rank "target_rank"
             * Note that "target_rank" could also be my own rank.
             */
#pragma omp critical
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }
    }

    Timers::stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache::empty<BarnesHutInvertedCell>();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    return synapse_creation_requests_outgoing;
}

void BarnesHutInverted::update_leaf_nodes(const std::vector<UpdateStatus>& disable_flags) {
    RelearnException::check(global_tree != nullptr, "BarnesHutInverted::update_leaf_nodes: global_tree was nullptr");

    const auto& leaf_nodes = global_tree->get_leaf_nodes();
    const auto num_leaf_nodes = leaf_nodes.size();
    const auto num_disable_flags = disable_flags.size();

    RelearnException::check(num_leaf_nodes == num_disable_flags, "BarnesHutInverted::update_leaf_nodes: The vectors were of different sizes");

    using counter_type = BarnesHutInvertedCell::counter_type;

    for (const auto& neuron_id : NeuronID::range(num_leaf_nodes)) {
        const auto local_neuron_id = neuron_id.get_local_id();

        auto* node = leaf_nodes[local_neuron_id];
        RelearnException::check(node != nullptr, "BarnesHutInverted::update_leaf_nodes: node was nullptr: {}", neuron_id);

        if (disable_flags[local_neuron_id] == UpdateStatus::DISABLED) {
            node->set_cell_number_axons(0, 0);
            continue;
        }

        const auto& cell = node->get_cell();
        const auto other_neuron_id = cell.get_neuron_id();

        RelearnException::check(neuron_id == other_neuron_id, "BarnesHutInverted::update_leaf_nodes: The nodes are not in order");

        const auto& [cell_xyz_min, cell_xyz_max] = cell.get_size();
        const auto& opt_excitatory_position = cell.get_excitatory_axons_position();
        const auto& opt_inhibitory_position = cell.get_inhibitory_axons_position();

        RelearnException::check(opt_excitatory_position.has_value(), "BarnesHutInverted::update_leaf_nodes: Neuron {} does not have an excitatory position", neuron_id);
        RelearnException::check(opt_inhibitory_position.has_value(), "BarnesHutInverted::update_leaf_nodes: Neuron {} does not have an inhibitory position", neuron_id);

        const auto& excitatory_position = opt_excitatory_position.value();
        const auto& inhibitory_position = opt_inhibitory_position.value();

        const auto excitatory_position_in_box = excitatory_position.check_in_box(cell_xyz_min, cell_xyz_max);
        const auto inhibitory_position_in_box = inhibitory_position.check_in_box(cell_xyz_min, cell_xyz_max);

        RelearnException::check(excitatory_position_in_box, "BarnesHutInverted::update_leaf_nodes: Excitatory position ({}) is not in cell: [({}), ({})]", excitatory_position, cell_xyz_min, cell_xyz_max);
        RelearnException::check(inhibitory_position_in_box, "BarnesHutInverted::update_leaf_nodes: Inhibitory position ({}) is not in cell: [({}), ({})]", inhibitory_position, cell_xyz_min, cell_xyz_max);

        const auto number_vacant_axons_excitatory = axons->get_free_elements(neuron_id, SignalType::EXCITATORY);
        const auto number_vacant_axons_inhibitory = axons->get_free_elements(neuron_id, SignalType::INHIBITORY);

        node->set_cell_number_axons(number_vacant_axons_excitatory, number_vacant_axons_inhibitory);
    }
}
