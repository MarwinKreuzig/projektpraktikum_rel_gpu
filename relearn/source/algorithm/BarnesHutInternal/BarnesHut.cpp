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

CommunicationMap<SynapseCreationRequest> BarnesHut::find_target_neurons(
    const size_t number_neurons,
    const std::vector<UpdateStatus>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    const auto number_ranks = MPIWrapper::get_num_ranks();

    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks);
    Timers::start(TimerRegion::FIND_TARGET_NEURONS);

    const auto root = global_tree->get_root();

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(root, number_neurons, extra_infos, disable_flags, synapse_creation_requests_outgoing)
    for (auto neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::DISABLED) {
            continue;
        }

        const NeuronID id{ neuron_id };
        const auto& axon_position = extra_infos->get_position(id);
        const auto dendrite_type_needed = axons->get_signal_type(id);

        const auto number_vacant_axons = axons->get_free_elements(id);
        if (number_vacant_axons == 0) {
            continue;
        }

        const auto& requests = BarnesHutBase::find_target_neurons(id, axon_position, number_vacant_axons, root, ElementType::DENDRITE, dendrite_type_needed);
        for (const auto& [target_rank, creation_request] : requests) {
#pragma omp critical
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }
    }

    Timers::stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache::empty<BarnesHutCell>();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    return synapse_creation_requests_outgoing;
}

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

        if (disable_flags[local_neuron_id] == UpdateStatus::DISABLED) {
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
