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

[[nodiscard]] std::optional<RankNeuronId> BarnesHut::find_target_neuron(const NeuronID& src_neuron_id, const position_type& axon_position, const SignalType dendrite_type_needed) {
    OctreeNode<BarnesHutCell>* node_selected = nullptr;
    OctreeNode<BarnesHutCell>* root_of_subtree = global_tree->get_root();

    RelearnException::check(root_of_subtree != nullptr, "BarnesHut::find_target_neuron: root_of_subtree was nullptr");

    while (true) {
        /**
         * Create vector with nodes that have at least one dendrite and are
         * precise enough given the position of an axon
         */
        const auto& vector = get_nodes_for_interval(axon_position, root_of_subtree, dendrite_type_needed);

        /**
         * Assign a probability to each node in the vector.
         * The probability for connecting to the same neuron (i.e., the axon's neuron) is set 0.
         * Nodes with 0 probability are removed.
         */
        const auto& [total_prob, probability_values] = create_interval(src_neuron_id, axon_position, dendrite_type_needed, vector);

        if (probability_values.empty()) {
            return {};
        }

        const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::Algorithm, 0.0, std::nextafter(total_prob, Constants::eps));

        /**
         * This is done in case of rounding errors.
         */
        auto counter = 0;
        for (auto sum_probabilities = 0.0; counter < probability_values.size() && sum_probabilities < random_number; counter++) {
            sum_probabilities += probability_values[counter];
        }
        node_selected = vector[counter - 1ULL];

        RelearnException::check(node_selected != nullptr, "BarnesHut::find_target_neuron: node_selected was nullptr");
                
        // Leave loop if the selected node is leaf node, i.e., contains normal neuron.
        if (const auto done = !node_selected->is_parent(); done) {
            break;
        }

        // Update root of subtree, we need to choose starting from this root again
        root_of_subtree = node_selected;
    }

    return RankNeuronId{ node_selected->get_rank(), node_selected->get_cell_neuron_id() };
}

CommunicationMap<SynapseCreationRequest> BarnesHut::find_target_neurons(
    const size_t number_neurons,
    const std::vector<UpdateStatus>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    auto work = [this](const NeuronID& id, const position_type& axon_position, const SignalType dendrite_type_needed) 
        -> std::vector<std::pair<int, SynapseCreationRequest>> {
        const auto number_vacant_axons = axons->get_free_elements(id);
        if (number_vacant_axons == 0) {
            return {};
        }

        std::vector<std::pair<int, SynapseCreationRequest>> resquests{};
        resquests.reserve(number_vacant_axons);

        // For all vacant axons of neuron "neuron_id"
        for (unsigned int j = 0; j < number_vacant_axons; j++) {
            /**
             * Find target neuron for connecting and
             * connect if target neuron has still dendrite available.
             *
             * The target neuron might not have any dendrites left
             * as other axons might already have connected to them.
             * Right now, those collisions are handled in a first-come-first-served fashion.
             */
            std::optional<RankNeuronId> rank_neuron_id = find_target_neuron(id, axon_position, dendrite_type_needed);
            if (!rank_neuron_id.has_value()) {
                // If finding failed, it won't succeed in later iterations
                break;
            }

            const auto& [target_rank, target_id] = rank_neuron_id.value();
            const SynapseCreationRequest creation_request(target_id, id, dendrite_type_needed);

            resquests.emplace_back(target_rank, creation_request);
        }

        return resquests;
    };

    const auto number_ranks = MPIWrapper::get_num_ranks();

    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks);
    Timers::start(TimerRegion::FIND_TARGET_NEURONS);

    // For my neurons; OpenMP is picky when it comes to the type of loop variable, so no ranges here
#pragma omp parallel for default(none) shared(number_neurons, extra_infos, disable_flags, synapse_creation_requests_outgoing)
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


        std::vector<std::pair<int, SynapseCreationRequest>> resquests{};
        resquests.reserve(number_vacant_axons);

        // For all vacant axons of neuron "neuron_id"
        for (unsigned int j = 0; j < number_vacant_axons; j++) {
            /**
             * Find target neuron for connecting and
             * connect if target neuron has still dendrite available.
             *
             * The target neuron might not have any dendrites left
             * as other axons might already have connected to them.
             * Right now, those collisions are handled in a first-come-first-served fashion.
             */
            std::optional<RankNeuronId> rank_neuron_id = find_target_neuron(id, axon_position, dendrite_type_needed);
            if (!rank_neuron_id.has_value()) {
                // If finding failed, it won't succeed in later iterations
                break;
            }

            const auto& [target_rank, target_id] = rank_neuron_id.value();
            const SynapseCreationRequest creation_request(target_id, id, dendrite_type_needed);

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

double BarnesHut::calc_attractiveness_to_connect(const NeuronID& src_neuron_id, const position_type& axon_position,
    const OctreeNode<BarnesHutCell>& node_with_dendrite, const SignalType dendrite_type_needed) const {
    /**
     * If the axon's neuron itself is considered as target neuron, set attractiveness to 0 to avoid forming an autapse (connection to itself).
     * This can be done as the axon's neuron cells are always resolved until the normal (vs. super) axon's neuron is reached.
     * That is, the dendrites of the axon's neuron are not included in any super neuron considered.
     * However, this only works under the requirement that "acceptance_criterion" is <= 0.5.
     */
    if ((!node_with_dendrite.is_parent()) && (src_neuron_id == node_with_dendrite.get_cell_neuron_id())) {
        return 0.0;
    }

    const auto sigma = get_probabilty_parameter();
    const auto squared_sigma = sigma * sigma;

    const auto& target_xyz = node_with_dendrite.get_cell().get_dendrites_position_for(dendrite_type_needed);
    RelearnException::check(target_xyz.has_value(), "BarnesHut::update_leaf_nodes: target_xyz is bad");

    const auto num_dendrites = node_with_dendrite.get_cell().get_number_dendrites_for(dendrite_type_needed);

    const auto position_diff = target_xyz.value() - axon_position;

    const auto numerator = position_diff.calculate_squared_2_norm();
    const auto exponent = -numerator / squared_sigma;

    // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
    const auto exp_val = exp(exponent);
    const auto ret_val = num_dendrites * exp_val;

    return ret_val;
}

std::pair<double, std::vector<double>> BarnesHut::create_interval(const NeuronID& src_neuron_id, const position_type& axon_position,
    const SignalType dendrite_type_needed, const std::vector<OctreeNode<BarnesHutCell>*>& vector) const {

    if (vector.empty()) {
        return { 0.0, {} };
    }

    double sum = 0.0;

    std::vector<double> probabilities{};
    probabilities.reserve(vector.size());

    std::transform(vector.begin(), vector.cend(), std::back_inserter(probabilities), [&](const OctreeNode<BarnesHutCell>* target_node) {
        RelearnException::check(target_node != nullptr, "BarnesHut::update_leaf_nodes: target_node was nullptr");
        const auto prob = calc_attractiveness_to_connect(src_neuron_id, axon_position, *target_node, dendrite_type_needed);
        sum += prob;
        return prob;
    });

    /**
     * Make sure that we don't divide by 0 in case all probabilities from above are 0.
     * There is no neuron to connect to in that case.
     */
    if (sum == 0.0) {
        return { 0.0, {} };
    }

    return { sum, std::move(probabilities) };
}

BarnesHut::AcceptanceStatus BarnesHut::acceptance_criterion_test(const position_type& axon_position, const OctreeNode<BarnesHutCell>* const node_with_dendrite,
    const SignalType dendrite_type_needed) const {

    RelearnException::check(node_with_dendrite != nullptr, "BarnesHut::update_leaf_nodes:  node_with_dendrite was nullptr");

    const auto& cell = node_with_dendrite->get_cell();
    const auto has_vacant_dendrites = cell.get_number_dendrites_for(dendrite_type_needed) != 0;
    const auto is_parent = node_with_dendrite->is_parent();

    if (!has_vacant_dendrites) {
        return AcceptanceStatus::Discard;
    }

    /**
     * Node is leaf node, i.e., not super neuron.
     * Thus the node is precise. Accept it no matter what.
     */
    if (!is_parent) {
        return AcceptanceStatus::Accept;
    }

    // Check distance between neuron with axon and neuron with dendrite
    const auto& target_xyz = cell.get_dendrites_position_for(dendrite_type_needed);

    // NOTE: This assertion fails when considering inner nodes that don't have dendrites.
    RelearnException::check(target_xyz.has_value(), "BarnesHut::update_leaf_nodes: target_xyz was bad");

    // Calc Euclidean distance between source and target neuron
    const auto distance_vector = target_xyz.value() - axon_position;

    const auto distance = distance_vector.calculate_2_norm();

    if (distance == 0.0) {
        return AcceptanceStatus::Discard;
    }

    const auto length = cell.get_maximal_dimension_difference();

    // Original Barnes-Hut acceptance criterion
    const auto ret_val = (length / distance) < acceptance_criterion;
    
    return ret_val ? AcceptanceStatus::Accept : AcceptanceStatus::Expand;
}

std::vector<OctreeNode<BarnesHutCell>*> BarnesHut::get_nodes_for_interval(const position_type& axon_position, OctreeNode<BarnesHutCell>* root,
    const SignalType dendrite_type_needed) {
    if (root == nullptr) {
        return {};
    }

    if (root->get_cell().get_number_dendrites_for(dendrite_type_needed) == 0) {
        return {};
    }

    if (!root->is_parent()) {
        /**
         * The root node is a leaf and thus contains the target neuron.
         *
         * NOTE: Root is not intended to be a leaf but we handle this as well.
         * Without pushing root onto the stack, it would not make it into the "vector" of nodes.
         */

        const auto status = acceptance_criterion_test(axon_position, root, dendrite_type_needed);

        if (status == AcceptanceStatus::Accept) {
            return { root };
        }

        return {};
    }

    std::vector<OctreeNode<BarnesHutCell>*> vector{};
    vector.reserve(Constants::number_prealloc_space);

    const auto add_children_to_vector = [&vector](OctreeNode<BarnesHutCell>* node) {
        const auto is_local = node->is_local();
        const auto& children = is_local ? node->get_children() : NodeCache::download_children<BarnesHutCell>(node);

        for (auto* it : children) {
            if (it != nullptr) {
                vector.emplace_back(it);
            }
        }
    };

    // The algorithm expects that root is not considered directly, rather its children
    add_children_to_vector(root);

    std::vector<OctreeNode<BarnesHutCell>*> nodes_to_consider{};
    nodes_to_consider.reserve(Constants::number_prealloc_space);

    while (!vector.empty()) {
        // Get top-of-stack node and remove it
        auto* node = vector[vector.size() - 1];
        vector.pop_back();

        /**
         * Should node be used for probability interval?
         * Only take those that have dendrites available
         */
        const auto status = acceptance_criterion_test(axon_position, node, dendrite_type_needed);

        if (status == AcceptanceStatus::Discard) {
            continue;
        }

        if (status == AcceptanceStatus::Accept) {
            // Insert node into vector
            nodes_to_consider.emplace_back(node);
            continue;
        }

        // Need to expand
        add_children_to_vector(node);
    } // while

    return nodes_to_consider;
}
