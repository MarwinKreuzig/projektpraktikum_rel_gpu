/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Naive.h"

#include "../io/LogFiles.h"
#include "../neurons/NeuronsExtraInfo.h"
#include "../neurons/models/SynapticElements.h"
#include "../structure/NodeCache.h"
#include "../structure/Octree.h"
#include "../structure/OctreeNode.h"
#include "../util/Random.h"
#include "../util/Timers.h"

#include <algorithm>
#include <array>
#include <stack>

[[nodiscard]] std::optional<RankNeuronId> Naive::find_target_neuron(const size_t src_neuron_id, const position_type& axon_pos_xyz, const SignalType dendrite_type_needed) {
    OctreeNode<NaiveCell>* node_selected = nullptr;
    OctreeNode<NaiveCell>* root_of_subtree = global_tree->get_root();

    RelearnException::check(root_of_subtree != nullptr, "Naive::find_target_neuron: root_of_subtree was nullptr");

    while (true) {
        /**
	     * Create vector with nodes that have at least one dendrite and are
	     * precise enough given the position of an axon
	     */
        const auto& vector = get_nodes_for_interval(axon_pos_xyz, root_of_subtree, dendrite_type_needed);

        /**
		 * Assign a probability to each node in the vector.
		 * The probability for connecting to the same neuron (i.e., the axon's neuron) is set 0.
		 * Nodes with 0 probability are removed.
		 * The probabilities of all vector elements sum up to 1.
		 */
        const auto& prob = create_interval(src_neuron_id, axon_pos_xyz, dendrite_type_needed, vector);

        if (prob.empty()) {
            return {};
        }

        const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::Algorithm, 0.0, std::nextafter(1.0, Constants::eps));

        /**
         * This is done in case of rounding errors. 
         */
        auto counter = 0;
        auto sum_probabilities = 0.0;
        while (counter < prob.size()) {
            if (sum_probabilities >= random_number) {
                break;
            }

            sum_probabilities += prob[counter];
            counter++;
        }
        node_selected = vector[counter - 1ull];

        RelearnException::check(node_selected != nullptr, "Naive::find_target_neuron: node_selected was nullptr");

        /**
	     * Leave loop if no node was selected OR
	     * the selected node is leaf node, i.e., contains normal neuron.
	     *
	     * No node is selected when all nodes in the interval, created in
	     * get_nodes_for_interval(), have probability 0 to connect.
	     */
        const auto done = !node_selected->is_parent();

        if (done) {
            break;
        }

        // Update root of subtree
        root_of_subtree = node_selected;
    }

    RankNeuronId rank_neuron_id{ node_selected->get_rank(), node_selected->get_cell_neuron_id() };
    return rank_neuron_id;
}

MapSynapseCreationRequests Naive::find_target_neurons(const size_t number_neurons, const std::vector<char>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos, const std::unique_ptr<SynapticElements>& axons) {
    MapSynapseCreationRequests synapse_creation_requests_outgoing;
    Timers::start(TimerRegion::FIND_TARGET_NEURONS);

    const std::vector<double>& axons_cnts = axons->get_total_counts();
    const std::vector<unsigned int>& axons_connected_cnts = axons->get_connected_count();
    const std::vector<SignalType>& axons_signal_types = axons->get_signal_types();

    // For my neurons
    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == 0) {
            continue;
        }

        // Number of vacant axons
        const auto num_vacant_axons = static_cast<unsigned int>(axons_cnts[neuron_id]) - axons_connected_cnts[neuron_id];
        RelearnException::check(num_vacant_axons >= 0, "Naive::find_target_neurons: num vacant axons is negative: {}", num_vacant_axons);

        if (num_vacant_axons == 0) {
            continue;
        }

        // DendriteType::EXCITATORY axon
        SignalType dendrite_type_needed = SignalType::EXCITATORY;
        if (SignalType::INHIBITORY == axons_signal_types[neuron_id]) {
            // DendriteType::INHIBITORY axon
            dendrite_type_needed = SignalType::INHIBITORY;
        }

        // Position of current neuron
        const Vec3d axon_xyz_pos = extra_infos->get_position(neuron_id);

        // For all vacant axons of neuron "neuron_id"
        for (size_t j = 0; j < num_vacant_axons; j++) {
            /**
			* Find target neuron for connecting and
			* connect if target neuron has still dendrite available.
			*
			* The target neuron might not have any dendrites left
			* as other axons might already have connected to them.
			* Right now, those collisions are handled in a first-come-first-served fashion.
			*/
            std::optional<RankNeuronId> rank_neuron_id = find_target_neuron(neuron_id, axon_xyz_pos, dendrite_type_needed);

            if (rank_neuron_id.has_value()) {
                RankNeuronId val = rank_neuron_id.value();
                /*
				* Append request for synapse creation to rank "target_rank"
				* Note that "target_rank" could also be my own rank.
				*/
                synapse_creation_requests_outgoing[val.get_rank()].append(neuron_id, val.get_neuron_id(), dendrite_type_needed);
            }
        } /* all vacant axons of a neuron */
    } /* my neurons */

    Timers::stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache::empty<NaiveCell>();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

void Naive::update_leaf_nodes(const std::vector<char>& disable_flags, const std::unique_ptr<SynapticElements>& axons,
    const std::unique_ptr<SynapticElements>& excitatory_dendrites, const std::unique_ptr<SynapticElements>& inhibitory_dendrites) {

    const std::vector<double>& dendrites_excitatory_counts = excitatory_dendrites->get_total_counts();
    const std::vector<unsigned int>& dendrites_excitatory_connected_counts = excitatory_dendrites->get_connected_count();

    const std::vector<double>& dendrites_inhibitory_counts = inhibitory_dendrites->get_total_counts();
    const std::vector<unsigned int>& dendrites_inhibitory_connected_counts = inhibitory_dendrites->get_connected_count();

    RelearnException::check(global_tree != nullptr, "Naive::update_leaf_nodes: global_tree was nullptr");

    const auto& leaf_nodes = global_tree->get_leaf_nodes();
    const auto num_leaf_nodes = leaf_nodes.size();
    const auto num_disable_flags = disable_flags.size();
    const auto num_dendrites_excitatory_counts = dendrites_excitatory_counts.size();
    const auto num_dendrites_excitatory_connected_counts = dendrites_excitatory_connected_counts.size();
    const auto num_dendrites_inhibitory_counts = dendrites_inhibitory_counts.size();
    const auto num_dendrites_inhibitory_connected_counts = dendrites_inhibitory_connected_counts.size();

    const auto all_same_size = num_leaf_nodes == num_disable_flags
        && num_leaf_nodes == num_dendrites_excitatory_counts
        && num_leaf_nodes == num_dendrites_excitatory_connected_counts
        && num_leaf_nodes == num_dendrites_inhibitory_counts
        && num_leaf_nodes == num_dendrites_inhibitory_connected_counts;

    RelearnException::check(all_same_size, "Naive::update_leaf_nodes: The vectors were of different sizes");

    for (size_t neuron_id = 0; neuron_id < num_leaf_nodes; neuron_id++) {
        auto* node = leaf_nodes[neuron_id];

        RelearnException::check(node != nullptr, "Naive::update_leaf_nodes: node was nullptr: ", neuron_id);

        const size_t other_neuron_id = node->get_cell().get_neuron_id();

        RelearnException::check(neuron_id == other_neuron_id, "Naive::update_leaf_nodes: The nodes are not in order");

        if (disable_flags[neuron_id] == 0) {
            continue;
        }

        const auto number_vacant_dendrites_excitatory = static_cast<unsigned int>(dendrites_excitatory_counts[neuron_id] - dendrites_excitatory_connected_counts[neuron_id]);
        const auto number_vacant_dendrites_inhibitory = static_cast<unsigned int>(dendrites_inhibitory_counts[neuron_id] - dendrites_inhibitory_connected_counts[neuron_id]);

        node->set_cell_number_dendrites(number_vacant_dendrites_excitatory, number_vacant_dendrites_inhibitory);
    }
}

[[nodiscard]] double Naive::calc_attractiveness_to_connect(const size_t src_neuron_id, const position_type& axon_pos_xyz,
    const OctreeNode<NaiveCell>& node_with_dendrite, const SignalType dendrite_type_needed) const {

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

    const auto& target_xyz = node_with_dendrite.get_cell().get_dendrites_position_for(dendrite_type_needed);
    RelearnException::check(target_xyz.has_value(), "Naive::update_leaf_nodes: target_xyz is bad");

    const auto num_dendrites = node_with_dendrite.get_cell().get_number_dendrites_for(dendrite_type_needed);

    const auto position_diff = target_xyz.value() - axon_pos_xyz;

    //const auto eucl_length = position_diff.calculate_p_norm(2.0);
    //const auto numerator = pow(eucl_length, 2.0);
    const auto numerator = position_diff.calculate_squared_2_norm();

    // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
    const auto ret_val = (num_dendrites * exp(-numerator / (sigma * sigma)));
    return ret_val;
}

[[nodiscard]] std::vector<double> Naive::create_interval(const size_t src_neuron_id, const position_type& axon_pos_xyz,
    const SignalType dendrite_type_needed, const std::vector<OctreeNode<NaiveCell>*>& vector) const {

    if (vector.empty()) {
        return {};
    }

    double sum = 0.0;

    std::vector<double> probabilities;
    std::for_each(vector.cbegin(), vector.cend(), [&](const OctreeNode<NaiveCell>* target_node) {
        RelearnException::check(target_node != nullptr, "Naive::update_leaf_nodes: target_node was nullptr");
        const auto prob = calc_attractiveness_to_connect(src_neuron_id, axon_pos_xyz, *target_node, dendrite_type_needed);
        probabilities.push_back(prob);
        sum += prob;
    });

    /**
	 * Make sure that we don't divide by 0 in case all probabilities from above are 0.
     * There is no neuron to connect to in that case.
	 */
    if (sum == 0.0) {
        return {};
    }

    std::transform(probabilities.begin(), probabilities.end(), probabilities.begin(), [sum](double prob) { return prob / sum; });

    return probabilities;
}

[[nodiscard]] std::tuple<bool, bool> Naive::acceptance_criterion_test(const position_type& axon_pos_xyz, const OctreeNode<NaiveCell>* const node_with_dendrite,
    const SignalType dendrite_type_needed) const {

    RelearnException::check(node_with_dendrite != nullptr, "Naive::update_leaf_nodes:  node_with_dendrite was nullptr");

    const auto& cell = node_with_dendrite->get_cell();
    const auto has_vacant_dendrites = cell.get_number_dendrites_for(dendrite_type_needed) != 0;
    const auto is_parent = node_with_dendrite->is_parent();

    // Accept leaf only
    return std::make_tuple(!is_parent, has_vacant_dendrites);
}

[[nodiscard]] std::vector<OctreeNode<NaiveCell>*> Naive::get_nodes_for_interval(const position_type& axon_pos_xyz, OctreeNode<NaiveCell>* root,
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

        const auto [accept, _] = acceptance_criterion_test(axon_pos_xyz, root, dendrite_type_needed);
        if (accept) {
            return { root };
        }

        return {};
    }

    std::stack<OctreeNode<NaiveCell>*> stack{};

    const auto add_children_to_stack = [&stack](OctreeNode<NaiveCell>* node) {
        std::array<OctreeNode<NaiveCell>*, Constants::number_oct> children{ nullptr };

        // Node is owned by this rank
        if (node->is_local()) {
            // Node is owned by this rank, so the pointers are good
            children = node->get_children();
        } else {
            // Node owned by different rank, so we have do download the data to local nodes
            children = NodeCache::download_children<NaiveCell>(node);
        }

        for (auto it = children.crbegin(); it != children.crend(); ++it) {
            if (*it != nullptr) {
                stack.push(*it);
            }
        }
    };

    // The algorithm expects that root is not considered directly, rather its children
    add_children_to_stack(root);

    std::vector<OctreeNode<NaiveCell>*> nodes_to_consider{};
    nodes_to_consider.reserve(Constants::number_oct);

    while (!stack.empty()) {
        // Get top-of-stack node and remove it from stack
        auto* stack_elem = stack.top();
        stack.pop();

        /**
		 * Should node be used for probability interval?
		 *
		 * Only take those that have dendrites available
		 */
        const auto [accept, has_vacant_dendrites] = acceptance_criterion_test(axon_pos_xyz, stack_elem, dendrite_type_needed);

        if (accept) {
            // Insert node into vector
            nodes_to_consider.emplace_back(stack_elem);
            continue;
        }

        if (!has_vacant_dendrites) {
            continue;
        }

        add_children_to_stack(stack_elem);
    } // while

    return nodes_to_consider;
}
