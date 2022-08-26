/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FastMultipoleMethods.h"

#include "algorithm/Kernel/Gaussian.h"
#include "structure/NodeCache.h"
#include "structure/Octree.h"
#include "structure/OctreeNode.h"
#include "util/Stack.h"
#include "util/Timers.h"

CommunicationMap<SynapseCreationRequest> FastMultipoleMethods::find_target_neurons([[maybe_unused]] size_t number_neurons,
    const std::vector<UpdateStatus>& disable_flags, [[maybe_unused]] const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    // TODO(hannah): Account for disable flags

    const auto number_ranks = MPIWrapper::get_num_ranks();

    const auto size_hint = std::min(size_t(number_ranks), number_neurons);
    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks, size_hint);

    OctreeNode<FastMultipoleMethodsCell>* root = global_tree->get_root();
    RelearnException::check(root != nullptr, "FastMultpoleMethods::find_target_neurons: root was nullptr");

    // Get number of dendrites
    const auto total_number_dendrites_ex = root->get_cell().get_number_excitatory_dendrites();
    const auto total_number_dendrites_in = root->get_cell().get_number_inhibitory_dendrites();

    if (total_number_dendrites_ex > 0) {
        make_creation_request_for(SignalType::Excitatory, synapse_creation_requests_outgoing);
    }
    if (total_number_dendrites_in > 0) {
        make_creation_request_for(SignalType::Inhibitory, synapse_creation_requests_outgoing);
    }

    // Stop Timer and make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache<FastMultipoleMethodsCell>::empty();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

void FastMultipoleMethods::update_leaf_nodes(const std::vector<UpdateStatus>& disable_flags) {

    const std::vector<double>& dendrites_excitatory_counts = excitatory_dendrites->get_grown_elements();
    const std::vector<unsigned int>& dendrites_excitatory_connected_counts = excitatory_dendrites->get_connected_elements();

    const std::vector<double>& dendrites_inhibitory_counts = inhibitory_dendrites->get_grown_elements();
    const std::vector<unsigned int>& dendrites_inhibitory_connected_counts = inhibitory_dendrites->get_connected_elements();

    const std::vector<double>& axons_counts = axons->get_grown_elements();
    const std::vector<unsigned int>& axons_connected_counts = axons->get_connected_elements();

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

    RelearnException::check(all_same_size, "FastMultipoleMethods::update_leaf_nodes: The vectors were of different sizes");

    using counter_type = FastMultipoleMethodsCell::counter_type;

    const auto& indices = Multiindex::get_indices();
    constexpr const auto num_coef = Multiindex::get_number_of_indices();

    for (const auto& neuron_id : NeuronID::range(num_leaf_nodes)) {
        auto* node = leaf_nodes[neuron_id.get_neuron_id()];

        RelearnException::check(node != nullptr, "FastMultipoleMethods::update_leaf_nodes: node was nullptr: {}", neuron_id);

        const auto other_neuron_id = node->get_cell().get_neuron_id();

        RelearnException::check(neuron_id == other_neuron_id, "FastMultipoleMethods::update_leaf_nodes: The nodes are not in order");

        if (disable_flags[neuron_id.get_neuron_id()] == UpdateStatus::Disabled) {
            continue;
        }

        const auto number_vacant_dendrites_excitatory = static_cast<counter_type>(dendrites_excitatory_counts[neuron_id.get_neuron_id()] - dendrites_excitatory_connected_counts[neuron_id.get_neuron_id()]);
        const auto number_vacant_dendrites_inhibitory = static_cast<counter_type>(dendrites_inhibitory_counts[neuron_id.get_neuron_id()] - dendrites_inhibitory_connected_counts[neuron_id.get_neuron_id()]);

        node->set_cell_number_dendrites(number_vacant_dendrites_excitatory, number_vacant_dendrites_inhibitory);

        const auto signal_type = axons->get_signal_type(neuron_id);

        if (signal_type == SignalType::Excitatory) {
            const auto number_vacant_excitatory_axons = static_cast<counter_type>(axons_counts[neuron_id.get_neuron_id()] - axons_connected_counts[neuron_id.get_neuron_id()]);
            const auto number_vacant_inhibitory_axons = 0;

            node->set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        } else {
            const auto number_vacant_excitatory_axons = 0;
            const auto number_vacant_inhibitory_axons = static_cast<counter_type>(axons_counts[neuron_id.get_neuron_id()] - axons_connected_counts[neuron_id.get_neuron_id()]);

            node->set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        }
    }
}

void FastMultipoleMethods::make_creation_request_for(const SignalType signal_type_needed, CommunicationMap<SynapseCreationRequest>& request) {
    Stack<std::pair<OctreeNode<FastMultipoleMethodsCell>*, interaction_list_type>> nodes_with_axons{ 200 };

    OctreeNode<FastMultipoleMethodsCell>* root = global_tree->get_root();
    const auto local_roots = global_tree->get_local_branch_nodes();

    if (local_roots.empty()) {
        return;
    }

    auto get_rid_of_null_elements = [signal_type_needed](const ElementType type, interaction_list_type& list) {
        for (auto& i : list) {
            if (i == nullptr) {
                continue;
            }

            if (i->get_cell().get_number_elements_for(type, signal_type_needed) == 0) {
                i = nullptr;
            }
        }
    };

    auto process_target_node_for_leaf_node = [&request, &nodes_with_axons, signal_type_needed](OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target) {
        if (target->is_parent()) {
            interaction_list_type new_interaction_list = Utilities::get_children_to_interaction_list(target);

            for (auto& target_child : new_interaction_list) {
                if (target_child == nullptr) {
                    continue;
                }
                // Since source_node is a leaf node, we have to make sure we do not connect to ourselves
                if (target_child == source) {
                    target_child = nullptr;
                    continue;
                }
                if (target_child->get_cell().get_number_dendrites_for(signal_type_needed) == 0) {
                    target_child = nullptr;
                    continue;
                }
            }
            nodes_with_axons.emplace_back(source, new_interaction_list);
        } else {
            // current target is a leaf node
            const auto target_id = target->get_cell().get_neuron_id();
            const auto source_id = source->get_cell().get_neuron_id();

            // No autapse
            if (target_id != source_id) {
                const auto target_rank = target->get_rank();
                const SynapseCreationRequest creation_request(target_id, source_id, signal_type_needed);
                request.append(target_rank, creation_request);
            }
        }
    };

    auto process_target_node_for_inner_node = [&nodes_with_axons, signal_type_needed, get_rid_of_null_elements, this](OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target) {
        const auto& source_children = source->get_children();

        // target is a inner node
        if (target->is_parent()) {
            interaction_list_type new_interaction_list = Utilities::get_children_to_interaction_list(target);
            get_rid_of_null_elements(ElementType::Dendrite, new_interaction_list);

            for (const auto& source_child_node : source_children) {
                if (source_child_node == nullptr) {
                    continue;
                }
                if (source_child_node->get_cell().get_number_axons_for(signal_type_needed) == 0) {
                    continue;
                }
                nodes_with_axons.emplace_back(source_child_node, new_interaction_list);
            }
        } else {
            // target_node is a leaf node
            std::vector<double> attractiveness(Constants::number_oct);

            interaction_list_type new_interaction_list{ nullptr };
            new_interaction_list[0] = target;

            for (auto i = 0; i < Constants::number_oct; i++) {
                OctreeNode<FastMultipoleMethodsCell>* source_child_node = source_children[i];
                if (source_child_node == nullptr) {
                    attractiveness[i] = 0;
                    continue;
                }
                if (source_child_node->get_cell().get_number_axons_for(signal_type_needed) == 0) {
                    attractiveness[i] = 0;
                    continue;
                }
                const auto& connection_probabilities = calc_attractiveness_to_connect(source, new_interaction_list, signal_type_needed);
                attractiveness[i] = connection_probabilities[0];
            }

            const auto chosen_index = choose_interval(attractiveness);
            nodes_with_axons.emplace_back(source_children[chosen_index], std::move(new_interaction_list));
        }
    };

    auto init_stack = [get_rid_of_null_elements, this, &nodes_with_axons, &root, &local_roots, signal_type_needed]() {
        interaction_list_type root_children = Utilities::get_children_to_interaction_list(root);
        auto const branch_level = global_tree->get_level_of_branch_nodes();

        if (branch_level == 0) {
            // only one MPI Process
            interaction_list_type source_list = root_children;
            interaction_list_type target_list = root_children;
            get_rid_of_null_elements(ElementType::Dendrite, target_list);
            get_rid_of_null_elements(ElementType::Axon, source_list);

            for (auto* source_child : source_list) {
                if (source_child == nullptr) {
                    continue;
                }
                nodes_with_axons.emplace_back(source_child, target_list);
            }
        } else {
            // multiple MPI processes

            // align level of branch nodes and target nodes
            get_rid_of_null_elements(ElementType::Dendrite, root_children);

            for (auto* current_branch_node : local_roots) {
                interaction_list_type temp_interaction_list = root_children;

                for (size_t current_target_level = 1; current_target_level < branch_level; current_target_level++) {
                    const auto& connection_probabilities = calc_attractiveness_to_connect(current_branch_node, temp_interaction_list, signal_type_needed);
                    const auto chosen_index = choose_interval(connection_probabilities);
                    OctreeNode<FastMultipoleMethodsCell>* target = Utilities::extract_element(temp_interaction_list, chosen_index);

                    temp_interaction_list = Utilities::get_children_to_interaction_list(target);
                    get_rid_of_null_elements(ElementType::Dendrite, temp_interaction_list);
                }

                // TODO(hannah): Was passiert hier mit der temp_interaction_list?
                nodes_with_axons.emplace_back(current_branch_node, temp_interaction_list);
            }
        }
    };

    init_stack();

    // start the calculation
    while (!nodes_with_axons.empty()) {
        // get node and interaction list from stack
        auto [source_node, interaction_list] = nodes_with_axons.pop_back();

        RelearnException::check(source_node != nullptr, "FastMultipoleMethods::make_creation_request_for: source_node was a nullptr.");

        const auto& cell = source_node->get_cell();

        // current source node is a leaf node
        if (!source_node->is_parent()) {
            OctreeNode<FastMultipoleMethodsCell>* target_node;

            const auto target_num = Utilities::count_non_zero_elements(interaction_list);
            if (target_num == 1) {
                target_node = Utilities::extract_element(interaction_list, 0);
            } else {
                const auto& connection_probabilities = calc_attractiveness_to_connect(source_node, interaction_list, signal_type_needed);
                const auto chosen_index = choose_interval(connection_probabilities);
                target_node = Utilities::extract_element(interaction_list, chosen_index);
            }

            process_target_node_for_leaf_node(source_node, target_node);
            continue;
        }

        // source is an inner node
        const auto& connection_probabilities = calc_attractiveness_to_connect(source_node, interaction_list, signal_type_needed);
        const auto chosen_index = choose_interval(connection_probabilities);
        auto* target_node = Utilities::extract_element(interaction_list, chosen_index);
        process_target_node_for_inner_node(source_node, target_node);
    }
}

std::vector<double> FastMultipoleMethods::calc_attractiveness_to_connect(OctreeNode<FastMultipoleMethodsCell>* source, const interaction_list_type& interaction_list, const SignalType signal_type_needed) {
    const auto sigma = GaussianDistributionKernel::get_sigma();

    std::array<double, Constants::p3> hermite_coefficients{ 0.0 };
    bool hermite_coefficients_init = false;

    std::vector<double> result{};
    result.reserve(interaction_list.size());

    // For every target calculate the attractiveness
    for (auto* current_target : interaction_list) {
        if (current_target == nullptr) {
            continue;
        }

        const auto calculation_type = check_calculation_requirements(source, current_target, signal_type_needed);

        if (calculation_type == CalculationType::Direct) {
            const auto direct_attraction = calc_direct_gauss(source, current_target, sigma, signal_type_needed);
            result.emplace_back(direct_attraction);
            continue;
        }

        if (calculation_type == CalculationType::Taylor) {
            const auto taylor_attraction = calc_taylor(source, current_target, sigma, signal_type_needed);
            result.emplace_back(taylor_attraction);
        }

        if (!hermite_coefficients_init) {
            // When the Calculation Type is Hermite, initialize the coefficients once.
            hermite_coefficients = calc_hermite_coefficients(source, sigma, signal_type_needed);
            hermite_coefficients_init = true;
        }
        const auto hermite_attraction = calc_hermite(source, current_target, hermite_coefficients, sigma, signal_type_needed);

        result.emplace_back(hermite_attraction);
    }

    return result;
}

CalculationType FastMultipoleMethods::check_calculation_requirements(OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, SignalType signal_type_needed) {
    if (!source->is_parent() || !target->is_parent()) {
        return CalculationType::Direct;
    }

    if (target->get_cell().get_number_dendrites_for(signal_type_needed) > Constants::max_neurons_in_target) {
        if (source->get_cell().get_number_axons_for(signal_type_needed) > Constants::max_neurons_in_source) {
            return CalculationType::Hermite;
        }

        return CalculationType::Taylor;
    }

    return CalculationType::Direct;
}

std::array<double, Constants::p3> FastMultipoleMethods::calc_taylor_coefficients(OctreeNode<FastMultipoleMethodsCell>* source, const SignalType& signal_type_needed, const position_type& target_center, const double sigma) {
    Timers::start(TimerRegion::CALC_TAYLOR_COEFFICIENTS);
    const auto& indices = Multiindex::get_indices();
    std::array<double, Constants::p3> taylor_coefficients{ 0.0 };
    // calculate taylor coefficients
    for (auto index = 0; index < Constants::p3; index++) {
        // NOLINTNEXTLINE
        const auto& current_index = indices[index];

        double child_attraction = 0;

        const auto& children = source->get_children();
        for (const auto* source_child : children) {
            if (source_child == nullptr) {
                continue;
            }

            const auto& cell = source_child->get_cell();
            const auto number_axons = cell.get_number_axons_for(signal_type_needed);
            if (number_axons == 0) {
                continue;
            }

            const auto& child_pos = cell.get_axons_position_for(signal_type_needed);
            RelearnException::check(child_pos.has_value(), "FastMultipoleMethods::calc_taylor: source child has no position.");

            const auto& temp_vec = (child_pos.value() - target_center) / sigma;
            child_attraction += number_axons * Utilities::h_multiindex(current_index, temp_vec);
        }

        const auto coefficient = child_attraction / Utilities::fac_multiindex(current_index);
        const auto absolute_multiindex = Utilities::abs_multiindex(current_index);

        if (absolute_multiindex % 2 == 0) {
            // NOLINTNEXTLINE
            taylor_coefficients[index] = coefficient;
        } else {
            // NOLINTNEXTLINE
            taylor_coefficients[index] = -coefficient;
        }
    }
    Timers::stop_and_add(TimerRegion::CALC_TAYLOR_COEFFICIENTS);

    return taylor_coefficients;
}

double FastMultipoleMethods::calc_taylor(OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, const double sigma, const SignalType signal_type_needed) {
    RelearnException::check(target->is_parent(), "FastMultipoleMethods::calc_taylor: target node was a leaf node.");
    RelearnException::check(source->is_parent(), "FastMultipoleMethods::calc_taylor: source node was a leaf node.");

    // Get the center of the target node.
    const auto& opt_target_center = target->get_cell().get_dendrites_position_for(signal_type_needed);
    RelearnException::check(opt_target_center.has_value(), "FastMultipoleMethods::calc_taylor: target node has no position.");

    const auto& target_center = opt_target_center.value();

    const auto& taylor_coefficients = calc_taylor_coefficients(source, signal_type_needed, target_center, sigma);

    double result = 0.0;
    const auto& indices = Multiindex::get_indices();
    const auto& target_children = Utilities::get_children_to_interaction_list(target);
    for (const auto* target_child : target_children) {
        if (target_child == nullptr) {
            continue;
        }

        const auto& cell = target_child->get_cell();
        const auto number_dendrites = cell.get_number_dendrites_for(signal_type_needed);
        if (number_dendrites == 0) {
            continue;
        }

        const auto& child_pos = cell.get_dendrites_position_for(signal_type_needed);
        RelearnException::check(child_pos.has_value(), "FastMultipoleMethods::calc_taylor: target child has no position.");

        const auto& temp_vec = (child_pos.value() - target_center) / sigma;

        double child_attraction = 0.0;
        for (auto b = 0; b < Constants::p3; b++) {
            // NOLINTNEXTLINE
            child_attraction += taylor_coefficients[b] * Utilities::pow_multiindex(temp_vec, indices[b]);
        }
        result += number_dendrites * child_attraction;
    }

    return result;
}

std::array<double, Constants::p3> FastMultipoleMethods::calc_hermite_coefficients(OctreeNode<FastMultipoleMethodsCell>* source, const double sigma, const SignalType signal_type_needed) {
    RelearnException::check(source->is_parent(), "FastMultipoleMethods::calc_hermite_coefficients: source node was a leaf node");

    Timers::start(TimerRegion::CALC_HERMITE_COEFFICIENTS);

    const auto& source_cell = source->get_cell();

    const auto& indices = Multiindex::get_indices();
    std::array<double, Constants::p3> hermite_coefficients{};

    for (auto a = 0; a < Constants::p3; a++) {
        auto child_attraction = 0.0;

        const auto& children = source->get_children();
        for (auto* child : children) {
            if (child == nullptr) {
                continue;
            }

            const auto& cell = child->get_cell();
            const auto child_number_axons = cell.get_number_axons_for(signal_type_needed);
            if (child_number_axons == 0) {
                continue;
            }

            const auto& child_pos = cell.get_axons_position_for(signal_type_needed);
            RelearnException::check(child_pos.has_value(), "FastMultipoleMethods::calc_hermite_coefficients: source child has no valid position.");

            const auto& source_pos = source_cell.get_axons_position_for(signal_type_needed);
            RelearnException::check(source_pos.has_value(), "FastMultipoleMethods::calc_hermite_coefficients: source has no valid position.");

            const auto& temp_vec = (child_pos.value() - source_pos.value()) / sigma;
            child_attraction += child_number_axons * Utilities::pow_multiindex(temp_vec, indices[a]);
        }

        const auto hermite_coefficient = child_attraction / Utilities::fac_multiindex(indices[a]);
        hermite_coefficients[a] = hermite_coefficient;
    }

    Timers::stop_and_add(TimerRegion::CALC_HERMITE_COEFFICIENTS);

    return hermite_coefficients;
}

double FastMultipoleMethods::calc_hermite(OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target,
    const std::array<double, Constants::p3>& coefficients_buffer, const double sigma, const SignalType signal_type_needed) {

    RelearnException::check(target->is_parent(), "FastMultipoleMethods::calc_hermite: target node was a leaf node");

    const auto& opt_source_center = source->get_cell().get_axons_position_for(signal_type_needed);
    RelearnException::check(opt_source_center.has_value(), "FastMultipoleMethods::calc_hermite: source node has no axon position.");

    const auto& source_center = opt_source_center.value();

    constexpr const auto indices = Multiindex::get_indices();
    constexpr const auto number_coefficients = Multiindex::get_number_of_indices();

    double total_attraction = 0.0;

    const auto& interaction_list = Utilities::get_children_to_interaction_list(target);
    for (const auto* child_target : interaction_list) {
        if (child_target == nullptr) {
            continue;
        }

        const auto& cell = child_target->get_cell();
        const auto number_dendrites = cell.get_number_dendrites_for(signal_type_needed);
        if (number_dendrites == 0) {
            continue;
        }

        const auto& child_pos = cell.get_dendrites_position_for(signal_type_needed);
        RelearnException::check(child_pos.has_value(), "FastMultipoleMethods::calc_hermite: target child node has no axon position.");

        const auto& temp_vec = (child_pos.value() - source_center) / sigma;

        double child_attraction = 0.0;
        for (auto a = 0; a < number_coefficients; a++) {
            // NOLINTNEXTLINE
            child_attraction += coefficients_buffer[a] * Utilities::h_multiindex(indices[a], temp_vec);
        }

        total_attraction += number_dendrites * child_attraction;
    }

    return total_attraction;
}

FastMultipoleMethods::interaction_list_type FastMultipoleMethods::Utilities::get_children_to_interaction_list(OctreeNode<AdditionalCellAttributes>* node) {
    const auto is_local = node->is_local();
    auto result = is_local ? node->get_children() : NodeCache<FastMultipoleMethodsCell>::download_children(node);

    return result;
}
