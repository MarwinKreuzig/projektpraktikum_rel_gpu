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
#include "FastMultipoleMethodsBase.h"

#include "algorithm/Kernel/Gaussian.h"
#include "structure/NodeCache.h"
#include "structure/Octree.h"
#include "structure/OctreeNode.h"
#include "util/Stack.h"
#include "util/Timers.h"

using AdditionalCellAttributes = FastMultipoleMethodsCell;

CommunicationMap<SynapseCreationRequest> FastMultipoleMethods::find_target_neurons([[maybe_unused]] size_t number_neurons,
    const std::vector<UpdateStatus>& disable_flags, [[maybe_unused]] const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    // TODO(hannah): Account for disable flags

    const auto number_ranks = MPIWrapper::get_num_ranks();

    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks);

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
        auto* node = leaf_nodes[neuron_id.get_local_id()];

        RelearnException::check(node != nullptr, "FastMultipoleMethods::update_leaf_nodes: node was nullptr: {}", neuron_id);

        const auto other_neuron_id = node->get_cell().get_neuron_id();

        RelearnException::check(neuron_id == other_neuron_id, "FastMultipoleMethods::update_leaf_nodes: The nodes are not in order");

        if (disable_flags[neuron_id.get_local_id()] == UpdateStatus::Disabled) {
            continue;
        }

        const auto number_vacant_dendrites_excitatory = static_cast<counter_type>(dendrites_excitatory_counts[neuron_id.get_local_id()] - dendrites_excitatory_connected_counts[neuron_id.get_local_id()]);
        const auto number_vacant_dendrites_inhibitory = static_cast<counter_type>(dendrites_inhibitory_counts[neuron_id.get_local_id()] - dendrites_inhibitory_connected_counts[neuron_id.get_local_id()]);

        node->set_cell_number_dendrites(number_vacant_dendrites_excitatory, number_vacant_dendrites_inhibitory);

        const auto signal_type = axons->get_signal_type(neuron_id);

        if (signal_type == SignalType::Excitatory) {
            const auto number_vacant_excitatory_axons = static_cast<counter_type>(axons_counts[neuron_id.get_local_id()] - axons_connected_counts[neuron_id.get_local_id()]);
            const auto number_vacant_inhibitory_axons = 0;

            node->set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        } else {
            const auto number_vacant_excitatory_axons = 0;
            const auto number_vacant_inhibitory_axons = static_cast<counter_type>(axons_counts[neuron_id.get_local_id()] - axons_connected_counts[neuron_id.get_local_id()]);

            node->set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        }
    }
}

void FastMultipoleMethods::print_calculations(OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, SignalType needed) {
    try {
        auto calc_type = check_calculation_requirements(source, target, needed);

        double direct = calc_direct_gauss(source, target, needed);
        double taylor = calc_taylor(source, target, needed);
        const auto& coefficients = calc_hermite_coefficients(source, needed);
        double hermite = calc_hermite(source, target, coefficients, needed);

        if (hermite != 0) {
            std::stringstream ss;
            ss << std::fixed;
            ss << direct << ",\t";
            ss << taylor << ",\t";
            ss << hermite << ",\t" << calc_type << '\n';

            LogFiles::write_to_file(LogFiles::EventType::Cout, true, ss.str());
        }
    } catch (...) {
    }
}

void FastMultipoleMethods::make_creation_request_for(const SignalType signal_type_needed, CommunicationMap<SynapseCreationRequest>& request) {

    Stack<node_pair> stack = init_stack(signal_type_needed);

    while (!stack.empty()) {
        unpack_node_pair(stack);
        // get node and interaction list from stack
        const auto& p = stack.pop_back();
        auto* source_node = p[0];
        auto* target_parent = p[1];

        if (source_node == nullptr) {
            continue;
        }
        if (source_node->get_cell().get_number_axons_for(signal_type_needed) == 0) {
            continue;
        }
        if (target_parent->get_cell().get_number_dendrites_for(signal_type_needed) == 0) {
            continue;
        }

        // extract target children to interaction_list if possible
        interaction_list_type interaction_list = align_interaction_list(source_node, target_parent, signal_type_needed);

        /*for (auto* target : interaction_list) {
            print_calculations(source_node, target, signal_type_needed);
        }*/

        // current source node is a leaf node
        if (!source_node->is_parent()) {
            auto const target_list = make_target_list(source_node, interaction_list, signal_type_needed);

            for (auto* target : target_list) {
                if (target->is_parent()) {
                    node_pair p = { source_node, target };
                    stack.emplace_back(p);
                    continue;
                }

                // current target is a leaf node
                const auto target_id = target->get_cell().get_neuron_id();
                const auto source_id = source_node->get_cell().get_neuron_id();

                // No autapse
                if (target_id != source_id) {
                    const auto target_rank = target->get_rank();
                    const SynapseCreationRequest creation_request(target_id, source_id, signal_type_needed);
                    request.append(target_rank, creation_request);
                }
            }
            continue;
        }

        // source is an inner node
        const auto& connection_probabilities = calc_attractiveness_to_connect(source_node, interaction_list, signal_type_needed);
        const auto chosen_index = FastMultipoleMethodsBase<AdditionalCellAttributes>::choose_interval(connection_probabilities);
        auto* target_node = FastMultipoleMethodsBase<AdditionalCellAttributes>::extract_element(interaction_list, chosen_index);
        const auto& source_children = source_node->get_children();

        // target is leaf
        if (!target_node->is_parent()) {
            make_stack_entries_for_leaf(target_node, signal_type_needed, stack, source_children);
            continue;
        }

        // target is inner node
        for (const auto& source_child_node : source_children) {
            node_pair p = { source_child_node, target_node };
            stack.emplace_back(p);
        }
    }
}

Stack<FastMultipoleMethods::node_pair> FastMultipoleMethods::init_stack(SignalType signal_type_needed) {
    Stack<node_pair> stack{ 200 };

    OctreeNode<FastMultipoleMethodsCell>* root = global_tree->get_root();
    const auto local_roots = global_tree->get_local_branch_nodes();

    if (local_roots.empty()) {
        return stack;
    }

    // init stack
    const auto& root_children = FastMultipoleMethodsBase<AdditionalCellAttributes>::get_children_to_array(root);
    auto const branch_level = global_tree->get_level_of_branch_nodes();

    if (branch_level < 1) {
        for (auto* child : root_children) {
            if (child == nullptr) {
                continue;
            }
            if (child->get_cell().get_number_axons_for(signal_type_needed) == 0) {
                continue;
            }
            node_pair p = { child, root };
            stack.emplace_back(p);
        }
        return stack;
    }

    for (auto* node : local_roots) {
        if (node == nullptr) {
            continue;
        }
        if (node->get_cell().get_number_dendrites_for(signal_type_needed) == 0) {
            continue;
        }

        const auto& children = node->get_children();
        for (auto* child : children) {
            if (child == nullptr) {
                continue;
            }
            if (child->get_cell().get_number_axons_for(signal_type_needed) == 0) {
                continue;
            }

            node_pair p = { child, node };
            stack.emplace_back(p);
        }
    }
    return stack;
}

void FastMultipoleMethods::unpack_node_pair(Stack<node_pair> stack) {
    if (Constants::unpacking == 0) {
        return;
    }

    const auto& p = stack.pop_back();
    auto* source_node = p[0];
    if (!source_node->is_parent()) {
        return;
    }
    const auto initial_level = source_node->get_level();
    Stack<node_pair> unpacking_stack{ 80 };
    unpacking_stack.emplace_back(p);

    while (!unpacking_stack.empty()) {
        const auto& p = unpacking_stack.pop_back();
        auto* source_node = p[0];
        auto* target_parent = p[1];
        if (source_node->get_level() == initial_level + Constants::unpacking || !source_node->is_parent()) {
            stack.emplace_back(p);
            continue;
        }

        const auto& source_children = source_node->get_children();
        for (auto* child : source_children) {
            node_pair p = { child, target_parent };
            unpacking_stack.emplace_back(p);
        }
    }
}

FastMultipoleMethods::interaction_list_type FastMultipoleMethods::align_interaction_list(OctreeNode<AdditionalCellAttributes>* source_node, OctreeNode<AdditionalCellAttributes>* target_parent, const SignalType signal_type) {
    interaction_list_type result{};
    result.reserve(pow(Constants::number_oct, Constants::unpacking + 1));
    if (!target_parent->is_parent()) {
        result.emplace_back(target_parent);
        return result;
    }

    Stack<OctreeNode<AdditionalCellAttributes>*> stack;
    stack.reserve(100);
    stack.emplace_back(target_parent);
    const auto desired_level = source_node->get_level();
    while (!stack.empty()) {
        auto* current_node = stack.pop_back();
        if (current_node->get_level() == desired_level || !current_node->is_parent()) {
            result.emplace_back(current_node);
            continue;
        }

        const auto& children = FastMultipoleMethodsBase<AdditionalCellAttributes>::get_children_to_array(current_node);
        for (auto* child : children) {
            if (child == nullptr) {
                continue;
            }
            if (child->get_cell().get_number_dendrites_for(signal_type) == 0) {
                continue;
            }
            
            stack.emplace_back(child);
        }
    }
    return result;
}

std::vector<OctreeNode<FastMultipoleMethods::AdditionalCellAttributes>*> FastMultipoleMethods::make_target_list(OctreeNode<FastMultipoleMethods::AdditionalCellAttributes>* source_node, FastMultipoleMethods::interaction_list_type interaction_list, const SignalType signal_type_needed) {

    auto target_num = FastMultipoleMethodsBase<AdditionalCellAttributes>::count_non_zero_elements(interaction_list);
    std::vector<OctreeNode<FastMultipoleMethods::AdditionalCellAttributes>*> target_list{};
    target_list.reserve(target_num);

    unsigned int source_number = source_node->get_cell().get_number_axons_for(signal_type_needed);
    auto connection_probabilities = calc_attractiveness_to_connect(source_node, interaction_list, signal_type_needed);
    while (source_number > 0 && target_num > 0) {
        const auto chosen_index = FastMultipoleMethodsBase<AdditionalCellAttributes>::choose_interval(connection_probabilities);
        const auto target_node = FastMultipoleMethodsBase<AdditionalCellAttributes>::extract_element(interaction_list, chosen_index);
        source_number -= target_node->get_cell().get_number_dendrites_for(signal_type_needed);
        target_num--;
        connection_probabilities[chosen_index] = 0;
        target_list.emplace_back(target_node);
    }
    return target_list;
}

void FastMultipoleMethods::make_stack_entries_for_leaf(OctreeNode<AdditionalCellAttributes>* target_node, const SignalType signal_type_needed,
    Stack<node_pair>& stack, const std::array<OctreeNode<FastMultipoleMethods::AdditionalCellAttributes>*, 8UL>& source_children) {

    unsigned int target_number = target_node->get_cell().get_number_dendrites_for(signal_type_needed);
    std::vector<double> attractiveness(Constants::number_oct, 0);

    interaction_list_type new_interaction_list{ nullptr };
    new_interaction_list[0] = target_node;

    unsigned int children_num = 0;
    for (unsigned int i = 0; i < Constants::number_oct; i++) {
        auto* const source_child = source_children[i];
        if (source_child == nullptr) {
            continue;
        }
        if (source_child->get_cell().get_number_axons_for(signal_type_needed) == 0) {
            continue;
        }
        children_num++;
        const auto& connection_probabilities = calc_attractiveness_to_connect(source_child, new_interaction_list, signal_type_needed);
        attractiveness[i] = connection_probabilities[0];
    }

    while (target_number > 0 && children_num > 0) {
        unsigned int chosen_index = FastMultipoleMethodsBase<AdditionalCellAttributes>::choose_interval(attractiveness);
        target_number -= source_children[chosen_index]->get_cell().get_number_axons_for(signal_type_needed);
        attractiveness[chosen_index] = 0;
        node_pair p = { source_children[chosen_index], target_node };
        stack.emplace_back(p);
    }
}

std::vector<double> FastMultipoleMethods::calc_attractiveness_to_connect(OctreeNode<FastMultipoleMethodsCell>* source, const interaction_list_type& interaction_list, SignalType signal_type_needed) {
    RelearnException::check(source != nullptr, "FastMultipoleMethods::calc_attractiveness_to_connect: Source was a nullptr.");

    const auto sigma = GaussianDistributionKernel::get_sigma();

    std::vector<double> hermite_coefficients{ 0.0 };
    hermite_coefficients.reserve(Constants::p3);
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
            const auto direct_attraction = calc_direct_gauss(source, current_target, signal_type_needed);
            result.emplace_back(direct_attraction);
            continue;
        }

        if (calculation_type == CalculationType::Taylor) {
            const auto taylor_attraction = calc_taylor(source, current_target, signal_type_needed);
            result.emplace_back(taylor_attraction);
        }

        if (!hermite_coefficients_init) {
            // When the Calculation Type is Hermite, initialize the coefficients once.
            hermite_coefficients = calc_hermite_coefficients(source, signal_type_needed);
            hermite_coefficients_init = true;
        }
        const auto hermite_attraction = calc_hermite(source, current_target, hermite_coefficients, signal_type_needed);

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

std::vector<double> FastMultipoleMethods::calc_taylor_coefficients(const OctreeNode<FastMultipoleMethodsCell>* source, const position_type& target_center, const SignalType& signal_type_needed) {
    Timers::start(TimerRegion::CALC_TAYLOR_COEFFICIENTS);
    const auto sigma = GaussianDistributionKernel::get_sigma();
    const auto& indices = Multiindex::get_indices();
    std::vector<double> taylor_coefficients{ 0.0 };
    taylor_coefficients.reserve(Constants::p3);
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
            child_attraction += number_axons * FastMultipoleMethodsBase<AdditionalCellAttributes>::h_multiindex(current_index, temp_vec);
        }

        const auto coefficient = child_attraction / FastMultipoleMethodsBase<AdditionalCellAttributes>::fac_multiindex(current_index);
        const auto absolute_multiindex = FastMultipoleMethodsBase<AdditionalCellAttributes>::abs_multiindex(current_index);

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

double FastMultipoleMethods::calc_taylor(const OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, SignalType signal_type_needed) {
    RelearnException::check(target->is_parent(), "FastMultipoleMethods::calc_taylor: target node was a leaf node.");
    RelearnException::check(source->is_parent(), "FastMultipoleMethods::calc_taylor: source node was a leaf node.");
    const auto sigma = GaussianDistributionKernel::get_sigma();

    // Get the center of the target node.
    const auto& opt_target_center = target->get_cell().get_dendrites_position_for(signal_type_needed);
    RelearnException::check(opt_target_center.has_value(), "FastMultipoleMethods::calc_taylor: target node has no position.");

    const auto& target_center = opt_target_center.value();

    const auto& taylor_coefficients = calc_taylor_coefficients(source, target_center, signal_type_needed);

    double result = 0.0;
    const auto& indices = Multiindex::get_indices();
    const auto& target_children = FastMultipoleMethodsBase<AdditionalCellAttributes>::get_children_to_array(target);
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
            child_attraction += taylor_coefficients[b] * FastMultipoleMethodsBase<AdditionalCellAttributes>::pow_multiindex(temp_vec, indices[b]);
        }
        result += number_dendrites * child_attraction;
    }

    return result;
}

double FastMultipoleMethods::calc_direct_gauss(OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, SignalType signal_type_needed) {
    const auto sigma = GaussianDistributionKernel::get_sigma();

    const auto& sources = FastMultipoleMethodsBase<AdditionalCellAttributes>::get_all_positions_for(source, ElementType::Axon, signal_type_needed);
    const auto& targets = FastMultipoleMethodsBase<AdditionalCellAttributes>::get_all_positions_for(target, ElementType::Dendrite, signal_type_needed);

    auto result = 0.0;

    for (const auto& [target_position, number_targets] : targets) {
        for (const auto& [source_position, number_sources] : sources) {
            const auto kernel_value = FastMultipoleMethodsBase<AdditionalCellAttributes>::kernel(target_position, source_position, sigma);
            result += kernel_value * number_sources * number_targets;
        }
    }

    return result;
}

std::vector<double> FastMultipoleMethods::calc_hermite_coefficients(const OctreeNode<FastMultipoleMethodsCell>* source, SignalType signal_type_needed) {
    RelearnException::check(source->is_parent(), "FastMultipoleMethods::calc_hermite_coefficients: source node was a leaf node");

    Timers::start(TimerRegion::CALC_HERMITE_COEFFICIENTS);

    const auto sigma = GaussianDistributionKernel::get_sigma();

    const auto& source_cell = source->get_cell();

    const auto& indices = Multiindex::get_indices();
    std::vector<double> hermite_coefficients{};
    hermite_coefficients.reserve(Constants::p3);

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
            child_attraction += child_number_axons * FastMultipoleMethodsBase<AdditionalCellAttributes>::pow_multiindex(temp_vec, indices[a]);
        }

        const auto hermite_coefficient = child_attraction / FastMultipoleMethodsBase<AdditionalCellAttributes>::fac_multiindex(indices[a]);
        hermite_coefficients[a] = hermite_coefficient;
    }

    Timers::stop_and_add(TimerRegion::CALC_HERMITE_COEFFICIENTS);

    return hermite_coefficients;
}

double
FastMultipoleMethods::calc_hermite(const OctreeNode<AdditionalCellAttributes>* source, OctreeNode<AdditionalCellAttributes>* target,
    const std::vector<double>& coefficients_buffer, const SignalType signal_type_needed) {
    const auto sigma = GaussianDistributionKernel::get_sigma();

    RelearnException::check(target->is_parent(), "FastMultipoleMethods::calc_hermite: target node was a leaf node");

    const auto& opt_source_center = source->get_cell().get_axons_position_for(signal_type_needed);
    RelearnException::check(opt_source_center.has_value(), "FastMultipoleMethods::calc_hermite: source node has no axon position.");

    const auto& source_center = opt_source_center.value();

    constexpr const auto indices = Multiindex::get_indices();
    constexpr const auto number_coefficients = Multiindex::get_number_of_indices();

    double total_attraction = 0.0;

    const auto& interaction_list = FastMultipoleMethodsBase<AdditionalCellAttributes>::get_children_to_array(target);
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
            child_attraction += coefficients_buffer[a] * FastMultipoleMethodsBase<AdditionalCellAttributes>::h_multiindex(indices[a], temp_vec);
        }

        total_attraction += number_dendrites * child_attraction;
    }

    return total_attraction;
}
