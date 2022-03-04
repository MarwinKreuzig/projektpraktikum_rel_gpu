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

#include "structure/NodeCache.h"
#include "structure/Octree.h"
#include "structure/OctreeNode.h"
#include "util/Timers.h"

CommunicationMap<SynapseCreationRequest> FastMultipoleMethods::find_target_neurons([[maybe_unused]] size_t number_neurons, 
    const std::vector<UpdateStatus>& disable_flags, [[maybe_unused]] const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    // TODO(hannah): Account for disable flags

    const auto number_ranks = MPIWrapper::get_num_ranks();

    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks);
    Timers::start(TimerRegion::FIND_TARGET_NEURONS);

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

    Timers::stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

    // Stop Timer and make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache::empty<FastMultipoleMethodsCell>();
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

    for (const auto neuron_id : NeuronID::range(num_leaf_nodes)) {
        auto* node = leaf_nodes[neuron_id.get_local_id()];

        RelearnException::check(node != nullptr, "FastMultipoleMethods::update_leaf_nodes: node was nullptr: {}", neuron_id);

        const auto other_neuron_id = node->get_cell().get_neuron_id();

        RelearnException::check(neuron_id == other_neuron_id, "FastMultipoleMethods::update_leaf_nodes: The nodes are not in order");

        if (disable_flags[neuron_id.get_local_id()] == UpdateStatus::DISABLED) {
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

void FastMultipoleMethods::make_creation_request_for(const SignalType signal_type_needed, CommunicationMap<SynapseCreationRequest>& request) {
    std::vector<std::pair<const OctreeNode<FastMultipoleMethodsCell>*, interaction_list_type>> nodes_with_axons{};
    nodes_with_axons.reserve(200);

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
            if (type == ElementType::Axon && i->get_cell().get_number_axons_for(signal_type_needed) == 0) {
                i = nullptr;
                continue;
            }
            if (type == ElementType::Dendrite && i->get_cell().get_number_dendrites_for(signal_type_needed) == 0) {
                i = nullptr;
                continue;
            }
        }
    };

    auto process_target_node_for_leaf_node = [&request, &nodes_with_axons, signal_type_needed](const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target) {
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
                if (target_child->get_cell().get_number_dendrites_for(signal_type_needed) <= 0) {
                    target_child = nullptr;
                    continue;
                }
            }
            nodes_with_axons.emplace_back(source, new_interaction_list);
        } else {
            //current target is a leaf node
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

    auto process_target_node_for_inner_node = [&nodes_with_axons, signal_type_needed, get_rid_of_null_elements, this]
    (const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target) {
        const auto& source_children = source->get_children();

        //target is a inner node
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

            const auto count = Utilities::count_non_zero_elements(source_list);
            for (unsigned int i = 0; i < count; i++) {
                nodes_with_axons.emplace_back(Utilities::extract_element(source_list, i), target_list);
            }
        } else {
            // multiple MPI processes

            // align level of branch nodes and target nodes
            get_rid_of_null_elements(ElementType::Dendrite, root_children);

            for (const auto* current_branch_node : local_roots) {
                interaction_list_type temp_interaction_list = root_children;

                for (size_t current_target_level = 1; current_target_level < branch_level; current_target_level++) {
                    const auto& connection_probabilities = calc_attractiveness_to_connect(current_branch_node, temp_interaction_list, signal_type_needed);
                    const auto chosen_index = choose_interval(connection_probabilities);
                    const OctreeNode<FastMultipoleMethodsCell>* target = Utilities::extract_element(temp_interaction_list, chosen_index);

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
        const auto& [source_node, interaction_list] = nodes_with_axons[nodes_with_axons.size() - 1];
        nodes_with_axons.pop_back();

        //interaction list is empty
        if (Utilities::count_non_zero_elements(interaction_list) == 0) {
            continue;
        }

        RelearnException::check(source_node != nullptr, "FastMultipoleMethods::make_creation_request_for: source_node was a nullptr.");

        const auto& cell = source_node->get_cell();

        // current source node is a leaf node
        if (!source_node->is_parent()) {
            const OctreeNode<FastMultipoleMethodsCell>* target_node;

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

        //source is an inner node
        const auto& connection_probabilities = calc_attractiveness_to_connect(source_node, interaction_list, signal_type_needed);
        const auto chosen_index = choose_interval(connection_probabilities);
        const auto* target_node = Utilities::extract_element(interaction_list, chosen_index);
        process_target_node_for_inner_node(source_node, target_node);
    }
}

std::vector<double> FastMultipoleMethods::calc_attractiveness_to_connect(const OctreeNode<FastMultipoleMethodsCell>* source, const interaction_list_type& interaction_list, const SignalType signal_type_needed) {

    const auto target_list_length = Utilities::count_non_zero_elements(interaction_list);
    std::vector<double> result(target_list_length, 0.0);

    const auto sigma = get_probabilty_parameter();

    std::array<double, Constants::p3> coefficents{ 0 };
    bool init = false;

    // For every target calculate the attractiveness
    for (unsigned int i = 0; i < target_list_length; i++) {
        const auto* current_target = Utilities::extract_element(interaction_list, i);
        const auto current_calculation = check_calculation_requirements(source, current_target, signal_type_needed);

        switch (current_calculation) {
        case CalculationType::Hermite: {
            if (!init) {
                // When the Calculation Type is Hermite, initialize the coefficients once.
                coefficents = calc_hermite_coefficients(source, sigma, signal_type_needed);
                init = true;
            }
            result[i] = calc_hermite(source, current_target, coefficents, sigma, signal_type_needed);
            break;
        }

        case CalculationType::Taylor: {
            result[i] = calc_taylor(source, current_target, sigma, signal_type_needed);
            break;
        }

        case CalculationType::Direct: {
            result[i] = calc_direct_gauss(source, current_target, sigma, signal_type_needed);
            break;
        }
        }
    }

    return result;
}

CalculationType FastMultipoleMethods::check_calculation_requirements(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, SignalType signal_type_needed) {
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

double FastMultipoleMethods::calc_taylor(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, const double sigma, const SignalType signal_type_needed) {
    // Start Timer
    Timers::start(TimerRegion::CALC_TAYLOR_COEFFICIENTS);
    RelearnException::check(target->is_parent(), "FastMultipoleMethods::calc_taylor: target node was a leaf node.");
    RelearnException::check(source->is_parent(), "FastMultipoleMethods::calc_taylor: source node was a leaf node.");

    // Get the center of the target node.
    const auto& opt_target_center = target->get_cell().get_dendrites_position_for(signal_type_needed);
    RelearnException::check(opt_target_center.has_value(), "FastMultipoleMethods::calc_taylor: target node has no position.");
    const auto& target_center = opt_target_center.value();

    // Prepare the Multiindex.
    const auto& indices = Multiindex::get_indices();

    // Buffer for coefficients
    std::array<double, Constants::p3> taylor_coefficients{};

    // calculate taylor coefficients
    for (auto index = 0; index < Constants::p3; index++) {
        // NOLINTNEXTLINE
        const auto& current_index = indices[index];

        double temp = 0;
        for (auto j = 0; j < Constants::number_oct; j++) {
            const auto* source_child = source->get_child(j);
            if (source_child == nullptr) {
                continue;
            }

            const auto number_axons = source_child->get_cell().get_number_axons_for(signal_type_needed);
            if (number_axons == 0) {
                continue;
            }

            const auto& child_pos = source_child->get_cell().get_axons_position_for(signal_type_needed);
            RelearnException::check(child_pos.has_value(), "FastMultipoleMethods::calc_taylor: source child has no position.");
            const auto& temp_vec = (child_pos.value() - target_center) / sigma;
            temp += number_axons * Utilities::h_multiindex(current_index, temp_vec);
        }

        const auto factorial_multiindex = Utilities::fac_multiindex(current_index);
        const auto coefficient = temp / factorial_multiindex;

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

    double result = 0.0;
    interaction_list_type target_children{ nullptr };

    // calculate attractiveness
    const auto children_counter = Utilities::count_non_zero_elements(target_children);
    for (unsigned int j = 0; j < children_counter; j++) {
        const auto* target_child = Utilities::extract_element(target_children, j);

        const auto number_dendrites = target_child->get_cell().get_number_dendrites_for(signal_type_needed);
        if (number_dendrites == 0) {
            continue;
        }

        const auto& child_pos = target_child->get_cell().get_dendrites_position_for(signal_type_needed);
        RelearnException::check(child_pos.has_value(), "FastMultipoleMethods::calc_taylor: target child has no position.");
        const auto& temp_vec = (child_pos.value() - target_center) / sigma;

        double temp = 0.0;
        for (auto b = 0; b < Constants::p3; b++) {
            // NOLINTNEXTLINE
            temp += taylor_coefficients[b] * Utilities::pow_multiindex(temp_vec, indices[b]);
        }
        result += number_dendrites * temp;
    }

    return result;
}

std::array<double, Constants::p3> FastMultipoleMethods::calc_hermite_coefficients(const OctreeNode<FastMultipoleMethodsCell>* source, const double sigma, const SignalType signal_type_needed) {
    Timers::start(TimerRegion::CALC_HERMITE_COEFFICIENTS);
    RelearnException::check(source->is_parent(), "FastMultipoleMethods::calc hermite_coefficients: source node was a leaf node");

    const auto& indices = Multiindex::get_indices();
    std::array<double, Constants::p3> result{};

    for (auto a = 0; a < Constants::p3; a++) {
        auto temp = 0.0;
        for (auto i = 0; i < Constants::number_oct; i++) {

            const auto* child = source->get_child(i);
            if (child == nullptr) {
                continue;
            }

            const auto child_number_axons = child->get_cell().get_number_axons_for(signal_type_needed);
            if (child_number_axons == 0) {
                continue;
            }

            const auto& child_pos = child->get_cell().get_axons_position_for(signal_type_needed);
            RelearnException::check(child_pos.has_value(), "FastMultipoleMethods::calc_hermite_coefficients: source child has no valid position.");
            const auto& source_pos = source->get_cell().get_axons_position_for(signal_type_needed);
            RelearnException::check(source_pos.has_value(), "FastMultipoleMethods::calc_hermite_coefficients: source has no valid position.");
            const auto& temp_vec = (child_pos.value() - source_pos.value()) / sigma;
            temp += child_number_axons * Utilities::pow_multiindex(temp_vec, indices[a]);
        }

        const auto hermite_coefficient = 1.0 * temp / Utilities::fac_multiindex(indices[a]);
        result[a] = hermite_coefficient;
    }
    Timers::stop_and_add(TimerRegion::CALC_HERMITE_COEFFICIENTS);

    return result;
}

double FastMultipoleMethods::calc_hermite(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, const std::array<double, Constants::p3>& coefficients_buffer, const double sigma, const SignalType signal_type_needed) {
    RelearnException::check(target->is_parent(), "FastMultipoleMethods::calc hermite: target node was a leaf node");

    const auto& opt_source_center = source->get_cell().get_axons_position_for(signal_type_needed);
    RelearnException::check(opt_source_center.has_value(), "FastMultipoleMethods::calc_hermite: source node has no axon position.");
    const auto& source_center = opt_source_center.value();

    constexpr const auto indices = Multiindex::get_indices();
    constexpr const auto number_coefficients = Multiindex::get_number_of_indices();

    double result = 0.0;

    interaction_list_type target_children = Utilities::get_children_to_interaction_list(target);
    unsigned int children_count = Utilities::count_non_zero_elements(target_children);

    for (unsigned int j = 0; j < children_count; j++) {
        const auto* child_target = Utilities::extract_element(target_children, j);
        if (child_target == nullptr) {
            continue;
        }

        double temp = 0.0;

        const auto number_dendrites = child_target->get_cell().get_number_dendrites_for(signal_type_needed);
        if (number_dendrites == 0) {
            continue;
        }

        const auto& child_pos = child_target->get_cell().get_dendrites_position_for(signal_type_needed);
        RelearnException::check(child_pos.has_value(), "FastMultipoleMethods::calc_hermite: target child node has no axon position.");
        const auto& temp_vec = (child_pos.value() - source_center) / sigma;

        for (auto a = 0; a < number_coefficients; a++) {
            // NOLINTNEXTLINE
            temp += coefficients_buffer[a] * Utilities::h_multiindex(indices[a], temp_vec);
        }

        result += number_dendrites * temp;
    }
    return result;
}

unsigned int FastMultipoleMethods::Utilities::count_non_zero_elements(const interaction_list_type& arr) {
    auto non_zero_counter = 0;
    for (auto i = 0; i < Constants::number_oct; i++) {
        if (arr[i] != nullptr) {
            non_zero_counter++;
        }
    }
    return non_zero_counter;
}

const OctreeNode<FastMultipoleMethodsCell>* FastMultipoleMethods::Utilities::extract_element(const interaction_list_type& arr, unsigned int index) {
    auto non_zero_counter = 0;
    for (auto i = 0; i < Constants::number_oct; i++) {
        if (arr[i] != nullptr) {
            if (index == non_zero_counter) {
                return arr[i];
            }
            non_zero_counter++;
        }
    }
    return nullptr;
}

FastMultipoleMethods::interaction_list_type FastMultipoleMethods::Utilities::get_children_to_interaction_list(const OctreeNode<AdditionalCellAttributes>* node) {
    interaction_list_type result{ nullptr };

    const auto is_local = node->is_local();
    const auto& children = is_local ? node->get_children() : NodeCache::download_children<FastMultipoleMethodsCell>(const_cast<OctreeNode<FastMultipoleMethodsCell>*>(node));

    unsigned int i = 0;

    for (auto it = children.crbegin(); it != children.crend(); ++it) {
        if (*it != nullptr) {
            result[i] = (*it);
        }
        i++;
    }
    return result;
}

const std::vector<std::pair<FastMultipoleMethods::position_type, FastMultipoleMethods::counter_type>>
FastMultipoleMethods::Utilities::get_all_positions_for(const OctreeNode<FastMultipoleMethodsCell>* node, const ElementType type, const SignalType signal_type_needed) {

    std::vector<std::pair<position_type, counter_type>> result{};

    std::vector<const OctreeNode<FastMultipoleMethodsCell>*> stack{};
    stack.reserve(30);
    stack.emplace_back(node);

    while (!stack.empty()) {
        const auto* current_node = stack[stack.size() - 1];
        stack.pop_back();

        // node is leaf
        if (!current_node->is_parent()) {
            // Get number and position, depending on which types were chosen.
            const auto& cell = current_node->get_cell();
            auto num_of_ports = 0;
            std::optional<VirtualPlasticityElementManual::position_type> opt_position;

            if (type == ElementType::Dendrite) {
                num_of_ports = cell.get_number_dendrites_for(signal_type_needed);
                opt_position = cell.get_dendrites_position();
            } else {
                num_of_ports = cell.get_number_axons_for(signal_type_needed);
                opt_position = cell.get_axons_position();
            }

            RelearnException::check(opt_position.has_value(), "FastMultipoleMethods::Utilities::get_all_positions_for: opt_position has no value.");
            // push number and position of dendritic elements to result
            result.emplace_back(std::pair<position_type, counter_type>(opt_position.value(), num_of_ports));
            continue;
        }

        // node is inner node
        const auto is_local = current_node->is_local();
        const auto& children = is_local ? current_node->get_children() : NodeCache::download_children<AdditionalCellAttributes>(const_cast<OctreeNode<AdditionalCellAttributes>*>(current_node));

        // push children to stack
        for (auto it = children.crbegin(); it != children.crend(); ++it) {
            if (*it == nullptr) {
                continue;
            }
            auto number_syn_elemts = type == ElementType::Dendrite ? (*it)->get_cell().get_number_dendrites_for(signal_type_needed) : (*it)->get_cell().get_number_axons_for(signal_type_needed);
            if (number_syn_elemts == 0) {
                continue;
            }
            stack.emplace_back(*it);
        }
    }
    return result;
}