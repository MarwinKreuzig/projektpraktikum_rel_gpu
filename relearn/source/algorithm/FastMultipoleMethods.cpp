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

#include "../structure/NodeCache.h"
#include "../structure/Octree.h"
#include "../structure/OctreeNode.h"
#include "../util/Timers.h"

unsigned int FastMultipoleMethods::do_random_experiment(const OctreeNode<FastMultipoleMethodsCell>* source, const std::vector<double>& attractiveness) {
    const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::Algorithm, 0.0, std::nextafter(1.0, Constants::eps));
    const auto vec_len = attractiveness.size();
    std::vector<double> intervals(vec_len + 1);
    intervals[0] = 0;

    auto sum = 0.0;
    for (int i = 0; i < vec_len; i++) {
        sum = sum + attractiveness[i];
    }

    for (auto i = 1; i < vec_len + 1; i++) {
        intervals[i] = intervals[i - 1ULL] + (attractiveness[i - 1ULL] / sum);
    }

    unsigned int i = 0;
    while (random_number > intervals[i + 1ULL] && i <= vec_len) {
        i++;
    }

    return i;
}

std::vector<double> FastMultipoleMethods::calc_attractiveness_to_connect_FMM(
    const OctreeNode<FastMultipoleMethodsCell>* source,
    const std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct>& interaction_list,
    const SignalType dendrite_type_needed) const {
    const auto& count_non_zero_elements = [](const std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct>& arr) {
        auto non_zero_counter = 0;
        for (auto i = 0; i < Constants::number_oct; i++) {
            if (arr[i] != nullptr) {
                non_zero_counter++;
            }
        }
        return non_zero_counter;
    };

    const auto& extract_element = [](const std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct>& arr, unsigned int index) -> const OctreeNode<FastMultipoleMethodsCell>* {
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
    };

    const auto sigma = get_probabilty_parameter();

    const auto source_number_axons = source->get_cell().get_number_axons_for(dendrite_type_needed);
    const auto target_list_length = count_non_zero_elements(interaction_list);

    std::vector<double> result(target_list_length, 0.0);

    if (source_number_axons > Constants::max_neurons_in_source) {
        // There are enough axons in the source box
        std::array<double, Constants::p3> coefficents{};
        calc_hermite_coefficients(source, coefficents, sigma, dendrite_type_needed);
        for (auto i = 0; i < target_list_length; i++) {
            const auto* current_target = extract_element(interaction_list, i);
            result[i] = calc_hermite(source, current_target, coefficents, sigma, dendrite_type_needed);
        }

        return result;
    }

    // There are not enough axons in the source box
    for (auto i = 0; i < target_list_length; i++) {
        const auto* current_target = extract_element(interaction_list, i);
        const auto target_number_dendrites = current_target->get_cell().get_number_dendrites_for(dendrite_type_needed);

        if (target_number_dendrites <= Constants::max_neurons_in_target) {
            // There are not enough dendrites in the target box

            const auto& target_neuron_positions = current_target->get_all_dendrite_positions_for(dendrite_type_needed);
            const auto& source_neuron_positions = source->get_all_axon_positions_for(dendrite_type_needed);

            result[i] = calc_direct_gauss(source_neuron_positions, target_neuron_positions, sigma);
        } else {
            // There are enough dendrites in the target box
            result[i] = calc_taylor_expansion(source, current_target, sigma, dendrite_type_needed);
        }
    }

    return result;
}

void FastMultipoleMethods::make_creation_request_for(SignalType needed, MapSynapseCreationRequests& request,
    std::stack<std::pair<OctreeNode<FastMultipoleMethodsCell>*, std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct>>>& nodes_with_axons) {

    const auto& count_non_zero_elements = [](const std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct>& arr) {
        auto non_zero_counter = 0;
        for (auto i = 0; i < Constants::number_oct; i++) {
            if (arr[i] != nullptr) {
                non_zero_counter++;
            }
        }
        return non_zero_counter;
    };

    const auto& extract_element = [](const std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct>& arr, unsigned int index) -> const OctreeNode<FastMultipoleMethodsCell>* {
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
    };

    while (!nodes_with_axons.empty()) {
        std::pair<OctreeNode<FastMultipoleMethodsCell>*, std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct>> pair = nodes_with_axons.top();
        nodes_with_axons.pop();

        OctreeNode<FastMultipoleMethodsCell>* source_node = std::get<0>(pair);
        std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct> interaction_list = std::get<1>(pair);

        const auto& cell = source_node->get_cell();

        /*
        - check if node is single neuron
        - set interaction list of current node when level 3
        - calculate atractiveness
        - do random experiment
        - set interaction list
        - push source_children to stack
         */

        // node is a leaf
        if (!source_node->is_parent()) {
            const auto source_id = cell.get_neuron_id();

            const OctreeNode<FastMultipoleMethodsCell>* target_node;

            const auto target_num = count_non_zero_elements(interaction_list);
            if (target_num == 1) {
                target_node = extract_element(interaction_list, 0);
            } else {
                const auto& connection_probabilities = calc_attractiveness_to_connect_FMM(source_node, interaction_list, needed);
                const auto chosen_index = do_random_experiment(source_node, connection_probabilities);
                target_node = extract_element(interaction_list, chosen_index);
            }

            if (target_node->is_parent()) {
                std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct> new_interaction_list{ nullptr };
                auto counter = 0;

                for (auto* target_child : target_node->get_children()) {
                    if (target_child == nullptr) {
                        continue;
                    }

                    // Since source_node is a leaf node, we have to make sure we do not connect to ourselves
                    if (target_child != source_node && target_child->get_cell().get_number_dendrites_for(needed) > 0) {
                        new_interaction_list[counter] = target_child;
                        counter++;
                    }
                }

                nodes_with_axons.emplace(source_node, new_interaction_list);
            } else {
                const auto target_id = target_node->get_cell().get_neuron_id();
                if (target_id != source_id) {
                    // No autapse
                    request[0].append(source_id, target_id, needed);
                }
            }
            continue;
        }

        if (count_non_zero_elements(interaction_list) == 0) {
            continue;
        }

        const auto& connection_probabilities = calc_attractiveness_to_connect_FMM(source_node, interaction_list, needed);
        const auto chosen_index = do_random_experiment(source_node, connection_probabilities);
        const auto* target_node = extract_element(interaction_list, chosen_index);

        const auto& source_children = source_node->get_children();

        if (target_node->is_parent()) {
            for (auto* source_child_node : source_children) {
                if (source_child_node == nullptr) {
                    continue;
                }

                if (source_child_node->get_cell().get_number_axons_for(needed) == 0) {
                    continue;
                }

                std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct> new_interaction_list{ nullptr };
                auto counter = 0;

                for (auto* target_child_node : target_node->get_children()) {
                    if (target_child_node == nullptr) {
                        continue;
                    }

                    if (target_child_node->get_cell().get_number_dendrites_for(needed) == 0) {
                        continue;
                    }

                    new_interaction_list[counter] = target_child_node;
                    counter++;
                }

                nodes_with_axons.emplace(source_child_node, new_interaction_list);
            }

            continue;
        }

        // source_node is a parent, but target_node is a leaf node

        std::vector<double> attractiveness{};
        std::vector<double> index{};

        for (auto i = 0; i < Constants::number_oct; i++) {
            OctreeNode<FastMultipoleMethodsCell>* source_child_node = source_children[i];
            if (source_child_node == nullptr) {
                continue;
            }

            if (source_child_node->get_cell().get_number_axons_for(needed) == 0) {
                continue;
            }

            std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct> new_interaction_list{ nullptr };
            new_interaction_list[0] = target_node;

            nodes_with_axons.emplace(source_child_node, new_interaction_list);
        }
    }
}

MapSynapseCreationRequests FastMultipoleMethods::find_target_neurons(size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos, const std::unique_ptr<SynapticElements>& axons) {

    MapSynapseCreationRequests synapse_creation_requests_outgoing;
    Timers::start(TimerRegion::FIND_TARGET_NEURONS);

    std::vector<OctreeNode<FastMultipoleMethodsCell>*> nodes_with_excitatory_dendrites{};
    std::vector<OctreeNode<FastMultipoleMethodsCell>*> nodes_with_inhibitory_dendrites{};

    std::stack<std::pair<OctreeNode<FastMultipoleMethodsCell>*, std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct>>> nodes_with_excitatory_axons{};
    std::stack<std::pair<OctreeNode<FastMultipoleMethodsCell>*, std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct>>> nodes_with_inhibitory_axons{};

    OctreeNode<FastMultipoleMethodsCell>* root = global_tree->get_root();
    const auto& children = root->get_children();

    for (auto* current_node : children) {
        const auto& cell = current_node->get_cell();
        if (cell.get_number_excitatory_dendrites() > 0) {
            nodes_with_excitatory_dendrites.push_back(current_node);
        }

        if (cell.get_number_inhibitory_dendrites() > 0) {
            nodes_with_inhibitory_dendrites.push_back(current_node);
        }
    }

    for (auto* current_node : children) {
        const auto& cell = current_node->get_cell();

        if (cell.get_number_excitatory_axons() > 0) {
            std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct> interaction_list{ nullptr };
            for (auto i = 0; i < nodes_with_excitatory_dendrites.size(); i++) {
                interaction_list[i] = nodes_with_excitatory_dendrites[i];
            }

            nodes_with_excitatory_axons.emplace(current_node, interaction_list);
        }

        if (cell.get_number_inhibitory_axons() > 0) {
            std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct> interaction_list{ nullptr };
            for (auto i = 0; i < nodes_with_inhibitory_dendrites.size(); i++) {
                interaction_list[i] = nodes_with_inhibitory_dendrites[i];
            }

            nodes_with_inhibitory_axons.emplace(current_node, interaction_list);
        }
    }

    if (!nodes_with_excitatory_axons.empty() && !nodes_with_excitatory_dendrites.empty()) {
        make_creation_request_for(SignalType::EXCITATORY, synapse_creation_requests_outgoing, nodes_with_excitatory_axons);
    }
    if (!nodes_with_inhibitory_axons.empty() && !nodes_with_inhibitory_dendrites.empty()) {
        make_creation_request_for(SignalType::INHIBITORY, synapse_creation_requests_outgoing, nodes_with_inhibitory_axons);
    }

    Timers::stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

    // Make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    NodeCache::empty<FastMultipoleMethodsCell>();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

void FastMultipoleMethods::update_leaf_nodes(const std::vector<UpdateStatus>& disable_flags, const std::unique_ptr<SynapticElements>& axons,
    const std::unique_ptr<SynapticElements>& excitatory_dendrites, const std::unique_ptr<SynapticElements>& inhibitory_dendrites) {

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
    const auto num_coef = Multiindex::get_number_of_indices();

    for (size_t neuron_id = 0; neuron_id < num_leaf_nodes; neuron_id++) {
        auto* node = leaf_nodes[neuron_id];
        const auto id = NeuronID{ neuron_id };

        RelearnException::check(node != nullptr, "FastMultipoleMethods::update_leaf_nodes: node was nullptr: ", neuron_id);

        const auto other_neuron_id = node->get_cell().get_neuron_id();

        RelearnException::check(id == other_neuron_id, "FastMultipoleMethods::update_leaf_nodes: The nodes are not in order");

        if (disable_flags[neuron_id] == UpdateStatus::DISABLED) {
            continue;
        }

        const auto number_vacant_dendrites_excitatory = static_cast<counter_type>(dendrites_excitatory_counts[neuron_id] - dendrites_excitatory_connected_counts[neuron_id]);
        const auto number_vacant_dendrites_inhibitory = static_cast<counter_type>(dendrites_inhibitory_counts[neuron_id] - dendrites_inhibitory_connected_counts[neuron_id]);

        node->set_cell_number_dendrites(number_vacant_dendrites_excitatory, number_vacant_dendrites_inhibitory);

        const auto signal_type = axons->get_signal_type(id);

        if (signal_type == SignalType::EXCITATORY) {
            const auto number_vacant_excitatory_axons = static_cast<counter_type>(axons_counts[neuron_id] - axons_connected_counts[neuron_id]);
            const auto number_vacant_inhibitory_axons = 0;

            node->set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        } else {
            const auto number_vacant_excitatory_axons = 0;
            const auto number_vacant_inhibitory_axons = static_cast<counter_type>(axons_counts[neuron_id] - axons_connected_counts[neuron_id]);

            node->set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        }
    }
}

double FastMultipoleMethods::calc_taylor_expansion(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, const double sigma, const SignalType needed) {
    const auto& opt_target_center = target->get_cell().get_dendrites_position_for(needed);
    RelearnException::check(opt_target_center.has_value(), "Target node has no position for Taylor calculation");

    const auto& target_center = opt_target_center.value();

    const auto& indices = Multiindex::get_indices();
    const auto number_coefficients = Multiindex::get_number_of_indices();

    std::array<double, Constants::p3> taylor_coefficients{};

    for (auto index = 0; index < number_coefficients; index++) {
        double temp = 0;
        // NOLINTNEXTLINE
        const auto& current_index = indices[index];

        for (auto j = 0; j < Constants::number_oct; j++) {
            const auto* source_child = source->get_child(j);
            if (source_child == nullptr) {
                continue;
            }

            const auto number_axons = source_child->get_cell().get_number_axons_for(needed);
            if (number_axons == 0) {
                continue;
            }

            const auto& child_pos = source_child->get_cell().get_axons_position_for(needed);
            const auto& temp_vec = (child_pos.value() - target_center) / sigma;
            temp += number_axons * h_multiindex(current_index, temp_vec);
        }

        const auto factorial_multiindex = fac_multiindex(current_index);
        const auto coefficient = temp / factorial_multiindex;

        const auto absolute_multiindex = abs_multiindex(current_index);

        if (absolute_multiindex % 2 == 0) {
            // NOLINTNEXTLINE
            taylor_coefficients[index] = coefficient;
        } else {
            // NOLINTNEXTLINE
            taylor_coefficients[index] = -coefficient;
        }
    }

    double result = 0.0;

    for (auto j = 0; j < Constants::number_oct; j++) {
        const auto* target_child = target->get_child(j);
        if (target_child == nullptr) {
            continue;
        }

        const auto number_dendrites = target_child->get_cell().get_number_dendrites_for(needed);
        if (number_dendrites == 0) {
            continue;
        }

        const auto& child_pos = target_child->get_cell().get_dendrites_position_for(needed);
        const auto& temp_vec = (child_pos.value() - target_center) / sigma;

        double temp = 0.0;
        for (auto b = 0; b < number_coefficients; b++) {
            // NOLINTNEXTLINE
            temp += taylor_coefficients[b] * pow_multiindex(temp_vec, indices[b]);
        }

        result += number_dendrites * temp;
    }

    return result;
}

void FastMultipoleMethods::calc_hermite_coefficients(const OctreeNode<FastMultipoleMethodsCell>* source, std::array<double, Constants::p3>& coefficients_buffer, const double sigma, const SignalType needed) {
    const auto& indices = Multiindex::get_indices();

    for (auto a = 0; a < Constants::p3; a++) {
        auto temp = 0.0;
        for (auto i = 0; i < Constants::number_oct; i++) {
            const auto* child = source->get_child(i);
            if (child == nullptr) {
                continue;
            }

            const auto child_number_axons = child->get_cell().get_number_axons_for(needed);
            if (child_number_axons == 0) {
                continue;
            }

            const auto& child_pos = child->get_cell().get_axons_position_for(needed);
            const auto& source_pos = source->get_cell().get_axons_position_for(needed);
            const auto& temp_vec = (child_pos.value() - source_pos.value()) / sigma;
            temp += child_number_axons * pow_multiindex(temp_vec, indices[a]);
        }

        const auto hermite_coefficient = 1.0 * temp / fac_multiindex(indices[a]);
        coefficients_buffer[a] = hermite_coefficient;
    }
}

double FastMultipoleMethods::calc_hermite(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, const std::array<double, Constants::p3>& coefficients_buffer, const double sigma, const SignalType needed) {
    const auto& opt_source_center = source->get_cell().get_axons_position_for(needed);
    RelearnException::check(opt_source_center.has_value(), "Source node has no axon position for Hermite calculation \n");
    const auto& source_center = opt_source_center.value();

    const auto& indices = Multiindex::get_indices();
    const auto number_coefficients = Multiindex::get_number_of_indices();

    double result = 0.0;

    for (auto j = 0; j < Constants::number_oct; j++) {
        const auto* child_target = target->get_child(j);
        if (child_target == nullptr) {
            continue;
        }

        double temp = 0.0;

        const auto number_dendrites = child_target->get_cell().get_number_dendrites_for(needed);
        if (number_dendrites == 0) {
            continue;
        }

        const auto& child_pos = child_target->get_cell().get_dendrites_position_for(needed);
        const auto& temp_vec = (child_pos.value() - source_center) / sigma;

        for (auto a = 0; a < number_coefficients; a++) {
            // NOLINTNEXTLINE
            temp += coefficients_buffer[a] * h_multiindex(indices[a], temp_vec);
        }

        result += number_dendrites * temp;
    }

    return result;
}