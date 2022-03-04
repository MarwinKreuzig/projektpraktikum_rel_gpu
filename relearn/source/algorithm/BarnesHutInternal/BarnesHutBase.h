#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"
#include "Types.h"
#include "algorithm/Algorithm.h"
#include "neurons/ElementType.h"
#include "neurons/SignalType.h"
#include "neurons/models/SynapticElements.h"
#include "structure/NodeCache.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

template <typename AdditionalCellAttributes>
class BarnesHutBase : public Algorithm {
public:
    constexpr static double default_theta{ 0.3 };
    constexpr static double max_theta{ 0.5 };

    using position_type = typename RelearnTypes::position_type;
    using counter_type = typename RelearnTypes::counter_type;

protected:
    double acceptance_criterion{ default_theta };

public:
    /**
     * This enum indicates for an OctreeNode what the acceptance status is
     * It can be:
     * - Discard (no dendrites there)
     * - Expand (would be too much approximation, need to expand)
     * - Accept (can use the node for the algorithm)
     */
    enum class AcceptanceStatus : char {
        Discard = 0,
        Expand = 1,
        Accept = 2,
    };

    /**
     * @brief Sets acceptance criterion for cells in the tree
     * @param acceptance_criterion The acceptance criterion, >= 0.0
     * @exception Throws a RelearnException if acceptance_criterion < 0.0
     */
    void set_acceptance_criterion(const double acceptance_criterion) {
        RelearnException::check(acceptance_criterion > 0.0, "BarnesHut::set_acceptance_criterion: acceptance_criterion was less than or equal to 0 ({})", acceptance_criterion);
        this->acceptance_criterion = acceptance_criterion;
    }

    /**
     * @brief Returns the currently used acceptance criterion
     * @return The currently used acceptance criterion
     */
    [[nodiscard]] double get_acceptance_criterion() const noexcept {
        return acceptance_criterion;
    }

protected:
    /**
     * @brief Calculates the attractiveness to connect on the basis of
     *      k * exp(-||s - t||_2^2 / sigma^2)
     * @param source_position The source position s
     * @param target_position The target position t
     * @param number_free_elements The linear scaling factor k
     * @param sigma The exponential scaling factor sigma
     * @return The calculated attractiveness
     */
    [[nodiscard]] double calculate_attractiveness_to_connect(const position_type& source_position, const position_type& target_position,
        const counter_type& number_free_elements, const double sigma) const noexcept {
        if (number_free_elements == 0) {
            return 0.0;
        }

        const auto squared_sigma = sigma * sigma;

        const auto position_diff = target_position - source_position;

        const auto numerator = position_diff.calculate_squared_2_norm();
        const auto exponent = -numerator / squared_sigma;

        // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
        const auto exp_val = std::exp(exponent);
        const auto ret_val = number_free_elements * exp_val;

        return ret_val;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of
     *      k * exp(-||s - t||_2^2 / sigma^2)
     *      It also prevents autapses and checks for algorithmic errors
     * @param source_position The source position s
     * @param target_position The target position t
     * @param number_free_elements The linear scaling factor k
     * @param sigma The exponential scaling factor sigma
     * @exception Throws a RelearnException if there was an algorithmic error somewhere
     * @return The calculated attractiveness
     */
    [[nodiscard]] double calculate_attractiveness_to_connect(const NeuronID& source_neuron_id, const position_type& source_position,
        const OctreeNode<AdditionalCellAttributes>* target_node, const ElementType element_type, const SignalType signal_type) const {

        // A neuron must not form an autapse, i.e., a synapse to itself
        if (target_node->is_child() && source_neuron_id == target_node->get_cell_neuron_id()) {
            return 0.0;
        }

        const auto& cell = target_node->get_cell();
        const auto& target_position = cell.get_position_for(element_type, signal_type);
        const auto& number_elements = cell.get_number_elements_for(element_type, signal_type);

        RelearnException::check(target_position.has_value(), "BarnesHutBase::calculate_attractiveness_to_connect: target_xyz is bad");

        const auto sigma = get_probabilty_parameter();
        return calculate_attractiveness_to_connect(source_position, target_position.value(), number_elements, sigma);
    }

    /**
     * @brief Tests the Barnes-Hut criterion on the source position and the target wrt. to required element type and signal type
     * @param source_position The source position of the calculation
     * @param target_node The target node within the Octree that should be considered
     * @param element_type The type of elements that are searched for
     * @param signal_type The signal type of the elements that are searched for
     * @exception Throws a RelearnEception if there was an algorithmic error
     * @return The acceptance status for the node, i.e., if it must be discarded, can be accepted, or must be expanded.
     */
    [[nodiscard]] AcceptanceStatus test_acceptance_criterion(const position_type& source_position, const OctreeNode<AdditionalCellAttributes>* target_node,
        const ElementType element_type, const SignalType signal_type) const {
        RelearnException::check(target_node != nullptr, "BarnesHutBase::test_acceptance_criterion: target_node was nullptr");

        const auto& cell = target_node->get_cell();

        // Never accept a node with zero vacant elements
        if (const auto number_vacant_elements = cell.get_number_elements_for(element_type, signal_type); number_vacant_elements == 0) {
            return AcceptanceStatus::Discard;
        }

        // Always accept a leaf noce
        if (const auto is_child = target_node->is_child(); is_child) {
            return AcceptanceStatus::Accept;
        }

        // Check distance between source and target
        const auto& target_position = cell.get_position_for(element_type, signal_type);

        // NOTE: This assertion fails when considering inner nodes that don't have the required elements.
        RelearnException::check(target_position.has_value(), "BarnesHutBase::test_acceptance_criterion: target_position was bad");

        // Calc Euclidean distance between source and target neuron
        const auto& distance_vector = target_position.value() - source_position;
        const auto distance = distance_vector.calculate_2_norm();

        // No autapse
        if (distance == 0.0) {
            return AcceptanceStatus::Discard;
        }

        const auto length = cell.get_maximal_dimension_difference();

        // Original Barnes-Hut acceptance criterion
        const auto ret_val = (length / distance) < acceptance_criterion;
        return ret_val ? AcceptanceStatus::Accept : AcceptanceStatus::Expand;
    }

    /**
     * @brief Calculates the probability for the source neuron to connect to each of the OctreeNodes in the vector,
     *      searching the specified element_type and signal_type
     * @param source_neuron_id The id of the source neuron, is used to prevent autapses
     * @param source_position The position of the source neuron
     * @param nodes All nodes from which the source neuron can pick
     * @param element_type The element type the source neuron searches
     * @param signal_type The signal type the source neuron searches
     * @exception Can Throw a RelearnException if an algorithmic errors occurs
     * @return A pair of (a) the total probability of all targets and (b) the respective probability of each target
     */
    [[nodiscard]] std::pair<double, std::vector<double>> create_probability_interval(const NeuronID& source_neuron_id, const position_type& source_position,
        const std::vector<OctreeNode<AdditionalCellAttributes>*>& nodes, const ElementType element_type, const SignalType signal_type) const {

        if (nodes.empty()) {
            return { 0.0, {} };
        }

        double sum = 0.0;

        std::vector<double> probabilities{};
        probabilities.reserve(nodes.size());

        std::transform(nodes.begin(), nodes.cend(), std::back_inserter(probabilities), [&](const OctreeNode<AdditionalCellAttributes>* target_node) {
            RelearnException::check(target_node != nullptr, "BarnesHut::update_leaf_nodes: target_node was nullptr");
            const auto prob = calculate_attractiveness_to_connect(source_neuron_id, source_position, target_node, element_type, signal_type);
            sum += prob;
            return prob;
        });

        // Short-cut an empty vector here for later uses
        if (sum == 0.0) {
            return { 0.0, {} };
        }

        return { sum, std::move(probabilities) };
    }

    /**
     * @brief Picks a target based on the supplied probabilities
     * @param nodes The target nodes
     * @param probability A pair of (a) the total probability of all targets and (b) the respective probability of each target
     * @exception Throws a RelearnException if the sizes of the vectors didn't match, or if one OctreeNode* was nullptr
     * @return The selected target node or nullptr in case that the total probability was 0.0
     */
    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* pick_target(const std::vector<OctreeNode<AdditionalCellAttributes>*>& nodes, const std::pair<double, std::vector<double>>& probability) const {
        const auto& [total_prob, probability_values] = probability;

        if (total_prob == 0.0) {
            return nullptr;
        }

        RelearnException::check(nodes.size() == probability_values.size(), "BarnesHutBase::pick_target: Had a different number of probabilities than nodes: {} vs {}", nodes.size(), probability_values.size());

        const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::Algorithm, 0.0, std::nextafter(total_prob, Constants::eps));
        auto counter = 0;
        for (auto sum_probabilities = 0.0; counter < probability_values.size() && sum_probabilities < random_number; counter++) {
            sum_probabilities += probability_values[counter];
        }
        auto* node_selected = nodes[counter - 1ULL];

        RelearnException::check(node_selected != nullptr, "BarnesHutBase::pick_target: node_selected was nullptr");

        return node_selected;
    }

    /**
     * @brief Searches all neurons that must be considered as targets starting at root
     * @param source_position The position of the source
     * @param root The start where the source searches for targets
     * @param element_type The element type that the source searches
     * @param signal_type The signal type that the source searches
     */
    [[nodiscard]] std::vector<OctreeNode<AdditionalCellAttributes>*> get_nodes_to_consider(const position_type& source_position, OctreeNode<AdditionalCellAttributes>* root,
        const ElementType element_type, const SignalType signal_type) const {
        if (root == nullptr) {
            return {};
        }

        if (root->get_cell().get_number_elements_for(element_type, signal_type) == 0) {
            return {};
        }

        if (root->is_child()) {
            /**
             * The root node is a leaf and thus contains the target neuron.
             *
             * NOTE: Root is not intended to be a leaf but we handle this as well.
             * Without pushing root onto the stack, it would not make it into the "vector" of nodes.
             */

            const auto status = test_acceptance_criterion(source_position, root, element_type, signal_type);
            if (status == AcceptanceStatus::Accept) {
                return { root };
            }

            return {};
        }

        std::vector<OctreeNode<AdditionalCellAttributes>*> vector{};
        vector.reserve(Constants::number_prealloc_space);

        const auto add_children_to_vector = [&vector](OctreeNode<AdditionalCellAttributes>* node) {
            const auto is_local = node->is_local();
            const auto& children = is_local ? node->get_children() : NodeCache::download_children<AdditionalCellAttributes>(node);

            for (auto* it : children) {
                if (it != nullptr) {
                    vector.emplace_back(it);
                }
            }
        };

        // The algorithm expects that root is not considered directly, rather its children
        add_children_to_vector(root);

        std::vector<OctreeNode<AdditionalCellAttributes>*> nodes_to_consider{};
        nodes_to_consider.reserve(Constants::number_prealloc_space);

        while (!vector.empty()) {
            // Get top-of-stack node and remove it
            auto* node = vector[vector.size() - 1];
            vector.pop_back();

            /**
             * Should node be used for probability interval?
             * Only take those that have axons available
             */
            const auto status = test_acceptance_criterion(source_position, node, element_type, signal_type);

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

    /**
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::download_children()
     * @param source_neuron_id The source neuron id
     * @param source_position The source position
     * @param root The starting position where to look
     * @param element_type The element type the source is looking for
     * @param signal_type The signal type the source is looking for
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorihtm found a matching neuron, its RankNeuronId is returned
     */
    [[nodiscard]] std::optional<RankNeuronId> find_target_neuron(const NeuronID& source_neuron_id, const position_type& source_position, OctreeNode<AdditionalCellAttributes>* root,
        const ElementType element_type, const SignalType signal_type) const {
        OctreeNode<AdditionalCellAttributes>* node_selected = nullptr;
        OctreeNode<AdditionalCellAttributes>* root_of_subtree = root;

        RelearnException::check(root_of_subtree != nullptr, "BarnesHut::find_target_neuron: root_of_subtree was nullptr");

        while (true) {
            /**
             * Create vector with nodes that have at least one axon and are
             * precise enough given the position of an axon
             */
            const auto& vector = get_nodes_to_consider(source_position, root_of_subtree, element_type, signal_type);

            /**
             * Assign a probability to each node in the vector.
             * The probability for connecting to the same neuron (i.e., the axon's neuron) is set 0.
             */
            const auto& probability = create_probability_interval(source_neuron_id, source_position, vector, element_type, signal_type);

            node_selected = pick_target(vector, probability);
            if (node_selected == nullptr) {
                return {};
            }

            // Leave loop if the selected node is leaf node, i.e., contains normal neuron.
            if (const auto done = !node_selected->is_parent(); done) {
                break;
            }

            // Update root of subtree, we need to choose starting from this root again
            root_of_subtree = node_selected;
        }

        return RankNeuronId{ node_selected->get_rank(), node_selected->get_cell_neuron_id() };
    }

    /**
     * @brief Finds target neurons for a specified source neuron
     * @param source_neuron_id The source neuron's id
     * @param source_position The source neuron's position
     * @param number_vacant_elements The number of vacant elements of the source neuron
     * @param root Where the source neuron should start to search for targets
     * @param element_type The element type the source neuron searches
     * @param signal_type The signal type the source neuron searches
     * @return A vector of pairs with (a) the target mpi rank and (b) the request for that rank
     */
    [[nodiscard]] std::vector<std::pair<int, SynapseCreationRequest>> find_target_neurons(const NeuronID& source_neuron_id, const position_type& source_position, const counter_type& number_vacant_elements,
        OctreeNode<AdditionalCellAttributes>* root, const ElementType element_type, const SignalType signal_type) {

        std::vector<std::pair<int, SynapseCreationRequest>> requests{};
        requests.reserve(number_vacant_elements);

        for (unsigned int j = 0; j < number_vacant_elements; j++) {
            // Find one target at the time
            std::optional<RankNeuronId> rank_neuron_id = find_target_neuron(source_neuron_id, source_position, root, element_type, signal_type);
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
};
