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
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Kernel.h"
#include "neurons/ElementType.h"
#include "neurons/SignalType.h"
#include "structure/NodeCache.h"
#include "structure/OctreeNode.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Stack.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

/**
 * This class provides all computational elements of the Barnes-Hut algorithm.
 * It purely calculates things, but does not change any state.
 * @tparam AdditionalCellAttributes The cell attributes that are
 */
template <typename AdditionalCellAttributes>
class BarnesHutBase {
public:
    constexpr static double default_theta{ 0.3 };
    constexpr static double max_theta{ 0.5 };

    using position_type = typename RelearnTypes::position_type;
    using counter_type = typename RelearnTypes::counter_type;

    using kernel_type = Kernel<AdditionalCellAttributes, GaussianKernel>;

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
     * @param acceptance_criterion The acceptance criterion, > 0.0
     * @exception Throws a RelearnException if acceptance_criterion <= 0.0
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

        Stack<OctreeNode<AdditionalCellAttributes>*> stack(Constants::number_prealloc_space);

        const auto add_children = [&stack](OctreeNode<AdditionalCellAttributes>* node) {
            const auto is_local = node->is_local();
            const auto& children = is_local ? node->get_children() : NodeCache<AdditionalCellAttributes>::download_children(node);

            for (auto* it : children) {
                if (it != nullptr) {
                    stack.emplace_back(it);
                }
            }
        };

        // The algorithm expects that root is not considered directly, rather its children
        add_children(root);

        std::vector<OctreeNode<AdditionalCellAttributes>*> nodes_to_consider{};
        nodes_to_consider.reserve(Constants::number_prealloc_space);

        while (!stack.empty()) {
            // Get top-of-stack node and remove it
            auto* node = stack.pop_back();

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
            add_children(node);
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
     *      If the algorithm found a matching neuron, its RankNeuronId is returned
     */
    [[nodiscard]] std::optional<std::pair<RankNeuronId, double>> find_target_neuron(const NeuronID& source_neuron_id, const position_type& source_position, OctreeNode<AdditionalCellAttributes>* const root,
        const ElementType element_type, const SignalType signal_type) const {
        RelearnException::check(root != nullptr, "BarnesHutBase::find_target_neuron: root was nullptr");

        for (auto root_of_subtree = root; true;) {
            const auto& vector = get_nodes_to_consider(source_position, root_of_subtree, element_type, signal_type);

            auto* node_selected = kernel_type::pick_target(source_neuron_id, source_position, vector, element_type, signal_type);
            if (node_selected == nullptr) {
                return {};
            }

            // A chosen child is a valid target
            if (const auto done = node_selected->is_child(); done) {
                const auto& cell = node_selected->get_cell();
                const auto& pos = cell.get_position_for(element_type, signal_type);
                const auto& position = pos.value();

                const auto& diff = source_position - position;
                const auto& length = diff.calculate_2_norm();

                return std::make_pair(RankNeuronId{ node_selected->get_rank(), node_selected->get_cell_neuron_id() }, length);
            }

            // We need to choose again, starting from the chosen virtual neuron
            root_of_subtree = node_selected;
        }
    }
};
