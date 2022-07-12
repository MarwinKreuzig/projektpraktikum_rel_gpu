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

#include "BarnesHutBase.h"
#include "Config.h"
#include "Types.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Kernel.h"
#include "mpi/MPIWrapper.h"
#include "neurons/ElementType.h"
#include "neurons/SignalType.h"
#include "neurons/helper/DistantNeuronRequests.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/NodeCache.h"
#include "structure/OctreeNode.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Stack.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <optional>
#include <string>
#include <vector>

/**
 * This class extends computational elements of the Barnes-Hut algorithm with localized searches.
 * It purely calculates things, but does not change any state.
 * @tparam AdditionalCellAttributes The cell attributes that are
 */
template <typename AdditionalCellAttributes>
class BarnesHutLocationAwareBase : public BarnesHutBase<AdditionalCellAttributes> {
public:
    using position_type = typename BarnesHutBase<AdditionalCellAttributes>::position_type;
    using kernel_type = BarnesHutBase<AdditionalCellAttributes>::kernel_type;

protected:
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

            const auto status = BarnesHutBase<AdditionalCellAttributes>::test_acceptance_criterion(source_position, root, element_type, signal_type);
            if (status == BarnesHutBase<AdditionalCellAttributes>::AcceptanceStatus::Accept) {
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
            const auto status = BarnesHutBase<AdditionalCellAttributes>::test_acceptance_criterion(source_position, node, element_type, signal_type);

            if (status == BarnesHutBase<AdditionalCellAttributes>::AcceptanceStatus::Discard) {
                continue;
            }

            const auto is_local = node->is_local();

            if (status == BarnesHutBase<AdditionalCellAttributes>::AcceptanceStatus::Accept || !is_local) {
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
     * @param sigma The probability parameter for the calculation
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorithm found a matching neuron, its RankNeuronId is returned
     */
    [[nodiscard]] std::optional<std::pair<int, DistantNeuronRequest<AdditionalCellAttributes>>> find_target_neuron(const NeuronID& source_neuron_id, const position_type& source_position, OctreeNode<AdditionalCellAttributes>* const root,
        const ElementType element_type, const SignalType signal_type, const double sigma, size_t level_of_branch_nodes) const {
        RelearnException::check(root != nullptr, "BarnesHutLocationAwareBase::find_target_neuron: root was nullptr");

        for (auto root_of_subtree = root; true;) {
            const auto& vector = get_nodes_to_consider(source_position, root_of_subtree, element_type, signal_type);

            auto* node_selected = kernel_type::pick_target(source_neuron_id, source_position, vector, element_type, signal_type, sigma);
            if (node_selected == nullptr) {
                return {};
            }

            if (node_selected == root_of_subtree) {
                return {};
            }

            // If the level of the selected node is too low, continue the search
            if (level_of_branch_nodes < node_selected->get_level()) {

                const auto& cell = node_selected->get_cell();
                const auto& pos = cell.get_position_for(element_type, signal_type);
                const auto& position = pos.value();
                const auto target_rank = node_selected->get_rank();
                const auto target_neuron_id = node_selected->get_cell_neuron_id();

                // If the node is not local, send request to the owning rank 
                if (const auto is_local = node_selected->is_local(); !is_local) {

                    const DistantNeuronRequest<AdditionalCellAttributes> neuron_request(
                        source_neuron_id,
                        source_position,
                        NodeCache<AdditionalCellAttributes>::translate(target_rank, node_selected),
                        signal_type);

                    return std::make_pair(target_rank, neuron_request);
                } 
                // If the node is a child, send request to yourself
                else if (const auto is_child = node_selected->is_child(); is_child) {

                    const DistantNeuronRequest<AdditionalCellAttributes> neuron_request(
                        source_neuron_id,
                        source_position,
                        node_selected,
                        signal_type);

                    return std::make_pair(target_rank, neuron_request);
                }
            }

            // We need to choose again, starting from the chosen virtual neuron
            root_of_subtree = node_selected;
        }
    }

public:

    /**
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::download_children()
     * @param source_neuron_id The source neuron id
     * @param source_position The source position
     * @param root The starting position where to look
     * @param element_type The element type the source is looking for
     * @param signal_type The signal type the source is looking for
     * @param sigma The probability parameter for the calculation
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorithm found a matching neuron, its RankNeuronId is returned
     */
    [[nodiscard]] std::optional<NeuronID> find_local_target_neuron(const NeuronID& source_neuron_id, const position_type& source_position, OctreeNode<AdditionalCellAttributes>* const root,
        const ElementType element_type, const SignalType signal_type, const double sigma) const {
        RelearnException::check(root != nullptr, "BarnesHutLocationAwareBase::find_local_target_neuron: root was nullptr");

        // If the target node is already a child, return its NeuronID
        if (const auto is_child = root->is_child(); is_child) {
            return root->get_cell_neuron_id();
        }

        // Otherwise continue the search in the subdomain of the target node
        for (auto root_of_subtree = root; true;) {
            const auto& vector = get_nodes_to_consider(source_position, root_of_subtree, element_type, signal_type);

            auto* node_selected = kernel_type::pick_target(source_neuron_id, source_position, vector, element_type, signal_type, sigma);
            if (node_selected == nullptr) {
                return {};
            }

            // If the target is not local, discontinue the search
            if (const auto is_local = node_selected->is_local(); !is_local) {
                return {};
            }

            // A chosen child is a valid target
            if (const auto is_child = node_selected->is_child(); is_child) {
                return node_selected->get_cell_neuron_id();
            } 

            // We need to choose again, starting from the chosen virtual neuron
            root_of_subtree = node_selected;
        }
    }
};