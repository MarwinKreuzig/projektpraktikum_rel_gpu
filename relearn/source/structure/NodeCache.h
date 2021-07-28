/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include "../Config.h"
#include "../algorithm/BarnesHutCell.h"
#include "../algorithm/FastMultipoleMethodsCell.h"

#include <array>
#include <map>
#include <type_traits>
#include <utility>

template <typename T>
class OctreeNode;

class NodeCache {
public:
    /**
     * @brief Empties the cache that was built during the connection phase and frees all local copies
     * @tparam AdditionalCellAttributes The additional cell attributes that are used for the plasticity algorithm
     */
    template <typename AdditionalCellAttributes>
    static void empty() {
        if constexpr (std::is_same_v<AdditionalCellAttributes, BarnesHutCell>) {
            empty_barnes_hut();
        }

        if constexpr (std::is_same_v<AdditionalCellAttributes, FastMultipoleMethodsCell>) {
            empty_fmm();
        }
    }

    /**
     * @brief Downloads the children of the node (must be on another MPI rank) and returns the children.
     *      Also saves to nodes locally in order to save bandwidth
     * @param node The node for which the children should be downloaded
     * @tparam AdditionalCellAttributes The additional cell attributes that are used for the plasticity algorithm
     * @exception Throws a RelearnException if node is on the current MPI process
     * @return The downloaded children (perfect copies of the actual children), does not transfer ownership
     */
    template <typename AdditionalCellAttributes>
    [[nodiscard]] static std::array<OctreeNode<AdditionalCellAttributes>*, Constants::number_oct> download_children(OctreeNode<AdditionalCellAttributes>* node) {
        if constexpr (std::is_same_v<AdditionalCellAttributes, BarnesHutCell>) {
            return download_children_barnes_hut(node);
        }

        if constexpr (std::is_same_v<AdditionalCellAttributes, FastMultipoleMethodsCell>) {
            return download_children_fmm(node);
        }
    }

private:
    static void empty_barnes_hut();

    static void empty_fmm();

    [[nodiscard]] static std::array<OctreeNode<BarnesHutCell>*, Constants::number_oct> download_children_barnes_hut(OctreeNode<BarnesHutCell>* node);

    [[nodiscard]] static std::array<OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct> download_children_fmm(OctreeNode<FastMultipoleMethodsCell>* node);

    template <typename AdditionalCellAttributes>
    using NodesCacheKey = std::pair<int, OctreeNode<AdditionalCellAttributes>*>;

    template <typename AdditionalCellAttributes>
    using NodesCacheValue = OctreeNode<AdditionalCellAttributes>*;

    template <typename AdditionalCellAttributes>
    using NodesCache = std::map<NodesCacheKey<AdditionalCellAttributes>, NodesCacheValue<AdditionalCellAttributes>>;

    static inline NodesCache<BarnesHutCell> remote_nodes_cache_barnes_hut{};
    static inline NodesCache<FastMultipoleMethodsCell> remote_nodes_cache_fmm{};
};
