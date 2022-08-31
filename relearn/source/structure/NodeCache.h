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
#include "mpi/MPIWrapper.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"
#include "util/SemiStableVector.h"

#include <array>
#include <iostream>
#include <map>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * This class caches octree nodes from other MPI ranks on the local MPI rank.
 * @tparam AdditionalCellAttributes The additional cell attributes that are used for the plasticity algorithm
 */
template <typename AdditionalCellAttributes>
class NodeCache {
    using node_type = OctreeNode<AdditionalCellAttributes>;
    using children_type = std::array<node_type*, Constants::number_oct>;

public:
    using NodesCacheKey = std::pair<int, node_type*>;
    using NodesCacheValue = children_type;
    using NodesCache = std::map<NodesCacheKey, NodesCacheValue>;

    /**
     * @brief Empties the cache that was built during the connection phase and frees all local copies
     */
    static void empty() {
        remote_nodes_cache.clear();
        memory.clear();
    }

    /**
     * @brief Downloads the children of the node (must be on another MPI rank) and returns the children.
     *      Also saves to nodes locally in order to save bandwidth
     * @param node The node for which the children should be downloaded, must be virtual
     * @exception Throws a RelearnException if node is on the current MPI process or if the saved neuron_id is not virtual
     * @return The downloaded children (perfect copies of the actual children), does not transfer ownership
     */
    [[nodiscard]] static std::array<node_type*, Constants::number_oct> download_children(node_type* const node) {
        const auto target_rank = node->get_rank();
        RelearnException::check(node->get_cell_neuron_id().is_virtual(), "NodeCache::download_children: Tried to download from a non-virtual node");
        RelearnException::check(target_rank != MPIWrapper::get_my_rank(), "NodeCache::download_children: Tried to download a local node");

        auto actual_download = [target_rank](node_type* const node) {
            children_type local_children{ nullptr };
            NodesCacheKey rank_address_pair{ target_rank, node };

            const auto& [iterator, inserted] = remote_nodes_cache.insert({ rank_address_pair, local_children });
            
            if (!inserted) {
                return iterator->second;
            }

            auto offset = node->get_cell_neuron_id().get_rma_offset();

            const auto current_memory_filling = memory.size();
            const auto required_memory_filling = current_memory_filling + Constants::number_oct;

            RelearnException::check(memory.capacity() >= required_memory_filling, "NodeCache::download_children: All {} cache places are full.", memory.capacity());
            memory.resize(required_memory_filling);

            auto* where_to_insert = memory.data() + current_memory_filling;

            // Start access epoch to remote rank
            MPIWrapper::lock_window(target_rank, MPI_Locktype::Shared);
            MPIWrapper::download_octree_node(where_to_insert, target_rank, offset, Constants::number_oct);
            MPIWrapper::unlock_window(target_rank);

            for (auto child_index = 0; child_index < Constants::number_oct; child_index++) {
                if (node->get_child(child_index) == nullptr) {
                    local_children[child_index] = nullptr;
                    continue;
                }

                local_children[child_index] = where_to_insert + child_index;
            }

            iterator->second = local_children;


            return local_children;
        };

        children_type local_children{ nullptr };

#pragma omp critical(node_cache_download)
        local_children = actual_download(node);

        return local_children;
    }

private:
    static inline std::vector<node_type> memory{};
    static inline NodesCache remote_nodes_cache{};
    static inline NodesCache inverse_remote_nodes_cache{};
};
