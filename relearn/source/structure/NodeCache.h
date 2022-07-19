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
#include <map>
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

public:
    using NodesCacheKey = std::pair<int, node_type*>;
    using NodesCacheValue = node_type*;
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
     * @param node The node for which the children should be downloaded
     * @exception Throws a RelearnException if node is on the current MPI process
     * @return The downloaded children (perfect copies of the actual children), does not transfer ownership
     */
    [[nodiscard]] static std::array<node_type*, Constants::number_oct> download_children(node_type* const node) {
        const auto target_rank = node->get_rank();
        RelearnException::check(target_rank != MPIWrapper::get_my_rank(), "NodeCache::download_children: Tried to download a local node");

        auto actual_download = [target_rank](node_type* node) {
            std::array<node_type*, Constants::number_oct> local_children{ nullptr };

            // Start access epoch to remote rank
            MPIWrapper::lock_window(target_rank, MPI_Locktype::Shared);

            // Fetch remote children if they exist
            for (auto child_index = 0; child_index < Constants::number_oct; child_index++) {
                auto* unusable_child_pointer_on_other_rank = node->get_child(child_index);
                if (nullptr == unusable_child_pointer_on_other_rank) {
                    // NOLINTNEXTLINE
                    local_children[child_index] = nullptr;
                    continue;
                }

                NodesCacheKey rank_addr_pair{ target_rank, unusable_child_pointer_on_other_rank };
                std::pair<NodesCacheKey, NodesCacheValue> cache_key_val_pair{ rank_addr_pair, nullptr };

                // Get cache entry for "cache_key_val_pair"
                // It is created if it does not exist yet
                const auto& [iterator, inserted] = remote_nodes_cache.insert(cache_key_val_pair);

                // Cache entry just inserted as it was not in cache
                // So, we still need to init the entry by fetching
                // from the target rank
                if (inserted) {
                    auto& ref = memory.emplace_back();
                    iterator->second = &ref;
                    auto* local_child_addr = iterator->second;

                    MPIWrapper::download_octree_node<AdditionalCellAttributes>(local_child_addr, target_rank, unusable_child_pointer_on_other_rank);
                }

                // Remember address of node
                // NOLINTNEXTLINE
                local_children[child_index] = iterator->second;
            }

            // Complete access epoch
            MPIWrapper::unlock_window(target_rank);

            return local_children;
        };

        std::array<node_type*, Constants::number_oct> local_children{ nullptr };

#pragma omp critical(node_cache_download)
        local_children = actual_download(node);

        return local_children;
    }

private:
    static inline SemiStableVector<node_type> memory{};
    static inline NodesCache remote_nodes_cache{};
};
