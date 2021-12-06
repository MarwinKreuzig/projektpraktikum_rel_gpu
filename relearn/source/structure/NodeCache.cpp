/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NodeCache.h"

#include "../mpi/MPIWrapper.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"

void NodeCache::empty_barnes_hut() {
    for (auto& remode_node_in_cache : remote_nodes_cache_barnes_hut) {
        OctreeNode<BarnesHutCell>::free(remode_node_in_cache.second);
    }

    remote_nodes_cache_barnes_hut.clear();
}

void NodeCache::empty_fmm() {
    for (auto& remode_node_in_cache : remote_nodes_cache_fmm) {
        OctreeNode<FastMultipoleMethodsCell>::free(remode_node_in_cache.second);
    }

    remote_nodes_cache_fmm.clear();
}

[[nodiscard]] std::array<OctreeNode<BarnesHutCell>*, Constants::number_oct> NodeCache::download_children_barnes_hut(OctreeNode<BarnesHutCell>* node) {
    std::array<OctreeNode<BarnesHutCell>*, Constants::number_oct> local_children{ nullptr };

    const auto target_rank = node->get_rank();

    RelearnException::check(target_rank != MPIWrapper::get_my_rank(), "NodeCache::download_children_barnes_hut: Tried to download a local node");

    // Start access epoch to remote rank
    MPIWrapper::lock_window(target_rank, MPI_Locktype::shared);

    // Fetch remote children if they exist
    // NOLINTNEXTLINE
    for (auto i = Constants::number_oct; i >= 1; i--) {
        const auto child_index = i - 1;

        if (nullptr == node->get_child(child_index)) {
            // NOLINTNEXTLINE
            local_children[child_index] = nullptr;
            continue;
        }

        NodesCacheKey<BarnesHutCell> rank_addr_pair{ target_rank, node->get_child(child_index) };
        std::pair<NodesCacheKey<BarnesHutCell>, NodesCacheValue<BarnesHutCell>> cache_key_val_pair{ rank_addr_pair, nullptr };

        // Get cache entry for "cache_key_val_pair"
        // It is created if it does not exist yet
        const auto& [iterator, inserted] = remote_nodes_cache_barnes_hut.insert(cache_key_val_pair);

        // Cache entry just inserted as it was not in cache
        // So, we still need to init the entry by fetching
        // from the target rank
        if (inserted) {
            iterator->second = OctreeNode<BarnesHutCell>::create();
            auto* local_child_addr = iterator->second;

            MPIWrapper::download_octree_node<BarnesHutCell>(local_child_addr, target_rank, node->get_child(child_index));
        }

        // Remember address of node
        // NOLINTNEXTLINE
        local_children[child_index] = iterator->second;
    }

    // Complete access epoch
    MPIWrapper::unlock_window(target_rank);

    return local_children;
}

std::array<OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct> NodeCache::download_children_fmm(OctreeNode<FastMultipoleMethodsCell>* node) {
    std::array<OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct> local_children{ nullptr };

    const auto target_rank = node->get_rank();

    RelearnException::check(target_rank != MPIWrapper::get_my_rank(), "NodeCache::download_children_fmm: Tried to download a local node");

    NodesCacheKey<FastMultipoleMethodsCell> rank_addr_pair{};
    rank_addr_pair.first = target_rank;

    // Start access epoch to remote rank
    MPIWrapper::lock_window(target_rank, MPI_Locktype::shared);

    // Fetch remote children if they exist
    // NOLINTNEXTLINE
    for (int i = Constants::number_oct - 1; i >= 0; i--) {
        if (nullptr == node->get_child(i)) {
            // NOLINTNEXTLINE
            local_children[i] = nullptr;
            continue;
        }

        rank_addr_pair.second = node->get_child(i);

        std::pair<NodesCacheKey<FastMultipoleMethodsCell>, NodesCacheValue<FastMultipoleMethodsCell>> cache_key_val_pair{ rank_addr_pair, nullptr };

        // Get cache entry for "cache_key_val_pair"
        // It is created if it does not exist yet
        std::pair<NodesCache<FastMultipoleMethodsCell>::iterator, bool> ret = remote_nodes_cache_fmm.insert(cache_key_val_pair);

        // Cache entry just inserted as it was not in cache
        // So, we still need to init the entry by fetching
        // from the target rank
        if (ret.second) {
            ret.first->second = OctreeNode<FastMultipoleMethodsCell>::create();
            auto* local_child_addr = ret.first->second;

            MPIWrapper::download_octree_node<FastMultipoleMethodsCell>(local_child_addr, target_rank, node->get_child(i));
        }

        // Remember address of node
        // NOLINTNEXTLINE
        local_children[i] = ret.first->second;
    }

    // Complete access epoch
    MPIWrapper::unlock_window(target_rank);

    return local_children;
}
