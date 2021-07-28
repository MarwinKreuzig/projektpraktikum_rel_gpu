/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

/*********************************************************************************
  * NOTE: We include Neurons.h here as the class Octree uses types from Neurons.h *
  * Neurons.h also includes Octree.h as it uses it too                            *
  *********************************************************************************/

#include "Octree.h"

#include "../mpi/MPI_RMA_MemAllocator.h"
#include "../io/LogFiles.h"
#include "../neurons/Neurons.h"
#include "../neurons/models/SynapticElements.h"
#include "../structure/SpaceFillingCurve.h"
#include "../util/Random.h"
#include "../util/RelearnException.h"
#include "../util/Timers.h"

#include <sstream>

/**
 * Do a postorder tree walk startring at "octree" and run the function "function" for every node when it is visited
 * Does ignore every node which's level in the octree is greater than "max_level"
 */
void Octree::tree_walk_postorder(std::function<void(OctreeNode<BarnesHutCell>*)> function, OctreeNode<BarnesHutCell>* root, size_t max_level) {
    RelearnException::check(root != nullptr, "In tree_walk_postorder, octree was nullptr");

    std::stack<StackElement> stack{};

    // Push node onto stack
    stack.emplace(root, 0);

    while (!stack.empty()) {
        // Get top-of-stack node
        auto& current_element = stack.top();
        const auto current_depth = current_element.get_depth_in_tree();
        auto* current_octree_node = current_element.get_octree_node();

        // Node should be visited now?
        if (current_element.get_visited()) {
            RelearnException::check(current_octree_node->get_level() <= max_level, "current_element had bad level");

            // Apply action to node
            function(current_octree_node);

            // Pop node from stack
            stack.pop();
        } else {
            // Mark node to be visited next time
            current_element.set_visited();

            // We're at the border of where we want to update, so don't push children
            if (current_depth >= max_level) {
                continue;
            }

            const auto& children = current_octree_node->get_children();
            for (auto it = children.crbegin(); it != children.crend(); ++it) {
                if (*it != nullptr) {
                    stack.emplace(*it, current_depth + 1);
                }
            }
        }
    } /* while */
}

void Octree::construct_global_tree_part() {
    RelearnException::check(root == nullptr, "root was not null in the construction of the global state!");

    SpaceFillingCurve<Morton> space_curve{ static_cast<uint8_t>(level_of_branch_nodes) };

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto num_cells_per_dimension = 1 << level_of_branch_nodes; // (2^level_of_branch_nodes)

    const auto& cell_length = (xyz_max - xyz_min) / num_cells_per_dimension;

    const auto cell_length_x = cell_length.get_x();
    const auto cell_length_y = cell_length.get_y();
    const auto cell_length_z = cell_length.get_z();

    OctreeNode<BarnesHutCell>* local_root = MPI_RMA_MemAllocator<BarnesHutCell>::new_octree_node();
    RelearnException::check(local_root != nullptr, "local_root is nullptr");

    local_root->set_cell_neuron_id(Constants::uninitialized);
    local_root->set_cell_size(xyz_min, xyz_max);
    local_root->set_level(0);
    local_root->set_rank(my_rank);
    local_root->set_cell_neuron_position(xyz_min + (cell_length / 2));

    const auto root_index1d = space_curve.map_3d_to_1d(Vec3s{ 0, 0, 0 });
    branch_nodes[root_index1d] = local_root;

    root = local_root;

    for (size_t id_x = 0; id_x < num_cells_per_dimension; id_x++) {
        for (size_t id_y = 0; id_y < num_cells_per_dimension; id_y++) {
            for (size_t id_z = 0; id_z < num_cells_per_dimension; id_z++) {
                if (id_x == 0 && id_y == 0 && id_z == 0) {
                    continue;
                }

                const Vec3d cell_offset{ id_x * cell_length_x, id_y * cell_length_y, id_z * cell_length_z };
                const auto& cell_min = xyz_min + cell_offset;
                const auto& cell_position = cell_min + (cell_length / 2);

                auto* current_node = root->insert(cell_position, Constants::uninitialized, my_rank);

                //const auto index1d = space_curve.map_3d_to_1d(Vec3s{ id_x, id_y, id_z });
                //branch_nodes[index1d] = current_node;
            }
        }
    }

    std::stack<std::pair<OctreeNode<BarnesHutCell>*, Vec3s>> stack{};
    stack.emplace(root, Vec3s{ 0, 0, 0 });

    while (!stack.empty()) {
        const auto [ptr, index3d] = stack.top();
        stack.pop();

        if (!ptr->is_parent()) {
            const auto index1d = space_curve.map_3d_to_1d(index3d);
            branch_nodes[index1d] = ptr;
            continue;
        }

        for (auto id = 0; id < 8; id++) {
            auto child_node = ptr->get_child(id);

            const size_t larger_x = ((id & 1) == 0) ? 0 : 1;
            const size_t larger_y = ((id & 2) == 0) ? 0 : 1;
            const size_t larger_z = ((id & 4) == 0) ? 0 : 1;

            const Vec3s offset{ larger_x, larger_y, larger_z };
            const Vec3s pos = (index3d * 2) + offset;
            stack.emplace(child_node, pos);
        }
    }
}

// Insert neuron into the tree
OctreeNode<BarnesHutCell>* Octree::insert(const Vec3d& position, size_t neuron_id, int rank) {
    RelearnException::check(xyz_min.get_x() <= position.get_x() && position.get_x() <= xyz_max.get_x(), "In Octree::insert, x was not in range");
    RelearnException::check(xyz_min.get_y() <= position.get_y() && position.get_y() <= xyz_max.get_y(), "In Octree::insert, x was not in range");
    RelearnException::check(xyz_min.get_z() <= position.get_z() && position.get_z() <= xyz_max.get_z(), "In Octree::insert, x was not in range");

    RelearnException::check(rank >= 0, "In Octree::insert, rank was smaller than 0");
    RelearnException::check(neuron_id < Constants::uninitialized, "In Octree::insert, neuron_id was too large");

    // Tree is empty
    if (nullptr == root) {
        // Create new tree node for the neuron
        OctreeNode<BarnesHutCell>* new_node_to_insert = MPI_RMA_MemAllocator<BarnesHutCell>::new_octree_node();
        RelearnException::check(new_node_to_insert != nullptr, "new_node_to_insert is nullptr");

        // Init cell size with simulation box size
        new_node_to_insert->set_cell_size(this->xyz_min, this->xyz_max);
        new_node_to_insert->set_cell_neuron_position({ position });
        new_node_to_insert->set_cell_neuron_id(neuron_id);
        new_node_to_insert->set_rank(rank);

        // Init root with tree's root level
        new_node_to_insert->set_level(0);
        root = new_node_to_insert;

        return new_node_to_insert;
    }

    return root->insert(position, neuron_id, rank);
}

void Octree::print() {
    std::stack<StackElement> stack;

    // Tree is empty
    if (root == nullptr) {
        return;
    }

    stack.emplace(root, 0);

    while (!stack.empty()) {

        std::stringstream ss;

        auto& elem = stack.top();
        const auto depth = static_cast<int>(elem.get_depth_in_tree());
        const auto* ptr = elem.get_octree_node();

        // Visit node now
        if (elem.get_visited()) {
            Vec3d xyz_min;
            Vec3d xyz_max;
            std::optional<Vec3d> xyz_pos;

            // Print node's address
            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }
            ss << "Address: " << ptr << "\n";

            // Print cell extent
            std::tie(xyz_min, xyz_max) = ptr->get_cell().get_size();
            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }

            ss << "Cell extent: (" << xyz_min.get_x() << " .. " << xyz_max.get_x() << ", "
               << xyz_min.get_y() << " .. " << xyz_max.get_y() << ", "
               << xyz_min.get_z() << " .. " << xyz_max.get_z() << ")\n";

            // Print neuron ID
            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }
            ss << "Neuron ID: " << ptr->get_cell().get_neuron_id() << "\n";

            // Print number of dendrites
            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }
            ss << "Number dendrites (exc, inh): (" << ptr->get_cell().get_number_excitatory_dendrites()
               << ", " << ptr->get_cell().get_number_inhibitory_dendrites() << ")\n";

            // Print position DendriteType::EXCITATORY
            xyz_pos = ptr->get_cell().get_excitatory_dendrite_position();
            // Note if position is invalid
            if (!xyz_pos.has_value()) {
                ss << "-- invalid!";
            }

            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }
            ss << "Position exc: (" << xyz_pos.value().get_x() << ", " << xyz_pos.value().get_y() << ", " << xyz_pos.value().get_z() << ") ";

            ss << "\n";
            // Print position DendriteType::INHIBITORY
            xyz_pos = ptr->get_cell().get_inhibitory_dendrite_position();
            // Note if position is invalid
            if (!xyz_pos.has_value()) {
                ss << "-- invalid!";
            }
            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }
            ss << "Position inh: (" << xyz_pos.value().get_x() << ", " << xyz_pos.value().get_y() << ", " << xyz_pos.value().get_z() << ") ";
            ss << "\n";
            ss << "\n";

            stack.pop();
        }
        // Visit children first
        else {
            elem.set_visited();

            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }

            ss << "Child indices: ";
            int id = 0;
            for (const auto& child : ptr->get_children()) {
                if (child != nullptr) {
                    ss << id << " ";
                }
                id++;
            }
            /**
			* Push in reverse order so that visiting happens in
			* increasing order of child indices
			*/
            for (auto it = ptr->get_children().crbegin(); it != ptr->get_children().crend(); ++it) {
                if (*it != nullptr) {
                    stack.emplace(*it, static_cast<size_t>(depth) + 1);
                }
            }
            ss << "\n";
        }

        LogFiles::write_to_file(LogFiles::EventType::Cout, true, ss.str());
    }
}

void Octree::initializes_leaf_nodes(size_t num_neurons) noexcept {
    std::vector<OctreeNode<BarnesHutCell>*> leaf_nodes(num_neurons);

    std::stack<OctreeNode<BarnesHutCell>*> stack;
    stack.emplace(root);

    while (!stack.empty()) {
        OctreeNode<BarnesHutCell>* node = stack.top();
        stack.pop();

        if (node->is_parent()) {
            for (auto* child : node->get_children()) {
                if (child == nullptr || child->get_cell_neuron_id() == Constants::uninitialized && !child->is_parent()) {
                    continue;
                }
                stack.emplace(child);
            }
        } else {
            if (node->get_cell_neuron_id() == Constants::uninitialized) {
                continue;
            }

            leaf_nodes[node->get_cell_neuron_id()] = node;
        }
    }

    all_leaf_nodes = std::move(leaf_nodes);
}

[[nodiscard]] std::array<OctreeNode<BarnesHutCell>*, Constants::number_oct> Octree::downloadChildren(OctreeNode<BarnesHutCell>* node) {
    std::array<OctreeNode<BarnesHutCell>*, Constants::number_oct> local_children{ nullptr };

    const auto target_rank = node->get_rank();

    RelearnException::check(target_rank != MPIWrapper::get_my_rank(), "Tried to download a local node");

    NodesCacheKey rank_addr_pair{};
    rank_addr_pair.first = target_rank;

    // Start access epoch to remote rank
    MPIWrapper::lock_window(target_rank, MPI_Locktype::shared);

    // Fetch remote children if they exist
    // NOLINTNEXTLINE
    for (auto i = 7; i >= 0; i--) {
        if (nullptr == node->get_child(i)) {
            // NOLINTNEXTLINE
            local_children[i] = nullptr;
            continue;
        }

        rank_addr_pair.second = node->get_child(i);

        std::pair<NodesCacheKey, NodesCacheValue> cache_key_val_pair{ rank_addr_pair, nullptr };

        // Get cache entry for "cache_key_val_pair"
        // It is created if it does not exist yet
        std::pair<NodesCache::iterator, bool> ret = remote_nodes_cache.insert(cache_key_val_pair);

        // Cache entry just inserted as it was not in cache
        // So, we still need to init the entry by fetching
        // from the target rank
        if (ret.second) {
            ret.first->second = MPI_RMA_MemAllocator<BarnesHutCell>::new_octree_node();
            auto* local_child_addr = ret.first->second;

            MPIWrapper::download_octree_node<BarnesHutCell>(local_child_addr, target_rank, node->get_child(i));
        }

        // Remember address of node
        // NOLINTNEXTLINE
        local_children[i] = ret.first->second;
    }

    // Complete access epoch
    MPIWrapper::unlock_window(target_rank);

    return local_children;
}

void Octree::empty_remote_nodes_cache() {
    for (auto& remode_node_in_cache : remote_nodes_cache) {
        MPI_RMA_MemAllocator<BarnesHutCell>::delete_octree_node(remode_node_in_cache.second);
    }

    remote_nodes_cache.clear();
}

void Octree::synchronize_local_trees() {
    // Lock local RMA memory for local stores
    /**
    * Exchange branch nodes
    */
    Timers::start(TimerRegion::EXCHANGE_BRANCH_NODES);
    const size_t num_rma_buffer_branch_nodes = branch_nodes.size();
    // Copy local trees' root nodes to correct positions in receive buffer

    std::vector<OctreeNode<BarnesHutCell>> exchange_branch_nodes(num_rma_buffer_branch_nodes);

    const size_t num_local_trees = num_rma_buffer_branch_nodes / MPIWrapper::get_num_ranks();
    for (size_t i = 0; i < num_rma_buffer_branch_nodes; i++) {
        exchange_branch_nodes[i] = *branch_nodes[i];
    }

    // Allgather in-place branch nodes from every rank
    MPIWrapper::all_gather_inline(exchange_branch_nodes.data(), num_local_trees, MPIWrapper::Scope::global);

    Timers::stop_and_add(TimerRegion::EXCHANGE_BRANCH_NODES);

    // Insert only received branch nodes into global tree
    // The local ones are already in the global tree
    Timers::start(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);
    for (size_t i = 0; i < num_rma_buffer_branch_nodes; i++) {
        *branch_nodes[i] = exchange_branch_nodes[i];
    }
    Timers::stop_and_add(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);

    // Update global tree
    Timers::start(TimerRegion::UPDATE_GLOBAL_TREE);

    // Only update whenever there are other branches to update
    if (level_of_branch_nodes > 0) {
        update_from_level(level_of_branch_nodes - 1);
    }

    Timers::stop_and_add(TimerRegion::UPDATE_GLOBAL_TREE);
}
