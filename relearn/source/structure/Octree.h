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

#include "../mpi/MPIWrapper.h"
#include "../neurons/SignalType.h"
#include "../neurons/helper/RankNeuronId.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"

#include <map>
#include <optional>
#include <stack>
#include <utility>
#include <vector>

class Neurons;
class Partition;
class SynapticElements;

class Octree {
public:
    friend class Partition;

    using AccessEpochsStarted = std::vector<bool>;

private:
    /**
	 * Type for stack used in postorder tree walk
	 */
    struct StackElement {
    private:
        OctreeNode* ptr;

        // True if node has been on stack already
        // twice and can be visited now
        bool already_visited{ false };

        // Node's depth in the tree
        size_t depth;

    public:
        StackElement(OctreeNode* octree_node, size_t depth_in_tree) noexcept
            : ptr(octree_node)
            , depth(depth_in_tree) {
        }

        [[nodiscard]] OctreeNode* get_octree_node() const noexcept {
            return ptr;
        }

        void set_visited() noexcept {
            already_visited = true;
        }

        [[nodiscard]] bool get_visited() const noexcept {
            return already_visited;
        }

        [[nodiscard]] size_t get_depth_in_tree() const noexcept {
            return depth;
        }
    };

public:
    Octree(const Vec3d& xyz_min, const Vec3d& xyz_max, size_t level_of_branch_nodes);
    ~Octree() = default;

    Octree(const Octree& other) = delete;
    Octree(Octree&& other) = delete;

    Octree& operator=(const Octree& other) = delete;
    Octree& operator=(Octree&& other) = delete;

    // Set simulation box size of the tree
    void set_size(const Vec3d& min, const Vec3d& max) {
        RelearnException::check(min.get_x() < max.get_x() && min.get_y() < max.get_y() && min.get_z() < max.get_z(), "In Octree::set_size, the minimum was not smaller than the maximum");

        xyz_min = min;
        xyz_max = max;
    }

    void set_root_level(size_t root_level) noexcept {
        this->root_level = root_level;
    }

    void set_level_of_branch_nodes(size_t level) noexcept {
        level_of_branch_nodes = level;
    }

    [[nodiscard]] const Vec3d& get_xyz_min() const noexcept {
        return xyz_min;
    }

    [[nodiscard]] const Vec3d& get_xyz_max() const noexcept {
        return xyz_max;
    }

    [[nodiscard]] OctreeNode* get_root() const noexcept {
        return root;
    }

    [[nodiscard]] size_t get_level_of_branch_nodes() const noexcept {
        return level_of_branch_nodes;
    }

    [[nodiscard]] size_t get_num_local_trees() const noexcept {
        return local_trees.size();
    }

    [[nodiscard]] OctreeNode* get_local_root(size_t local_id) noexcept {
        OctreeNode* local_tree = local_trees[local_id];
        return local_tree;
    }

    void print();

    // Insert neuron into the tree
    [[nodiscard]] OctreeNode* insert(const Vec3d& position, size_t neuron_id, int rank);

    // Insert an octree node with its subtree into the tree
    void insert_local_tree(OctreeNode* node_to_insert, size_t index_1d) {
        *local_trees[index_1d] = *node_to_insert;
    }

    [[nodiscard]] std::array<OctreeNode*, Constants::number_oct> downloadChildren(OctreeNode* root);

    // The caller must ensure that only inner nodes are visited.
    // "max_level" must be chosen correctly for this
    void update_from_level(size_t max_level);

    void update_local_trees(const SynapticElements& dendrites_exc, const SynapticElements& dendrites_inh, size_t num_neurons);

    void empty_remote_nodes_cache();

    void synchronize_local_trees();

private:
    /**
	 * Do a postorder tree walk startring at "octree" and run the function "visit" for every node when it is visited
     * Does ignore every node which's level in the octree is greater than "max_level"
	 */
    template <typename Functor>
    void tree_walk_postorder(OctreeNode* root, Functor visit, size_t max_level = std::numeric_limits<size_t>::max()) {
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
                visit(current_octree_node);

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

    /**
	 * Print tree in postorder
	 */
    void postorder_print();

    void construct_global_tree_part();

    // Root of the tree
    OctreeNode* root{ nullptr };

    std::vector<OctreeNode*> local_trees{};

    // Level which is assigned to the root of the tree (default = 0)
    size_t root_level{ Constants::uninitialized };

    // Two points describe simulation box size of the tree
    Vec3d xyz_min{ 0 };
    Vec3d xyz_max{ 0 };

    size_t level_of_branch_nodes{ Constants::uninitialized };

    // Cache with nodes owned by other ranks
    using NodesCacheKey = std::pair<int, OctreeNode*>;
    using NodesCacheValue = OctreeNode*;
    using NodesCache = std::map<NodesCacheKey, NodesCacheValue>;
    NodesCache remote_nodes_cache{};
};
