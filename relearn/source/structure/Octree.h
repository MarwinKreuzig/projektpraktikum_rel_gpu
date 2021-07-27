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
#include "../algorithm/BarnesHut.h"
#include "../mpi/MPIWrapper.h"
#include "../neurons/SignalType.h"
#include "../neurons/helper/RankNeuronId.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"

#include <functional>
#include <map>
#include <optional>
#include <stack>
#include <utility>
#include <vector>

class Neurons;
class Partition;
class SynapticElements;

/**
 * This type represents the (spatial) Octree in which the neurons are organised.
 * It offers general informations about the structure, the functionality to insert new neurons,
 * update from the bottom up, and synchronize parts with MPI.
 */
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
        OctreeNode* ptr{ nullptr };

        // True if node has been on stack already
        // twice and can be visited now
        bool already_visited{ false };

        // Node's depth in the tree
        size_t depth{ Constants::uninitialized };

    public:
        /**
         * @brief Constructs a new object that holds the given node with a specific depth, which is marked as not already visited
         * @param octree_node The node that should be visited, not nullptr
         * @param depth_in_tree The depth of the current node
         * @exception Throws a RelearnException if octree_node is nullptr or depth_in_tree is larger than Cosntants::unitialized
        */
        StackElement(OctreeNode* octree_node, size_t depth_in_tree)
            : ptr(octree_node)
            , depth(depth_in_tree) {
            RelearnException::check(octree_node != nullptr, "StackElement::StackElement, octree_node was nullptr");
            RelearnException::check(depth_in_tree < Constants::uninitialized, "StackElement::StackElement, depth_in_tree was too large");
        }

        /**
         * @brief Returns the node
         * @return The node
         */
        [[nodiscard]] OctreeNode* get_octree_node() const noexcept {
            return ptr;
        }

        /**
         * @brief Sets the flag that indicated if this node was already visited
         * @exception Throws a RelearnException if this node was already visited before
         */
        void set_visited() {
            RelearnException::check(!already_visited, "StackElement::set_visited, element is already visited");
            already_visited = true;
        }

        /**
         * @brief Returns the flag indicating if this node was already visited
         * @return True iff the node was already visited
         */
        [[nodiscard]] bool get_visited() const noexcept {
            return already_visited;
        }

        /**
         * @brief Returns the node
         * @return The node
         */
        [[nodiscard]] size_t get_depth_in_tree() const noexcept {
            return depth;
        }
    };

public:
    /**
     * @brief Constructs a new Octree with the the given size and constructs the "internal" part up to and including the level_of_branch_nodes
     * @param xyz_min The minimum positions of this octree
     * @param xyz_max The maximum positions of this octree
     * @param level_of_branch_nodes The level at which the branch nodes (that are exchanged via MPI) are
     * @exception Throws a RelearnException if xyz_min is not componentwise smaller than xyz_max
     */
    Octree(const Vec3d& xyz_min, const Vec3d& xyz_max, size_t level_of_branch_nodes);
    ~Octree() = default;

    Octree(const Octree& other) = delete;
    Octree(Octree&& other) = delete;

    Octree& operator=(const Octree& other) = delete;
    Octree& operator=(Octree&& other) = delete;

    /**
     * @brief Returns the minimum position in the Octree
     * @return The minimum position in the Octree
     */
    [[nodiscard]] const Vec3d& get_xyz_min() const noexcept {
        return xyz_min;
    }

    /**
     * @brief Returns the maximum position in the Octree
     * @return The maximum position in the Octree
     */
    [[nodiscard]] const Vec3d& get_xyz_max() const noexcept {
        return xyz_max;
    }

    /**
     * @brief Returns the root of the Octree
     * @return The root of the Octree. Ownership is not transfered
     */
    [[nodiscard]] OctreeNode* get_root() const noexcept {
        return root;
    }

    /**
     * @brief Returns the level at which the branch nodes (that are exchanged via MPI) are
     * @return The level at which the branch nodes (that are exchanged via MPI) are
     */
    [[nodiscard]] size_t get_level_of_branch_nodes() const noexcept {
        return level_of_branch_nodes;
    }

    /**
     * @brief Returns the number of branch nodes (that are exchanged via MPI)
     * @return The number of branch nodes (that are exchanged via MPI)
     */
    [[nodiscard]] size_t get_num_local_trees() const noexcept {
        return local_trees.size();
    }

    /**
     * @brief Returns the branch node with the specified local id
     * @param local_id The (1d-) id of the branch node that should be set
     * @exception Throws a RelearnException if local_id is too large
     * @return The branch node with the specified local id. Ownership is not transfered
     */
    [[nodiscard]] OctreeNode* get_local_root(size_t local_id) {
        RelearnException::check(local_id < local_trees.size(), "Octree::get_local_root, local_id was too large");
        OctreeNode* local_tree = local_trees[local_id];
        return local_tree;
    }

    /**
     * @brief Sets the branch node with the specified local id
     * @param node_to_insert The node to insert as local tree, not nullptr, ownership is not taken, must be a parent node
     * @param index_1d The id for which to insert the local tree
     * @exception Throws a RelearnException if index_1d is too large or node_to_insert is nullptr
     */
    void insert_local_tree(OctreeNode* node_to_insert, size_t index_1d) {
        RelearnException::check(index_1d < local_trees.size(), "Octree::get_local_root, local_id was too large");
        RelearnException::check(node_to_insert != nullptr, "Octree::get_local_root, node_to_insert is nullptr");
        RelearnException::check(node_to_insert->is_parent(), "Cannot insert an empty node");
        *local_trees[index_1d] = *node_to_insert;
    }

    /**
     * @brief Inserts a neuron with the specified id and the specified MPI rank into the octree.
     *      If there are no other nodes in this tree, the node becomes the root
     * @param position The position of the new neuron
     * @param neuron_id The id of the new neuron, < Constants::uninitialized (only use for actual neurons, virtual neurons are inserted automatically)
     * @param rank The MPI rank of the new neuron, >= 0
     * @exception Throws a RelearnException if one of the following happens:
     *      (a) The position is not within the octree's boundaries
     *      (b) rank is < 0
     *      (c) neuron_id >= Constants::uninitialized
     *      (d) Allocating a new object in the shared memory window fails
     *      (e) Something went wrong within the insertion
     * @return A pointer to the newly created and inserted node
     */
    [[nodiscard]] OctreeNode* insert(const Vec3d& position, size_t neuron_id, int rank);

    /**
     * @brief Downloads the children of the node (must be on another MPI rank) and returns the children.
     *      Also saves to nodes locally in order to save bandwidth
     * @param node The node for which the children should be downloaded
     * @exception Throws a RelearnException if node is on the current MPI process
     * @return The downloaded children (perfect copies of the actual children), does not transfer ownership
     */
    [[nodiscard]] std::array<OctreeNode*, Constants::number_oct> downloadChildren(OctreeNode* node);

    /**
     * @brief This function updates the Octree starting from max_level. Is is required that it only visits inner nodes
     * @param max_level The maximum level (inclusive) on which the nodes should be updated
     * @exception Throws a RelearnException if the functor throws
     */
    void update_from_level(size_t max_level) {
        tree_walk_postorder(BarnesHut::update_functor, root, max_level);
    }

    /**
     * @brief Updates all local (!) branch nodes and their induced subtrees.
     * @exception Throws a RelearnException if the functor throws
     */
    void update_local_trees() {
        for (auto* local_tree : local_trees) {
            if (!local_tree->is_local()) {
                continue;
            }

            tree_walk_postorder(BarnesHut::update_functor, local_tree);
        }
    }

    /**
     * @brief Empty the cache that was built during the connection phase and frees all local copies
     */
    void empty_remote_nodes_cache();

    /**
     * @brief Synchronizes all (locally) updated branch nodes with all other MPI ranks
     */
    void synchronize_local_trees();

    /** 
     * @brief Prints the Octree to LogFiles::EventType::Cout
     */
    void print();

    /**
     * @brief Returns a constant reference to all leaf nodes
     *      The reference is never invalidated
     * @return All leaf nodes
     */
    const std::vector<OctreeNode*>& get_leaf_nodes() const noexcept {
        return all_leaf_nodes;
    }

    /**
     * @brief Gathers all leaf nodes and makes them available via get_leaf_nodes
     * @param num_neurons The number of neurons
     */
    void initializes_leaf_nodes(size_t num_neurons) noexcept;

private:
    // Set simulation box size of the tree
    void set_size(const Vec3d& min, const Vec3d& max) {
        RelearnException::check(min.get_x() < max.get_x() && min.get_y() < max.get_y() && min.get_z() < max.get_z(), "In Octree::set_size, the minimum was not smaller than the maximum");

        xyz_min = min;
        xyz_max = max;
    }

    /**
	 * Do a postorder tree walk startring at "octree" and run the function "function" for every node when it is visited
     * Does ignore every node which's level in the octree is greater than "max_level"
	 */
    void tree_walk_postorder(std::function<void(OctreeNode*)> function, OctreeNode* root, size_t max_level = std::numeric_limits<size_t>::max());

    void construct_global_tree_part();

    // Root of the tree
    OctreeNode* root{ nullptr };

    std::vector<OctreeNode*> local_trees{};

    std::vector<OctreeNode*> all_leaf_nodes{};

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
