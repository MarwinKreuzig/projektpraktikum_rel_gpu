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
#include "../algorithm/BarnesHutCell.h"
#include "../mpi/MPIWrapper.h"
#include "../neurons/SignalType.h"
#include "../neurons/helper/RankNeuronId.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"
#include "../mpi/MPI_RMA_MemAllocator.h"
#include "../io/LogFiles.h"
#include "../neurons/Neurons.h"
#include "../neurons/models/SynapticElements.h"
#include "../structure/SpaceFillingCurve.h"
#include "../util/Random.h"
#include "../util/RelearnException.h"
#include "../util/Timers.h"

#include <sstream>

#include <functional>
#include <map>
#include <optional>
#include <stack>
#include <utility>
#include <variant>
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

    /**
     * @brief Constructs a new Octree with the the given size and constructs the "internal" part up to and including the level_of_branch_nodes
     * @param xyz_min The minimum positions of this octree
     * @param xyz_max The maximum positions of this octree
     * @param level_of_branch_nodes The level at which the branch nodes (that are exchanged via MPI) are
     * @exception Throws a RelearnException if xyz_min is not componentwise smaller than xyz_max
     */
    Octree(const Vec3d& xyz_min, const Vec3d& xyz_max, size_t level_of_branch_nodes)
        : level_of_branch_nodes(level_of_branch_nodes) {
        set_size(xyz_min, xyz_max);
    }
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
     * @brief Returns the number of branch nodes (that are exchanged via MPI)
     * @return The number of branch nodes (that are exchanged via MPI)
     */
    [[nodiscard]] virtual size_t get_num_local_trees() const noexcept = 0;

    /**
     * @brief Returns the level at which the branch nodes (that are exchanged via MPI) are
     * @return The level at which the branch nodes (that are exchanged via MPI) are
     */
    [[nodiscard]] size_t get_level_of_branch_nodes() const noexcept {
        return level_of_branch_nodes;
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
    virtual void insert(const Vec3d& position, size_t neuron_id, int rank) = 0;

    /**
     * @brief This function updates the Octree starting from max_level. Is is required that it only visits inner nodes
     * @param max_level The maximum level (inclusive) on which the nodes should be updated
     * @exception Throws a RelearnException if the functor throws
     */
    virtual void update_from_level(size_t max_level) = 0;

    /**
     * @brief Updates all local (!) branch nodes and their induced subtrees.
     * @exception Throws a RelearnException if the functor throws
     */
    virtual void update_local_trees() = 0;

    /**
     * @brief Synchronizes all (locally) updated branch nodes with all other MPI ranks
     */
    virtual void synchronize_local_trees() = 0;

    /**
     * @brief Gathers all leaf nodes and makes them available via get_leaf_nodes
     * @param num_neurons The number of neurons
     */
    virtual void initializes_leaf_nodes(size_t num_neurons) = 0;

protected:
    // Set simulation box size of the tree
    void set_size(const Vec3d& min, const Vec3d& max) {
        RelearnException::check(min.get_x() < max.get_x() && min.get_y() < max.get_y() && min.get_z() < max.get_z(), "In Octree::set_size, the minimum was not smaller than the maximum");

        xyz_min = min;
        xyz_max = max;
    }

    // Two points describe simulation box size of the tree
    Vec3d xyz_min{ 0 };
    Vec3d xyz_max{ 0 };

    size_t level_of_branch_nodes{ Constants::uninitialized };
};

template <typename AdditionalCellAttributes>
class OctreeImplementation : public Octree {
protected:
    /**
	 * Type for stack used in postorder tree walk
	 */
    struct StackElement {
    private:
        OctreeNode<AdditionalCellAttributes>* ptr{ nullptr };

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
        StackElement(OctreeNode<AdditionalCellAttributes>* octree_node, size_t depth_in_tree)
            : ptr(octree_node)
            , depth(depth_in_tree) {
            RelearnException::check(octree_node != nullptr, "StackElement::StackElement, octree_node was nullptr");
            RelearnException::check(depth_in_tree < Constants::uninitialized, "StackElement::StackElement, depth_in_tree was too large");
        }

        /**
         * @brief Returns the node
         * @return The node
         */
        [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_octree_node() const noexcept {
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
    OctreeImplementation(const Vec3d& xyz_min, const Vec3d& xyz_max, size_t level_of_branch_nodes)
        : Octree(xyz_min, xyz_max, level_of_branch_nodes) {

        const auto num_local_trees = 1ULL << (3 * level_of_branch_nodes);
        branch_nodes.resize(num_local_trees, nullptr);

        construct_global_tree_part();
    }

    /**
     * @brief Returns the root of the Octree
     * @return The root of the Octree. Ownership is not transfered
     */
    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_root() const noexcept {
        return root;
    }

    /**
     * @brief Returns the number of branch nodes (that are exchanged via MPI)
     * @return The number of branch nodes (that are exchanged via MPI)
     */
    [[nodiscard]] size_t get_num_local_trees() const noexcept override {
        return branch_nodes.size();
    }

    /**
     * @brief This function updates the Octree starting from max_level. Is is required that it only visits inner nodes
     * @param max_level The maximum level (inclusive) on which the nodes should be updated
     * @exception Throws a RelearnException if the functor throws
     */
    void update_from_level(size_t max_level) override {
        tree_walk_postorder(BarnesHut::update_functor, root, max_level);
    }

    /**
     * @brief Updates all local (!) branch nodes and their induced subtrees.
     * @exception Throws a RelearnException if the functor throws
     */
    void update_local_trees() override {
        for (auto* local_tree : branch_nodes) {
            if (!local_tree->is_local()) {
                continue;
            }

            tree_walk_postorder(BarnesHut::update_functor, local_tree);
        }
    }

    void initializes_leaf_nodes(size_t num_neurons) override {
        std::vector<OctreeNode<AdditionalCellAttributes>*> leaf_nodes(num_neurons);

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack;
        stack.emplace(root);

        while (!stack.empty()) {
            OctreeNode<AdditionalCellAttributes>* node = stack.top();
            stack.pop();

            if (node->is_parent()) {
                for (auto* child : node->get_children()) {

                    if (child == nullptr) {
                        continue;
                    }

                    if (!child->is_parent() && child->get_cell_neuron_id() >= Constants::uninitialized) {
                        continue;
                    }

                    stack.emplace(child);
                }
            } else {
                const auto neuron_id = node->get_cell_neuron_id();
                RelearnException::check(neuron_id < leaf_nodes.size(), "Neuron id was too large for leaf nodes");
                leaf_nodes[neuron_id] = node;
            }
        }

        all_leaf_nodes = std::move(leaf_nodes);
    }

    /**
     * @brief Synchronizes all (locally) updated branch nodes with all other MPI ranks
     */
    void synchronize_local_trees() override {
        /**
         * Exchange branch nodes
         */
        Timers::start(TimerRegion::EXCHANGE_BRANCH_NODES);
        const size_t num_rma_buffer_branch_nodes = branch_nodes.size();
        // Copy local trees' root nodes to correct positions in receive buffer

        std::vector<OctreeNode<AdditionalCellAttributes>> exchange_branch_nodes(num_rma_buffer_branch_nodes);

        const size_t num_local_trees = num_rma_buffer_branch_nodes / MPIWrapper::get_num_ranks();
        for (size_t i = 0; i < num_rma_buffer_branch_nodes; i++) {
            exchange_branch_nodes[i] = *branch_nodes[i];
        }

        // Allgather in-place branch nodes from every rank
        MPIWrapper::all_gather_inline(exchange_branch_nodes.data(), num_local_trees);

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

    /**
     * @brief Sets the branch node with the specified local id
     * @param node_to_insert The node to insert as local tree, not nullptr, ownership is not taken, must be a parent node
     * @param index_1d The id for which to insert the local tree
     * @exception Throws a RelearnException if index_1d is too large or node_to_insert is nullptr
     */
    void insert_local_tree(OctreeNode<AdditionalCellAttributes>* node_to_insert, size_t index_1d) {
        RelearnException::check(index_1d < branch_nodes.size(), "Octree::insert_local_tree, local_id was too large");
        RelearnException::check(node_to_insert != nullptr, "Octree::insert_local_tree, node_to_insert is nullptr");
        RelearnException::check(node_to_insert->is_parent(), "Cannot insert an empty node");
        *branch_nodes[index_1d] = *node_to_insert;
    }

    /**
     * @brief Returns a constant reference to all leaf nodes
     *      The reference is never invalidated
     * @return All leaf nodes
     */
    const std::vector<OctreeNode<AdditionalCellAttributes>*>& get_leaf_nodes() const noexcept {
        return all_leaf_nodes;
    }

    void insert(const Vec3d& position, size_t neuron_id, int rank) {
        RelearnException::check(xyz_min.get_x() <= position.get_x() && position.get_x() <= xyz_max.get_x(), "In Octree::insert, x was not in range");
        RelearnException::check(xyz_min.get_y() <= position.get_y() && position.get_y() <= xyz_max.get_y(), "In Octree::insert, x was not in range");
        RelearnException::check(xyz_min.get_z() <= position.get_z() && position.get_z() <= xyz_max.get_z(), "In Octree::insert, x was not in range");

        RelearnException::check(rank >= 0, "In Octree::insert, rank was smaller than 0");
        RelearnException::check(neuron_id < Constants::uninitialized, "In Octree::insert, neuron_id was too large");

        // Tree is empty
        if (nullptr == root) {
            // Create new tree node for the neuron
            OctreeNode<AdditionalCellAttributes>* new_node_to_insert = OctreeNode<AdditionalCellAttributes>::create();
            RelearnException::check(new_node_to_insert != nullptr, "new_node_to_insert is nullptr");

            // Init cell size with simulation box size
            new_node_to_insert->set_cell_size(this->xyz_min, this->xyz_max);
            new_node_to_insert->set_cell_neuron_position({ position });
            new_node_to_insert->set_cell_neuron_id(neuron_id);
            new_node_to_insert->set_rank(rank);

            // Init root with tree's root level
            new_node_to_insert->set_level(0);
            root = new_node_to_insert;

            return;
        }

        auto* res = root->insert(position, neuron_id, rank);
        RelearnException::check(res != nullptr, "Octree::insert had nullptr");
    }

    void tree_walk_postorder(std::function<void(OctreeNode<AdditionalCellAttributes>*)> function, OctreeNode<AdditionalCellAttributes>* root, size_t max_level = std::numeric_limits<size_t>::max()) {
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

protected:
    void construct_global_tree_part() {
        RelearnException::check(root == nullptr, "root was not null in the construction of the global state!");

        SpaceFillingCurve<Morton> space_curve{ static_cast<uint8_t>(level_of_branch_nodes) };

        const auto my_rank = MPIWrapper::get_my_rank();

        const auto num_cells_per_dimension = 1 << level_of_branch_nodes; // (2^level_of_branch_nodes)

        const auto& cell_length = (xyz_max - xyz_min) / num_cells_per_dimension;

        const auto cell_length_x = cell_length.get_x();
        const auto cell_length_y = cell_length.get_y();
        const auto cell_length_z = cell_length.get_z();

        OctreeNode<AdditionalCellAttributes>* local_root = OctreeNode<AdditionalCellAttributes>::create();
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

        std::stack<std::pair<OctreeNode<AdditionalCellAttributes>*, Vec3s>> stack{};
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

protected:
    // Root of the tree
    OctreeNode<AdditionalCellAttributes>* root{ nullptr };

    std::vector<OctreeNode<AdditionalCellAttributes>*> branch_nodes{};

    std::vector<OctreeNode<AdditionalCellAttributes>*> all_leaf_nodes{};
};
