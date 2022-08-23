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
#include "OctreeNode.h"
#include "SpaceFillingCurve.h"
#include "Types.h"
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"
#include "neurons/Neurons.h"
#include "neurons/SignalType.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/models/SynapticElements.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Stack.h"
#include "util/Timers.h"
#include "util/Vec3.h"

#include <climits>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <stack>
#include <utility>
#include <variant>
#include <vector>

class Neurons;
class Partition;
class SynapticElements;

/**
 * This type represents the interface of the (spatial) Octree.
 */
class Octree {
public:
    friend class Partition;

    using box_size_type = RelearnTypes::box_size_type;
    using position_type = RelearnTypes::position_type;

    using AccessEpochsStarted = std::vector<bool>;

    /**
     * @brief Constructs a new Octree with the the given size and constructs the "internal" part up to and including the level_of_branch_nodes
     * @param xyz_min The minimum positions of this octree
     * @param xyz_max The maximum positions of this octree
     * @param level_of_branch_nodes The level at which the branch nodes (that are exchanged via MPI) are
     * @exception Throws a RelearnException if xyz_min is not componentwise smaller than xyz_max
     */
    Octree(const box_size_type& xyz_min, const box_size_type& xyz_max, const std::uint16_t level_of_branch_nodes)
        : level_of_branch_nodes(level_of_branch_nodes) {
        set_size(xyz_min, xyz_max);
    }

    virtual ~Octree() = default;

    Octree(const Octree& other) = delete;
    Octree(Octree&& other) = delete;

    Octree& operator=(const Octree& other) = delete;
    Octree& operator=(Octree&& other) = delete;

    /**
     * @brief Returns the minimum position in the Octree
     * @return The minimum position in the Octree
     */
    [[nodiscard]] const box_size_type& get_xyz_min() const noexcept {
        return xyz_min;
    }

    /**
     * @brief Returns the maximum position in the Octree
     * @return The maximum position in the Octree
     */
    [[nodiscard]] const box_size_type& get_xyz_max() const noexcept {
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
    [[nodiscard]] std::uint16_t get_level_of_branch_nodes() const noexcept {
        return level_of_branch_nodes;
    }

    /**
     * @brief Inserts a neuron with the specified id and the specified position into the octree.
     * @param position The position of the new neuron
     * @param neuron_id The id of the new neuron, < Constants::uninitialized (only use for actual neurons, virtual neurons are inserted automatically)
     * @exception Throws a RelearnException if one of the following happens:
     *      (a) The position is not within the octree's boundaries
     *      (b) neuron_id >= Constants::uninitialized
     *      (c) Allocating a new object in the shared memory window fails
     *      (d) Something went wrong within the insertion
     * @return A pointer to the newly created and inserted node
     */
    virtual void insert(const box_size_type& position, const NeuronID& neuron_id) = 0;

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

    virtual void* get_branch_node_pointer(size_t index) = 0;

protected:
    // Set simulation box size of the tree
    void set_size(const box_size_type& min, const box_size_type& max) {
        const auto& [min_x, min_y, min_z] = min;
        const auto& [max_x, max_y, max_z] = max;

        RelearnException::check(min_x <= max_x, "Octree::set_size: x was not ok");
        RelearnException::check(min_y <= max_y, "Octree::set_size: y was not ok");
        RelearnException::check(min_z <= max_z, "Octree::set_size: z was not ok");

        xyz_min = min;
        xyz_max = max;
    }

private:
    // Two points describe simulation box size of the tree
    box_size_type xyz_min{ 0 };
    box_size_type xyz_max{ 0 };

    std::uint16_t level_of_branch_nodes{ std::numeric_limits<std::uint16_t>::max() };
};

/**
 * This type represents the (spatial) Octree in which the neurons are organised.
 * It offers general informations about the structure, the functionality to insert new neurons,
 * update from the bottom up, and synchronize parts with MPI.
 * It is templated by the algorithm with which it is used. The type must provide
 * Algorithm::AdditionalCellAttributes - will be used to template the Cell
 * void Algorithm::update_functor(OctreeNode<Algorithm::AdditionalCellAttributes>*)
 */
template <typename Algorithm>
class OctreeImplementation : public Octree {
public:
    using AdditionalCellAttributes = typename Algorithm::AdditionalCellAttributes;

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
        StackElement(OctreeNode<AdditionalCellAttributes>* octree_node, const size_t depth_in_tree)
            : ptr(octree_node)
            , depth(depth_in_tree) {
            RelearnException::check(octree_node != nullptr, "StackElement::StackElement: octree_node was nullptr");
            RelearnException::check(depth_in_tree < Constants::uninitialized, "StackElement::StackElement: depth_in_tree was too large: {}", depth_in_tree);
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
            RelearnException::check(!already_visited, "StackElement::set_visited: element is already visited");
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
     * @brief Constructs a new OctreeImplementation with the the given size and constructs the "internal" part up to and including the level_of_branch_nodes
     * @param xyz_min The minimum positions of this octree
     * @param xyz_max The maximum positions of this octree
     * @param level_of_branch_nodes The level at which the branch nodes (that are exchanged via MPI) are
     * @exception Throws a RelearnException if xyz_min is not componentwise smaller than xyz_max
     */
    OctreeImplementation(const box_size_type& xyz_min, const box_size_type& xyz_max, const std::uint16_t level_of_branch_nodes)
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
     * @brief Returns the branch node with the specified local id
     * @param local_id The (1d-) id of the branch node that should be set
     * @exception Throws a RelearnException if local_id is too large
     * @return The branch node with the specified local id. Ownership is not transfered
     */
    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_local_root(const size_t local_id) {
        RelearnException::check(local_id < branch_nodes.size(), "Octree::get_local_root: local_id was {}", local_id);
        OctreeNode<AdditionalCellAttributes>* local_tree = branch_nodes[local_id];
        return local_tree;
    }

    /**
     * @brief Returns the number of branch nodes (that are exchanged via MPI)
     * @return The number of branch nodes (that are exchanged via MPI)
     */
    [[nodiscard]] size_t get_num_local_trees() const noexcept override {
        return branch_nodes.size();
    }

    /**
     * @brief Get all local branch nodes
     * @return a vector the local branch nodes
     */
    [[nodiscard]] std::vector<const OctreeNode<AdditionalCellAttributes>*> get_local_branch_nodes() const {
        std::vector<const OctreeNode<AdditionalCellAttributes>*> result{};
        for (const auto* node : branch_nodes) {
            if (!node->is_local()) {
                continue;
            }
            result.emplace_back(node);
        }

        return result;
    }

    /**
     * @brief Get all local branch nodes
     * @return a vector the local branch nodes
     */
    [[nodiscard]] std::vector<OctreeNode<AdditionalCellAttributes>*> get_local_branch_nodes() {
        std::vector<OctreeNode<AdditionalCellAttributes>*> result{};
        for (auto* node : branch_nodes) {
            if (!node->is_local()) {
                continue;
            }
            result.emplace_back(node);
        }

        return result;
    }

    /**
     * @brief This function updates the Octree starting from max_level. Is is required that it only visits inner nodes
     * @param max_level The maximum level (inclusive) on which the nodes should be updated
     * @exception Throws a RelearnException if the functor throws
     */
    void update_from_level(const size_t max_level) override {
        tree_walk_postorder(Algorithm::update_functor, root, max_level);
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

            tree_walk_postorder(Algorithm::update_functor, local_tree);
        }
    }

    /**
     * @brief Gathers all leaf nodes and makes them available via get_leaf_nodes
     * @param num_neurons The number of neurons
     */
    void initializes_leaf_nodes(const size_t num_neurons) override {
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

                    if (const auto neuron_id = child->get_cell_neuron_id(); !child->is_parent() && (neuron_id.is_virtual() || !neuron_id.is_initialized())) {
                        continue;
                    }

                    stack.emplace(child);
                }
            } else {
                const auto neuron_id = node->get_cell_neuron_id();
                RelearnException::check(neuron_id.get_neuron_id() < leaf_nodes.size(), "Octree::initializes_leaf_nodes: Neuron id was too large for leaf nodes: {}", neuron_id);
                leaf_nodes[neuron_id.get_neuron_id()] = node;
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
        RelearnException::check(num_local_trees < static_cast<size_t>(std::numeric_limits<int>::max()),
            "Octree::synchronize_local_trees: Too many branch nodes: {}", num_local_trees);
        MPIWrapper::all_gather_inline(std::span{ exchange_branch_nodes.data(), num_local_trees });

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

        const auto level_of_branch_nodes = get_level_of_branch_nodes();

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
    void insert_local_tree(OctreeNode<AdditionalCellAttributes>* node_to_insert, const size_t index_1d) {
        RelearnException::check(index_1d < branch_nodes.size(), "Octree::insert_local_tree: local_id was {}", index_1d);
        RelearnException::check(node_to_insert != nullptr, "Octree::insert_local_tree: node_to_insert is nullptr");
        RelearnException::check(node_to_insert->is_parent(), "Octree::insert_local_tree: Cannot insert an empty node");
        *branch_nodes[index_1d] = *node_to_insert;
    }

    /**
     * @brief Returns a constant reference to all leaf nodes
     *      The reference is never invalidated
     * @return All leaf nodes
     */
    [[nodiscard]] const std::vector<OctreeNode<AdditionalCellAttributes>*>& get_leaf_nodes() const noexcept {
        return all_leaf_nodes;
    }

    virtual void* get_branch_node_pointer(size_t index) {
        RelearnException::check(index >= branch_nodes.size(), "OctreeImplementation::get_branch_node_pointer(): index ({}) is larger than or equal to the number of branch nodes ({}).", index, branch_nodes.size());
        return branch_nodes[index];
    }

    /**
     * @brief Inserts a neuron at the specified position with the neuron id and the position
     * @param position The position of the neuron
     * @param neuron_id The local neuron id
     * @exception Throws a RelearnException if the root is nullptr, the position is not in the box,
     *      neuron_id is uninitialized, or OctreeNode::insert throws a RelearnException
     */
    void insert(const box_size_type& position, const NeuronID& neuron_id) override {
        RelearnException::check(root != nullptr, "Octree::insert: root was nullptr");

        const auto& xyz_min = get_xyz_min();
        const auto& xyz_max = get_xyz_max();

        const auto& [min_x, min_y, min_z] = xyz_min;
        const auto& [max_x, max_y, max_z] = xyz_max;
        const auto& [pos_x, pos_y, pos_z] = position;

        RelearnException::check(min_x <= pos_x && pos_x <= max_x, "Octree::insert: x was not in range: {} vs [{}, {}]", pos_x, min_x, max_x);
        RelearnException::check(min_y <= pos_y && pos_y <= max_y, "Octree::insert: y was not in range: {} vs [{}, {}]", pos_y, min_y, max_y);
        RelearnException::check(min_z <= pos_z && pos_z <= max_z, "Octree::insert: z was not in range: {} vs [{}, {}]", pos_z, min_z, max_z);

        RelearnException::check(neuron_id.is_initialized(), "Octree::insert: neuron_id {} was uninitialized", neuron_id);

        auto* res = root->insert(position, neuron_id, get_level_of_branch_nodes());
        RelearnException::check(res != nullptr, "Octree::insert: res was nullptr");

        // LogFiles::print_message_rank(-1, "Inserted {:X} into the octree", reinterpret_cast<unsigned long long int>(res));
    }

    void print() {
        std::stringstream ss{};

        ss << "I'm rank " << MPIWrapper::get_my_rank() << '\n';
        ss << "root: " << root << '\n';

        for (auto* child : root->get_children()) {
            ss << "\tchild: " << child << '\n';
            for (auto* grandchild : child->get_children()) {
                ss << "\t\tgrandchild: " << grandchild << '\n';
            }
        }

        LogFiles::print_message_rank(-1, ss.str());
    }

protected:
    void tree_walk_postorder(std::function<void(OctreeNode<AdditionalCellAttributes>*)> function,
        OctreeNode<AdditionalCellAttributes>* root, const size_t max_level = std::numeric_limits<size_t>::max()) {
        RelearnException::check(root != nullptr, "Octree::tree_walk_postorder: octree was nullptr");

        if (max_level > 2) {
            tree_walk_postorder_parallel(function, root, max_level);
            return;
        }

        walk_post_order(root, 0, max_level, function);
    }

    void tree_walk_postorder_parallel(std::function<void(OctreeNode<AdditionalCellAttributes>*)> function,
        OctreeNode<AdditionalCellAttributes>* root, const size_t max_level = std::numeric_limits<size_t>::max()) {

        std::vector<OctreeNode<AdditionalCellAttributes>*> subtrees{};
        std::stack<OctreeNode<AdditionalCellAttributes>*> tree_upper_part{};

        tree_upper_part.emplace(root);

        for (const auto& root_child : root->get_children()) {
            if (root_child == nullptr) {
                continue;
            }

            tree_upper_part.emplace(root_child);

            for (const auto& root_child_child : root_child->get_children()) {
                if (root_child_child == nullptr) {
                    continue;
                }

                tree_upper_part.emplace(root_child_child);
                subtrees.emplace_back(root_child_child);
            }
        }

#pragma omp parallel for shared(subtrees, max_level, function) default(none)
        for (auto i = 0; i < subtrees.size(); i++) {
            auto* local_tree_root = subtrees[i];

            walk_post_order(local_tree_root, 2, max_level, function);
        }

        while (!tree_upper_part.empty()) {
            auto* node = tree_upper_part.top();
            tree_upper_part.pop();

            function(node);
        }
    }

    void walk_post_order(OctreeNode<AdditionalCellAttributes>* local_tree_root,
        const size_t current_level, const size_t max_level, std::function<void(OctreeNode<AdditionalCellAttributes>*)> function) {
        std::stack<StackElement> stack{};

        // Push node onto stack
        stack.emplace(local_tree_root, current_level);

        while (!stack.empty()) {
            // Get top-of-stack node
            auto& current_element = stack.top();
            const auto current_depth = current_element.get_depth_in_tree();
            auto* current_octree_node = current_element.get_octree_node();

            // Node should be visited now?
            if (current_element.get_visited()) {
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

    void construct_global_tree_part() {
        RelearnException::check(root == nullptr, "Octree::construct_global_tree_part: root was not null");

        const auto level_of_branch_nodes = get_level_of_branch_nodes();

        const auto number_nodes_in_upper_part = (std::pow(8.0, level_of_branch_nodes + 1) - 1.0) / 7.0;
        const auto number_nodes_cast = static_cast<size_t>(number_nodes_in_upper_part);

        upper_portion_holder.reserve(number_nodes_cast);

        SpaceFillingCurve<Morton> space_curve{ static_cast<uint8_t>(level_of_branch_nodes) };

        const auto my_rank = MPIWrapper::get_my_rank();

        const auto num_cells_per_dimension = 1ULL << level_of_branch_nodes; // (2^level_of_branch_nodes)

        const auto& xyz_min = get_xyz_min();
        const auto& xyz_max = get_xyz_max();

        const auto& box_size = (xyz_max - xyz_min);

        auto& root_node = upper_portion_holder.emplace_back();
        root_node.set_cell_size(xyz_min, xyz_max);
        root_node.set_cell_neuron_id(NeuronID::virtual_id());
        root_node.set_cell_neuron_position(xyz_min + (box_size / 2.0));
        root_node.set_rank(my_rank);
        root_node.set_level(0);

        Stack<OctreeNode<AdditionalCellAttributes>*> stack_upper_part{ number_nodes_cast };
        stack_upper_part.emplace_back(&root_node);

        while (!stack_upper_part.empty()) {
            const auto current_node = stack_upper_part.pop_back();
            const auto current_level = current_node->get_level();

            if (current_level == level_of_branch_nodes) {
                continue;
            }

            for (unsigned char child_id = 0; child_id < Constants::number_oct; child_id++) {
                const auto& [child_min, child_max] = current_node->get_cell().get_size_for_octant(child_id);

                auto& child_node = upper_portion_holder.emplace_back();
                child_node.set_cell_size(child_min, child_max);
                child_node.set_cell_neuron_id(NeuronID::virtual_id());
                child_node.set_cell_neuron_position((child_max + child_min) / 2.0);
                child_node.set_rank(my_rank);
                child_node.set_level(current_level + 1);

                current_node->set_child(&child_node, child_id);

                if (current_level + 1 < level_of_branch_nodes) {
                    stack_upper_part.emplace_back(&child_node);
                }
            }
        }

        root = &root_node;

        Stack<std::pair<OctreeNode<AdditionalCellAttributes>*, Vec3s>> stack{ number_nodes_cast };
        stack.emplace_back(root, Vec3s{ 0, 0, 0 });

        while (!stack.empty()) {
            const auto [ptr, index3d] = stack.pop_back();

            if (!ptr->is_parent()) {
                const auto index1d = space_curve.map_3d_to_1d(index3d);
                branch_nodes[index1d] = ptr;
                continue;
            }

            for (size_t id = 0; id < Constants::number_oct; id++) {
                auto child_node = ptr->get_child(id);

                const auto larger_x = ((id & 1ULL) == 0) ? 0ULL : 1ULL;
                const auto larger_y = ((id & 2ULL) == 0) ? 0ULL : 1ULL;
                const auto larger_z = ((id & 4ULL) == 0) ? 0ULL : 1ULL;

                const Vec3s offset{ larger_x, larger_y, larger_z };
                const Vec3s pos = (index3d * 2) + offset;
                stack.emplace_back(child_node, pos);
            }
        }
    }

private:
    // Root of the tree
    OctreeNode<AdditionalCellAttributes>* root{ nullptr };

    std::vector<OctreeNode<AdditionalCellAttributes>> upper_portion_holder{};
    std::vector<OctreeNode<AdditionalCellAttributes>*> branch_nodes{};
    std::vector<OctreeNode<AdditionalCellAttributes>*> all_leaf_nodes{};
};
