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
#include "Types.h"
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"
#include "structure/OctreeNode.h"
#include "structure/SpaceFillingCurve.h"
#include "util/RelearnException.h"
#include "util/Stack.h"
#include "util/Timers.h"
#include "util/Vec3.h"

#include <climits>
#include <cstdint>
#include <functional>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

class Partition;

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
     * @brief Gathers all leaf nodes and makes them available via get_leaf_nodes
     * @param num_neurons The number of neurons
     */
    virtual void initializes_leaf_nodes(RelearnTypes::number_neurons_type num_neurons) = 0;

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
 * It is templated by the additional cell attributes that the algorithm will need the cell to have.
 */
template <typename AdditionalCellAttributes>
class OctreeImplementation : public Octree {
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

        const auto num_local_trees = 1ULL << (3U * level_of_branch_nodes);
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
     * @return All local branch nodes
     */
    [[nodiscard]] std::vector<const OctreeNode<AdditionalCellAttributes>*> get_local_branch_nodes() const {
        std::vector<const OctreeNode<AdditionalCellAttributes>*> result{};
        result.reserve(4);

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
     * @return All local branch nodes
     */
    [[nodiscard]] std::vector<OctreeNode<AdditionalCellAttributes>*> get_local_branch_nodes() {
        std::vector<OctreeNode<AdditionalCellAttributes>*> result{};
        result.reserve(4);

        for (auto* node : branch_nodes) {
            if (!node->is_local()) {
                continue;
            }
            result.emplace_back(node);
        }

        return result;
    }

    /**
     * @brief Gathers all leaf nodes and makes them available via get_leaf_nodes
     * @param num_neurons The number of neurons
     */
    void initializes_leaf_nodes(const RelearnTypes::number_neurons_type num_neurons) override {
        std::vector<OctreeNode<AdditionalCellAttributes>*> leaf_nodes(num_neurons);

        Stack<OctreeNode<AdditionalCellAttributes>*> stack{ num_neurons };
        stack.emplace_back(root);

        while (!stack.empty()) {
            OctreeNode<AdditionalCellAttributes>* node = stack.pop_back();

            if (node->is_leaf()) {
                const auto neuron_id = node->get_cell_neuron_id();
                RelearnException::check(neuron_id.get_neuron_id() < leaf_nodes.size(), "Octree::initializes_leaf_nodes: Neuron id was too large for leaf nodes: {}", neuron_id);

                leaf_nodes[neuron_id.get_neuron_id()] = node;
                continue;
            }

            for (auto* child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                if (const auto neuron_id = child->get_cell_neuron_id(); !child->is_parent() && (neuron_id.is_virtual() || !neuron_id.is_initialized())) {
                    continue;
                }

                stack.emplace_back(child);
            }
        }

        RelearnException::check(leaf_nodes.size() == num_neurons, "Octree::initializes_leaf_nodes: Less number of leaf nodes than number of local neurons {} != {}", leaf_nodes.size(), num_neurons);

        for (const auto neuron_id : NeuronID::range(num_neurons)) {
            const auto& node = leaf_nodes[neuron_id.get_neuron_id()];
            RelearnException::check(node != nullptr, "Octree::initializes_leaf_nodes: Leaf node {} is null", neuron_id);
            RelearnException::check(node->is_leaf(), "Octree::initializes_leaf_nodes: Leaf node {} is not a leaf node", neuron_id);
            RelearnException::check(node->is_local(), "Octree::initializes_leaf_nodes: Leaf node {} is not local", neuron_id);
            RelearnException::check(node->get_cell().get_neuron_id() == neuron_id, "Octree::initializes_leaf_nodes: Leaf node {} has wrong neuron id {}", neuron_id, node->get_cell().get_neuron_id());
        }

        all_leaf_nodes = std::move(leaf_nodes);
    }

    /**
     * @brief Synchronizes the octree with all MPI ranks
     */
    void synchronize_tree() {
        // Update my local trees bottom-up
        update_local_trees();

        // Exchange the local trees
        synchronize_local_trees();
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

    /**
     * @brief Returns the branch node with the (global) index, cast to a void*
     * @param index The global index of the requested branch node
     * @exception Throws a RelearnException if index is larger than or equal to the number of branch nodes
     * @return The requested branch node
     */
    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_branch_node_pointer(size_t index) {
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
    /**
     * @brief Constructs the upper portion of the tree, i.e., all nodes at depths [0, level_of_branch_nodes].
     */
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

        auto& root_node = upper_portion_holder.emplace_back();
        root_node.set_cell_size(xyz_min, xyz_max);
        root_node.set_cell_neuron_id(NeuronID::virtual_id());
        root_node.set_cell_neuron_position(xyz_min.get_midpoint(xyz_max));
        root_node.set_rank(MPIRank(my_rank));
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
                child_node.set_cell_neuron_position(child_min.get_midpoint(child_max));
                child_node.set_rank(MPIRank(my_rank));
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

    /**
     * @brief Updates all local (!) branch nodes and their induced subtrees.
     * @exception Throws a RelearnException if the functor throws
     */
    void update_local_trees() {
        Timers::start(TimerRegion::UPDATE_LOCAL_TREES);
        for (auto* local_tree : branch_nodes) {
            if (!local_tree->is_local()) {
                continue;
            }

            update_tree_parallel(local_tree);
        }
        Timers::stop_and_add(TimerRegion::UPDATE_LOCAL_TREES);
    }

    /**
     * @brief Synchronizes all (locally) updated branch nodes with all other MPI ranks
     */
    void synchronize_local_trees() {
        Timers::start(TimerRegion::EXCHANGE_BRANCH_NODES);
        const auto number_branch_nodes = branch_nodes.size();

        // Copy local trees' root nodes to correct positions in receive buffer
        std::vector<OctreeNode<AdditionalCellAttributes>> exchange_branch_nodes(number_branch_nodes);
        for (size_t i = 0; i < number_branch_nodes; i++) {
            exchange_branch_nodes[i] = *branch_nodes[i];
        }

        // Allgather in-place branch nodes from every rank
        const auto number_local_branch_nodes = number_branch_nodes / MPIWrapper::get_num_ranks();
        RelearnException::check(number_local_branch_nodes < static_cast<size_t>(std::numeric_limits<int>::max()),
            "OctreeImplementation::synchronize_local_trees: Too many branch nodes: {}", number_local_branch_nodes);
        MPIWrapper::all_gather_inline(std::span{ exchange_branch_nodes.data(), number_local_branch_nodes });

        Timers::stop_and_add(TimerRegion::EXCHANGE_BRANCH_NODES);

        Timers::start(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);
        for (size_t i = 0; i < number_branch_nodes; i++) {
            *branch_nodes[i] = exchange_branch_nodes[i];
        }
        Timers::stop_and_add(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);

        Timers::start(TimerRegion::UPDATE_GLOBAL_TREE);
        if (const auto level_of_branch_nodes = get_level_of_branch_nodes(); level_of_branch_nodes > 0) {
            // Only update whenever there are other branches to update
            // The nodes at level_of_branch_nodes are already updated (by other MPI ranks)
            update_tree_parallel(root, level_of_branch_nodes - 1);
        }
        Timers::stop_and_add(TimerRegion::UPDATE_GLOBAL_TREE);
    }

    /**
     * @brief Updates the tree induced by local_tree_root until the desired level.
     *      Uses OctreeNode::get_level() to determine the depth. The nodes at that depth are still updated, but not their children.
     *      Potentially updates in parallel based on the depth of the updates.
     * @param local_tree_root The root of the tree from where to update
     * @param max_depth The depth where the updates shall stop
     * @exception Throws a RelearnException if local_tree_root is nullptr or if max_depth is smaller than the depth of local_tree_root
     */
    void update_tree_parallel(OctreeNode<AdditionalCellAttributes>* local_tree_root, const std::uint16_t max_depth = std::numeric_limits<std::uint16_t>::max()) {
        RelearnException::check(local_tree_root != nullptr, "OctreeImplementation::update_tree_parallel: local_tree_root was nullptr");
        RelearnException::check(local_tree_root->get_level() <= max_depth, "OctreeImplementation::update_tree_parallel: The root had a larger depth than max_depth.");

        if (const auto update_height = max_depth - local_tree_root->get_level(); update_height < 3) {
            // If the update concerns less than 3 levels, update serially
            update_tree(local_tree_root, max_depth);
            return;
        }

        // Gather all subtrees two levels down from the current node, update the induced trees in parallel, and then update the upper portion serially

        constexpr auto maximum_number_subtrees = 64;
        std::vector<OctreeNode<AdditionalCellAttributes>*> subtrees{};
        subtrees.reserve(maximum_number_subtrees);

        constexpr auto maximum_number_nodes = 64 + 8 + 1;
        Stack<OctreeNode<AdditionalCellAttributes>*> tree_upper_part{ maximum_number_nodes };
        tree_upper_part.emplace_back(local_tree_root);

        for (const auto& root_child : local_tree_root->get_children()) {
            if (root_child == nullptr) {
                continue;
            }

            tree_upper_part.emplace_back(root_child);

            for (const auto& root_child_child : root_child->get_children()) {
                if (root_child_child == nullptr) {
                    continue;
                }

                tree_upper_part.emplace_back(root_child_child);
                subtrees.emplace_back(root_child_child);
            }
        }

#pragma omp parallel for shared(subtrees, max_depth) default(none)
        for (auto i = 0; i < subtrees.size(); i++) {
            auto* local_tree_root = subtrees[i];
            update_tree(local_tree_root, max_depth);
        }

        while (!tree_upper_part.empty()) {
            auto* node = tree_upper_part.top();
            tree_upper_part.pop();

            if (node->is_parent()) {
                node->update();
            }
        }
    }

    /**
     * @brief Updates the tree induced by local_tree_root until the desired level.
     *      Uses OctreeNode::get_level() to determine the depth. The nodes at that depth are still updated, but not their children.
     * @param local_tree_root The root of the tree from where to update
     * @param max_depth The depth where the updates shall stop
     * @exception Throws a RelearnException if local_tree_root is nullptr or if max_depth is smaller than the depth of local_tree_root
     */
    void update_tree(OctreeNode<AdditionalCellAttributes>* local_tree_root, const std::uint16_t max_depth) {
        struct StackElement {
        private:
            OctreeNode<AdditionalCellAttributes>* ptr{ nullptr };

            // True if node has been on stack already
            // twice and can be visited now
            bool already_visited{ false };

        public:
            /**
             * @brief Constructs a new object that holds the given node, which is marked as not already visited
             * @param octree_node The node that should be visited, not nullptr
             * @exception Throws a RelearnException if octree_node is nullptr
             */
            explicit StackElement(OctreeNode<AdditionalCellAttributes>* octree_node)
                : ptr(octree_node) {
                RelearnException::check(octree_node != nullptr, "StackElement::StackElement: octree_node was nullptr");
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
            [[nodiscard]] bool was_already_visited() const noexcept {
                return already_visited;
            }
        };

        RelearnException::check(local_tree_root != nullptr, "OctreeImplementation::update_tree: local_tree_root is nullptr.");
        RelearnException::check(local_tree_root->get_level() <= max_depth, "OctreeImplementation::update_tree: The root had a larger depth than max_depth.");

        Stack<StackElement> stack{};
        stack.emplace_back(local_tree_root);

        while (!stack.empty()) {
            auto& current_element = stack.top();
            auto* current_octree_node = current_element.get_octree_node();

            if (current_element.was_already_visited()) {
                // Make sure that the element was visited before, i.e., its children are processed
                if (current_octree_node->is_parent()) {
                    // Don't update leaf nodes, they were updated before
                    current_octree_node->update();
                }

                stack.pop();
                continue;
            }

            // Mark node to be visited next time now, because it's a reference and will change once we push the other elements
            current_element.set_visited();

            const auto current_depth = current_octree_node->get_level();
            if (current_depth >= max_depth) {
                // We're at the border of where we want to update, so don't push children
                if (current_octree_node->is_parent()) {
                    // Don't update leaf nodes, they were updated before
                    current_octree_node->update();
                }

                stack.pop();
                continue;
            }

            for (auto* child : current_octree_node->get_children()) {
                if (child == nullptr) {
                    continue;
                }
                stack.emplace_back(child);
            }
        } /* while */
    }

private:
    // Root of the tree
    OctreeNode<AdditionalCellAttributes>* root{ nullptr };

    std::vector<OctreeNode<AdditionalCellAttributes>> upper_portion_holder{};
    std::vector<OctreeNode<AdditionalCellAttributes>*> branch_nodes{};
    std::vector<OctreeNode<AdditionalCellAttributes>*> all_leaf_nodes{};
};
