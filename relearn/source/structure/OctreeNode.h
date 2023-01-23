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
#include "neurons/helper/RankNeuronId.h"
#include "structure/Cell.h"
#include "util/MemoryHolder.h"
#include "util/MPIRank.h"
#include "util/RelearnException.h"
#include "util/Stack.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>

/**
 * This class serves as the basic building blocks of the Octree.
 * Each object has up to Constants::number_oct children (can be nullptr) and a Cell which summarizes the relevant biological aspects.
 * Additionally, an object stores its its MPI rank and whether or not it is an inner node.
 */
template <typename AdditionalCellAttributes>
class OctreeNode {
public:
    using OctreeNodePtr = OctreeNode<AdditionalCellAttributes>*;

    using position_type = typename Cell<AdditionalCellAttributes>::position_type;
    using counter_type = typename Cell<AdditionalCellAttributes>::counter_type;
    using box_size_type = typename Cell<AdditionalCellAttributes>::box_size_type;

    constexpr static bool has_excitatory_dendrite = AdditionalCellAttributes::has_excitatory_dendrite;
    constexpr static bool has_inhibitory_dendrite = AdditionalCellAttributes::has_inhibitory_dendrite;
    constexpr static bool has_excitatory_axon = AdditionalCellAttributes::has_excitatory_axon;
    constexpr static bool has_inhibitory_axon = AdditionalCellAttributes::has_inhibitory_axon;

    /**
     * @brief Returns the MPI rank this node belongs to
     * @return The MPI rank
     */
    [[nodiscard]] constexpr MPIRank get_mpi_rank() const noexcept {
        return rank;
    }

    /**
     * @brief Returns a flag that indicates if this node is an inner node or a leaf node
     * @return True iff it is an inner node
     */
    [[nodiscard]] constexpr bool is_parent() const noexcept {
        return parent;
    }

    /**
     * @brief Returns a flag that indicates if this node is an inner node or a leaf node
     * @return True iff it is a leaf node
     */
    [[nodiscard]] constexpr bool is_leaf() const noexcept {
        return !parent;
    }

    /**
     * @brief Checks if the specified neuron is stored within *this* node
     * @param rank_neuron_id The MPI rank and neuron id in question
     * @return True iff the specified neuron is stored in this node
    */
    [[nodiscard]] constexpr bool contains(const RankNeuronId& rank_neuron_id) const {
        if (is_parent()) {
            return false;
        }

        const auto& [rank, id] = rank_neuron_id;

        if (rank != this->rank) {
            return false;
        }

        return cell.get_neuron_id() == id;
    }

    /**
     * @brief Returns a constant view on the associated child nodes. This reference is not invalidated by calls to other methods
     * @return A constant view on the associated child nodes
     */
    [[nodiscard]] constexpr const std::array<OctreeNodePtr, Constants::number_oct>& get_children() const noexcept {
        return children;
    }

    /**
     * @brief Returns the child node with the requested id (id calculation base on Cell::get_octant_for_position)
     * @exception Throws a RelearnException if idx >= Constants::number_oct
     * @return The associated child
     */
    [[nodiscard]] OctreeNodePtr get_child(const size_t idx) const {
        RelearnException::check(idx < Constants::number_oct, "OctreeNode::get_child const: idx was: {}", idx);
        // NOLINTNEXTLINE
        return children[idx];
    }

    /**
     * @brief Returns the child node with the requested id (id calculation base on Cell::get_octant_for_position)
     * @exception Throws a RelearnException if idx >= Constants::number_oct
     * @return The associated child
     */
    [[nodiscard]] OctreeNodePtr get_child(const size_t idx) {
        RelearnException::check(idx < Constants::number_oct, "OctreeNode::get_child: idx was: {}", idx);
        // NOLINTNEXTLINE
        return children[idx];
    }

    /**
     * @brief Returns a constant view on the associated cell. This reference is not invalidated by calls to other methods
     * @return A constant view on the associated cell
     */
    [[nodiscard]] constexpr const Cell<AdditionalCellAttributes>& get_cell() const noexcept {
        return cell;
    }

    /**
     * @brief Returns a flag that indicates if this object belongs to the current MPI process.
     *      Achieves so by calling MPIWrapper::get_my_rank()
     * @exception Throws a RelearnException if the MPIWrapper is not properly initialized
     * @return True iff this object belongs to the current MPI process
     */
    [[nodiscard]] bool is_local() const {
        return rank == MPIWrapper::get_my_rank();
    }

    /**
     * @brief Inserts a neuron with the specified id and the specified position into the subtree that is induced by this object.
     * @param position The position of the new neuron
     * @param neuron_id The id of the new neuron (can be Constants::uninitialized to indicate a virtual neuron), <= Constants::uninitialized
     * @param level_of_branch_nodes The level of the branch nodes in the Octree, is optional
     * @exception Throws a RelearnException if one of the following happens:
     *      (a) The position is not within the cell's boundaries
     *      (b) neuron_id > Constants::uninitialized
     *      (c) Allocating a new object in the shared memory window fails
     *      (d) Something went wrong within the insertion
     * @return A pointer to the newly created and inserted node
     */
    [[nodiscard]] OctreeNodePtr insert(const box_size_type& position, const NeuronID& neuron_id, [[maybe_unused]] std::uint16_t level_of_branch_nodes = 0) {
        const auto& [cell_xyz_min, cell_xyz_max] = cell.get_size();
        const auto is_in_box = position.check_in_box(cell_xyz_min, cell_xyz_max);

        RelearnException::check(is_in_box, "OctreeNode::insert: position is not in box: {} in [{}, {}]", position, cell_xyz_min, cell_xyz_max);
        RelearnException::check(neuron_id.is_initialized(), "OctreeNode::insert, neuron_id is not initialized");

        const auto my_rank = MPIWrapper::get_my_rank();

        OctreeNodePtr parent_node = nullptr;
        for (OctreeNodePtr current_node = this; nullptr != current_node;) {
            /**
             * My parent already exists.
             * Calc which child to follow, i.e., determine octant
             */
            const auto new_position_octant = current_node->get_cell().get_octant_for_position(position);

            parent_node = current_node;
            current_node = current_node->get_child(new_position_octant);
        }

        // Now we know the furthest-down node and the respective octant where to insert the position
        RelearnException::check(parent_node != nullptr, "OctreeNode::insert: parent_node is nullptr");

        if (parent_node->is_leaf()) {
            /**
             * Found my octant, but I'm the very first child of that node.
             * I.e., the node is a leaf.
             */

            if (parent_node->get_cell_neuron_id().is_virtual() && neuron_id != parent_node->get_cell_neuron_id()) {
                /**
                 * The found parent node is virtual and can just be substituted,
                 * i.e., it was constructed while constructing the upper part to the branch nodes.
                 */
                parent_node->set_cell_neuron_id(neuron_id);
                parent_node->set_cell_neuron_position({ position });
                parent_node->set_rank(my_rank);
                return parent_node;
            }

            while (true) {
                // The found parent node contains a neuron; it has to be transformed to a virtual node and the neuron shifted one down
                const auto& parent_cell = parent_node->get_cell();
                const auto& parent_neuron_id = parent_cell.get_neuron_id();

                // Determine octant for neuron
                const auto& parent_position = parent_cell.get_neuron_position();
                RelearnException::check(parent_position.has_value(), "OctreeNode::insert: While building the octree, the cell doesn't have a position");

                const auto parent_own_octant = parent_cell.get_octant_for_position(parent_position.value());
                const auto& [minimum_position, maximum_position] = parent_cell.get_size_for_octant(parent_own_octant);

                // The child copies the parent node
                auto* new_node = MemoryHolder<AdditionalCellAttributes>::get_available(parent_node, parent_own_octant);
                new_node->set_cell_size(minimum_position, maximum_position);
                new_node->set_cell_neuron_position(parent_position);
                new_node->set_cell_neuron_id(parent_neuron_id);
                new_node->set_rank(parent_node->get_mpi_rank());
                new_node->set_level(parent_node->get_level() + 1);

                // Set the child and mark the parent as virtual
                parent_node->set_child(new_node, parent_own_octant);

                // If parent_node is larger than level_of_branch_nodes, it is in the RMA window and thus needs its offset
                const auto parent_node_offset = MemoryHolder<AdditionalCellAttributes>::get_offset(parent_node);
                parent_node->set_cell_neuron_id(NeuronID::virtual_id(parent_node_offset));

                if (const auto insert_octant = parent_cell.get_octant_for_position(position); insert_octant == parent_own_octant) {
                    // The moved parent and the position to insert still are in the same octant -> try again
                    parent_node = new_node;
                } else {
                    break;
                }
            }
        }

        // Now parent_node is virtual and the octant the new position will occupy is empty
        const auto new_position_octant = parent_node->get_cell().get_octant_for_position(position);

        auto* new_node_to_insert = MemoryHolder<AdditionalCellAttributes>::get_available(parent_node, new_position_octant);
        RelearnException::check(new_node_to_insert != nullptr, "OctreeNode::insert: new_node_to_insert is nullptr");

        parent_node->set_child(new_node_to_insert, new_position_octant);
        const auto parent_level = parent_node->get_level();

        const auto& [minimum_position, maximum_position] = parent_node->get_cell().get_size_for_octant(new_position_octant);
        new_node_to_insert->set_cell_size(minimum_position, maximum_position);
        new_node_to_insert->set_cell_neuron_position({ position });
        new_node_to_insert->set_cell_neuron_id(neuron_id);
        new_node_to_insert->set_rank(rank);
        new_node_to_insert->set_level(parent_level + 1);

        return new_node_to_insert;
    }

    /**
     * @brief Sets the associated MPI rank
     * @param new_rank The associated MPI rank, must be initialized
     * @exception Throws a RelearnException if new_rank is not initialized
     */
    void set_rank(const MPIRank new_rank) {
        RelearnException::check(new_rank.is_initialized(), "OctreeNode::set_rank: new_rank not initialized", new_rank);
        rank = new_rank;
    }

    /**
     * @brief Marks this node as a parent (an inner node)
     */
    constexpr void set_parent() noexcept {
        parent = true;
    }

    /**
     * @brief Sets the node as the child with the given index and updates the parent flag accordingly
     * @param node The new child node (can be nullptr)
     * @param idx The index of the child which shall be set, < Constants::number_oct
     * @exception Throws a RelearnException if idx >= Constants::number_oct
     */
    void set_child(OctreeNodePtr node, const size_t idx) {
        RelearnException::check(idx < Constants::number_oct, "OctreeNode::set_child: idx is {}", idx);
        // NOLINTNEXTLINE
        children[idx] = node;

        bool has_children = false;

        for (auto* child : children) {
            if (child != nullptr) {
                has_children = true;
            }
        }

        parent = has_children;
    }

    /**
     * @brief Returns the level of this node
     * @return The level
     */
    [[nodiscard]] constexpr std::uint16_t get_level() const noexcept {
        return level;
    }

    /**
     * @brief Sets the level of this node
     * @param new_level The new level of this node
     */
    constexpr void set_level(std::uint16_t new_level) noexcept {
        level = new_level;
    }

    /**
     * @brief Resets the current object:
     *      (a) The children are newly constructed with nullptr
     *      (b) The cell is newly constructed
     *      (c) level is -1
     *      (d) parent is false
     *      (e) rank is MPIRank::uninitialized_rank()
     */
    constexpr void reset() noexcept {
        children = std::array<OctreeNodePtr, Constants::number_oct>{ nullptr };
        cell = Cell<AdditionalCellAttributes>{};
        level = -1;
        parent = false;
        rank = MPIRank::uninitialized_rank();
    }

    /**
     * @brief Prints the octree node to the output stream
     * @param output_stream The output stream
     * @param octree_node The octree node to print
     * @return The output stream after printing the octree node
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const OctreeNode<AdditionalCellAttributes>& octree_node) {
        output_stream << "== OctreeNode (" << &octree_node << ") ==\n";

        output_stream << "  children[8]: ";
        for (const auto* const child : octree_node.get_children()) {
            output_stream << child << " ";
        }
        output_stream << '\n';

        output_stream << "  is_parent  : " << octree_node.is_parent() << "\n\n";
        output_stream << "  rank       : " << octree_node.get_rank() << "\n\n";
        output_stream << octree_node.get_cell();
        output_stream << '\n';

        return output_stream;
    }

private:
    std::array<OctreeNodePtr, Constants::number_oct> children{ nullptr };
    Cell<AdditionalCellAttributes> cell{};

    std::uint16_t level{ std::numeric_limits<std::uint16_t>::max() };
    bool parent{ false };

    MPIRank rank{ MPIRank::uninitialized_rank() }; // MPI rank who owns this octree node

public:
    /**
     * @brief Sets the optional position for both the excitatory and inhibitory positions in the associated cell
     * @param opt_position The optional position, can be empty
     */
    void set_cell_neuron_position(const std::optional<position_type>& opt_position) {
        cell.set_neuron_position(opt_position);
    }

    /**
     * @brief Sets the number of free excitatory and inhibitory dendrites in the associated cell
     * @param number_excitatory_dendrites The number of free excitatory dendrites
     * @param number_inhibitory_dendrites The number of free inhibitory dendrites
     */
    constexpr void set_cell_number_dendrites(const counter_type number_excitatory_dendrites, const counter_type number_inhibitory_dendrites) noexcept {
        cell.set_number_excitatory_dendrites(number_excitatory_dendrites);
        cell.set_number_inhibitory_dendrites(number_inhibitory_dendrites);
    }

    /**
     * @brief Sets the number of free excitatory and inhibitory axons in the associated cell
     * @param number_excitatory_axons The number of free excitatory axons
     * @param number_inhibitory_axons The number of free inhibitory axons
     */
    constexpr void set_cell_number_axons(const counter_type number_excitatory_axons, const counter_type number_inhibitory_axons) noexcept {
        cell.set_number_excitatory_axons(number_excitatory_axons);
        cell.set_number_inhibitory_axons(number_inhibitory_axons);
    }

    /**
     * @brief Sets the number of free excitatory dendrites int he associated cell
     * @param number_dendrites The number of free excitatory dendrites
     */
    constexpr void set_cell_number_excitatory_dendrites(const counter_type number_dendrites) noexcept {
        cell.set_number_excitatory_dendrites(number_dendrites);
    }

    /**
     * @brief Sets the number of free excitatory axons int he associated cell
     * @param number_axons The number of free excitatory axons
     */
    constexpr void set_cell_number_excitatory_axons(const counter_type number_axons) noexcept {
        cell.set_number_excitatory_axons(number_axons);
    }

    /**
     * @brief Sets the number of free inhibitory dendrites int he associated cell
     * @param number_dendrites The number of free inhibitory dendrites
     */
    constexpr void set_cell_number_inhibitory_dendrites(const counter_type number_dendrites) noexcept {
        cell.set_number_inhibitory_dendrites(number_dendrites);
    }

    /**
     * @brief Sets the number of free inhibitory axons int he associated cell
     * @param number_axons The number of free inhibitory axons
     */
    constexpr void set_cell_number_inhibitory_axons(const counter_type number_axons) noexcept {
        cell.set_number_inhibitory_axons(number_axons);
    }

    /**
     * @brief Sets the optional position for the excitatory dendrites position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    constexpr void set_cell_excitatory_dendrites_position(const std::optional<position_type>& opt_position) {
        cell.set_excitatory_dendrites_position(opt_position);
    }

    /**
     * @brief Sets the optional position for the inhibitory dendrites position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    constexpr void set_cell_inhibitory_dendrites_position(const std::optional<position_type>& opt_position) {
        cell.set_inhibitory_dendrites_position(opt_position);
    }

    /**
     * @brief Sets the optional position for the excitatory axons position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    constexpr void set_cell_excitatory_axons_position(const std::optional<position_type>& opt_position) {
        cell.set_excitatory_axons_position(opt_position);
    }

    /**
     * @brief Sets the optional position for the inhibitory axons position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    constexpr void set_cell_inhibitory_axons_position(const std::optional<position_type>& opt_position) {
        cell.set_inhibitory_axons_position(opt_position);
    }

    /**
     * @brief Returns the neuron id for the associated cell
     * @return The neuron id
     */
    [[nodiscard]] constexpr const NeuronID& get_cell_neuron_id() const noexcept {
        return cell.get_neuron_id();
    }

    /**
     * @brief Sets the neuron id for the associated cell
     * @param neuron_id The neuron id
     * @exception Throws a RelearnException if the neuron_id is not initialized
     */
    constexpr void set_cell_neuron_id(const NeuronID& neuron_id) {
        cell.set_neuron_id(neuron_id);
    }

    /**
     * @brief Sets the min and max of the associated cell
     * @param min The minimum boundary of the cell
     * @param max The maximum boundary of the cell
     * @exception Might throw a RelearnException
     */
    constexpr void set_cell_size(const box_size_type& min, const box_size_type& max) {
        cell.set_size(min, max);
    }

    /**
     * @brief Returns the size of the associated cell
     * @return The size of the cell
     */
    [[nodiscard]] constexpr std::tuple<box_size_type, box_size_type> get_size() const noexcept {
        return cell.get_size();
    }
};

template <typename AdditionalCellAttributes>
class OctreeNodeUpdater {
private:
    using position_type = typename Cell<AdditionalCellAttributes>::position_type;
    using counter_type = typename Cell<AdditionalCellAttributes>::counter_type;
    using box_size_type = typename Cell<AdditionalCellAttributes>::box_size_type;

    constexpr static bool has_excitatory_dendrite = AdditionalCellAttributes::has_excitatory_dendrite;
    constexpr static bool has_inhibitory_dendrite = AdditionalCellAttributes::has_inhibitory_dendrite;
    constexpr static bool has_excitatory_axon = AdditionalCellAttributes::has_excitatory_axon;
    constexpr static bool has_inhibitory_axon = AdditionalCellAttributes::has_inhibitory_axon;

public:
    /**
     * @brief Update the node based on its children. Saves the sum of vacant elements and the weighted average of the positions in the node
     * @param node The node to update
     * @exception Throws a RelearnException if node is nullptr or one of the children had vacant elements but not a position
     */
    static void update_node(OctreeNode<AdditionalCellAttributes>* node) {
        RelearnException::check(node != nullptr, "OctreeNodeUpdater::update_node: node was nullptr");

        if constexpr (has_excitatory_dendrite) {
            position_type my_position = { 0., 0., 0. };
            counter_type my_free_elements = 0;

            for (const auto& child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                const auto& child_cell = child->get_cell();

                // Sum up number of free elements
                const auto child_free_elements = child_cell.get_number_excitatory_dendrites();
                my_free_elements += child_free_elements;

                const auto& opt_child_position = child_cell.get_excitatory_dendrites_position();

                // We can use position if it's valid or if corresponding number of elements is 0
                RelearnException::check(opt_child_position.has_value() || (0 == child_free_elements), "OctreeNodeUpdater::update_node: The child had excitatory dendrites, but no position. ID: {}", child_cell.get_neuron_id());

                if (opt_child_position.has_value()) {
                    const auto& child_position = opt_child_position.value();

                    const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                    const auto is_in_box = child_position.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                    RelearnException::check(is_in_box, "OctreeNodeUpdater::update_node: The excitatory dendrites of the child are not in its cell");

                    const auto& scaled_position = child_position * static_cast<double>(child_free_elements);
                    my_position += scaled_position;
                }
            }

            node->set_cell_number_excitatory_dendrites(my_free_elements);

            /**
             * For calculating the new weighted position, make sure that we don't
             * divide by 0. This happens if the my number of dendrites is 0.
             */
            if (0 == my_free_elements) {
                node->set_cell_excitatory_dendrites_position({});
            } else {
                const auto scaled_position = my_position / my_free_elements;
                node->set_cell_excitatory_dendrites_position(std::optional<position_type>{ scaled_position });
            }
        }

        if constexpr (has_inhibitory_dendrite) {
            position_type my_position = { 0., 0., 0. };
            counter_type my_free_elements = 0;

            for (const auto& child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                const auto& child_cell = child->get_cell();

                // Sum up number of free elements
                const auto child_free_elements = child_cell.get_number_inhibitory_dendrites();
                my_free_elements += child_free_elements;

                const auto& opt_child_position = child_cell.get_inhibitory_dendrites_position();

                // We can use position if it's valid or if corresponding number of elements is 0
                RelearnException::check(opt_child_position.has_value() || (0 == child_free_elements), "OctreeNodeUpdater::update_node: The child had inhibitory dendrites, but no position. ID: {}", child_cell.get_neuron_id());

                if (opt_child_position.has_value()) {
                    const auto& child_position = opt_child_position.value();

                    const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                    const auto is_in_box = child_position.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                    RelearnException::check(is_in_box, "OctreeNodeUpdater::update_node: The inhibitory dendrites of the child are not in its cell");

                    const auto& scaled_position = child_position * static_cast<double>(child_free_elements);
                    my_position += scaled_position;
                }
            }

            node->set_cell_number_inhibitory_dendrites(my_free_elements);

            /**
             * For calculating the new weighted position, make sure that we don't
             * divide by 0. This happens if the my number of dendrites is 0.
             */
            if (0 == my_free_elements) {
                node->set_cell_inhibitory_dendrites_position({});
            } else {
                const auto scaled_position = my_position / my_free_elements;
                node->set_cell_inhibitory_dendrites_position(std::optional<position_type>{ scaled_position });
            }
        }

        if constexpr (has_excitatory_axon) {
            position_type my_position = { 0., 0., 0. };
            counter_type my_free_elements = 0;

            for (const auto& child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                const auto& child_cell = child->get_cell();

                // Sum up number of free elements
                const auto child_free_elements = child_cell.get_number_excitatory_axons();
                my_free_elements += child_free_elements;

                const auto& opt_child_position = child_cell.get_excitatory_axons_position();

                // We can use position if it's valid or if corresponding number of elements is 0
                RelearnException::check(opt_child_position.has_value() || (0 == child_free_elements), "OctreeNodeUpdater::update_node: The child had excitatory axons, but no position. ID: {}", child_cell.get_neuron_id());

                if (opt_child_position.has_value()) {
                    const auto& child_position = opt_child_position.value();

                    const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                    const auto is_in_box = child_position.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                    RelearnException::check(is_in_box, "OctreeNodeUpdater::update_node: The excitatory axons of the child are not in its cell");

                    const auto& scaled_position = child_position * static_cast<double>(child_free_elements);
                    my_position += scaled_position;
                }
            }

            node->set_cell_number_excitatory_axons(my_free_elements);

            /**
             * For calculating the new weighted position, make sure that we don't
             * divide by 0. This happens if the my number of axons is 0.
             */
            if (0 == my_free_elements) {
                node->set_cell_excitatory_axons_position({});
            } else {
                const auto scaled_position = my_position / my_free_elements;
                node->set_cell_excitatory_axons_position(std::optional<position_type>{ scaled_position });
            }
        }

        if constexpr (has_inhibitory_axon) {
            position_type my_position = { 0., 0., 0. };
            counter_type my_free_elements = 0;

            for (const auto& child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                const auto& child_cell = child->get_cell();

                // Sum up number of free elements
                const auto child_free_elements = child_cell.get_number_inhibitory_axons();
                my_free_elements += child_free_elements;

                const auto& opt_child_position = child_cell.get_inhibitory_axons_position();

                // We can use position if it's valid or if corresponding number of elements is 0
                RelearnException::check(opt_child_position.has_value() || (0 == child_free_elements), "OctreeNodeUpdater::update_node: The child had inhibitory axons, but no position. ID: {}", child_cell.get_neuron_id());

                if (opt_child_position.has_value()) {
                    const auto& child_position = opt_child_position.value();

                    const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                    const auto is_in_box = child_position.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                    RelearnException::check(is_in_box, "OctreeNodeUpdater::update_node: The inhibitory axons of the child are not in its cell");

                    const auto& scaled_position = child_position * static_cast<double>(child_free_elements);
                    my_position += scaled_position;
                }
            }

            node->set_cell_number_inhibitory_axons(my_free_elements);

            /**
             * For calculating the new weighted position, make sure that we don't
             * divide by 0. This happens if the my number of axons is 0.
             */
            if (0 == my_free_elements) {
                node->set_cell_inhibitory_axons_position({});
            } else {
                const auto scaled_position = my_position / my_free_elements;
                node->set_cell_inhibitory_axons_position(std::optional<position_type>{ scaled_position });
            }
        }
    }

    /**
     * @brief Updates the tree until the desired level.
     *      Uses OctreeNode::get_level() to determine the depth. The nodes at that depth are still updated, but not their children.
     * @param tree The root of the tree from where to update
     * @param max_depth The depth where the updates shall stop
     * @exception Throws a RelearnException if tree is nullptr or if max_depth is smaller than the depth of local_tree_root
     */
    static void update_tree(OctreeNode<AdditionalCellAttributes>* tree, const std::uint16_t max_depth = std::numeric_limits<std::uint16_t>::max()) {
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

        RelearnException::check(tree != nullptr, "OctreeNodeUpdater::update_tree: tree is nullptr.");
        RelearnException::check(tree->get_level() <= max_depth, "OctreeNodeUpdater::update_tree: The root had a larger depth than max_depth.");

        Stack<StackElement> stack{};
        stack.emplace_back(tree);

        while (!stack.empty()) {
            auto& current_element = stack.top();
            auto* current_octree_node = current_element.get_octree_node();

            if (current_element.was_already_visited()) {
                // Make sure that the element was visited before, i.e., its children are processed
                if (current_octree_node->is_parent()) {
                    // Don't update leaf nodes, they were updated before
                    OctreeNodeUpdater<AdditionalCellAttributes>::update_node(current_octree_node);
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
                    OctreeNodeUpdater<AdditionalCellAttributes>::update_node(current_octree_node);
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
};
