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

#include "Cell.h"
#include "Config.h"
#include "mpi/MPIWrapper.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"

#include <array>
#include <cstdint>
#include <optional>
#include <ostream>
#include <stack>
#include <vector>

/**
 * This class serves as the basic building blocks of the Octree.
 * Each object has up to Config::number_oct children (can be nullptr) and a Cell which summarizes the relevant biological aspects.
 * Additionally, an object stores its its MPI rank and whether or not it is an inner node.
 */
template <typename AdditionalCellAttributes>
class OctreeNode {
public:
    using OctreeNodePtr = OctreeNode<AdditionalCellAttributes>*;

    using position_type = typename Cell<AdditionalCellAttributes>::position_type;
    using counter_type = typename Cell<AdditionalCellAttributes>::counter_type;
    using box_size_type = typename Cell<AdditionalCellAttributes>::box_size_type;

    /**
     * @brief Returns a pointer to a fresh OctreeNode in the MPI memory window.
     *      Does not transfer ownership.
     * @expection Throws a RelearnException if not enough memory is available.
     * @return A valid pointer to an OctreeNode
     */
    [[nodiscard]] static OctreeNodePtr create() {
        return MemoryHolder<AdditionalCellAttributes>::get_available();
    }

    /**
     * @brief Deletes the object pointed to. Internally calls OctreeNode::reset().
     *      The pointer is invalidated.
     * @param node The pointer to object that shall be deleted
     */
    static void free(OctreeNodePtr node) {
        MemoryHolder<AdditionalCellAttributes>::make_available(node);
    }

    /**
     * @brief Returns the MPI rank this node belongs to
     * @return The MPI rank
     */
    [[nodiscard]] int get_rank() const noexcept {
        return rank;
    }

    /**
     * @brief Returns a flag that indicates if this node is an inner node or a leaf node
     * @return True iff it is an inner node
     */
    [[nodiscard]] bool is_parent() const noexcept {
        return parent;
    }

    /**
     * @brief Returns a flag that indicates if this node is an inner node or a leaf node
     * @return True iff it is a leaf node
     */
    [[nodiscard]] bool is_child() const noexcept {
        return !parent;
    }

    /**
     * @brief Returns a constant view on the associated child nodes. This reference is not invalidated by calls to other methods
     * @return A constant view on the associated child nodes
     */
    [[nodiscard]] const std::array<OctreeNodePtr, Constants::number_oct>& get_children() const noexcept {
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
    [[nodiscard]] const Cell<AdditionalCellAttributes>& get_cell() const noexcept {
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
     * @brief Inserts a neuron with the specified id and the specified MPI rank into the subtree that is induced by this object.
     * @param position The position of the new neuron
     * @param neuron_id The id of the new neuron (can be Constants::uninitialized to inidicate a virtual neuron), <= Constants::uninitialized
     * @param rank The MPI rank of the new neuron, >= 0
     * @exception Throws a RelearnException if one of the following happens:
     *      (a) The position is not within the cell's boundaries
     *      (b) rank is < 0
     *      (c) neuron_id > Constants::uninitialized
     *      (d) Allocating a new object in the shared memory window fails
     *      (e) Something went wrong within the insertion
     * @return A pointer to the newly created and inserted node
     */
    [[nodiscard]] OctreeNodePtr insert(const box_size_type& position, const NeuronID& neuron_id, const int rank) {
        const auto& [cell_xyz_min, cell_xyz_max] = cell.get_size();
        const auto is_in_box = position.check_in_box(cell_xyz_min, cell_xyz_max);

        RelearnException::check(is_in_box, "OctreeNode::insert: position is not in box: {} in [{}, {}]", position, cell_xyz_min, cell_xyz_max);
        RelearnException::check(rank >= 0, "OctreeNode::insert: rank was {}", rank);
        RelearnException::check(neuron_id.is_initialized(), "OctreeNode::insert, neuron_id is not initialized");

        unsigned char new_position_octant = 0;

        OctreeNodePtr parent_node = nullptr;

        // Correct position for new node not found yet
        for (OctreeNodePtr current_node = this; nullptr != current_node;) {
            /**
             * My parent already exists.
             * Calc which child to follow, i.e., determine octant
             */
            new_position_octant = current_node->get_cell().get_octant_for_position(position);

            parent_node = current_node;
            current_node = current_node->get_child(new_position_octant);
        }

        RelearnException::check(parent_node != nullptr, "OctreeNode::insert: parent_node is nullptr");

        /**
         * Found my octant, but
         * I'm the very first child of that node.
         * I.e., the node is a leaf.
         */
        if (parent_node->is_child()) {
            /**
             * The found parent node is virtual and can just be substituted,
             * i.e., it was constructed while constructing the upper part to the branch nodes.
             */
            if (parent_node->get_cell_neuron_id().is_virtual() && neuron_id != parent_node->get_cell_neuron_id()) {
                parent_node->set_cell_neuron_id(neuron_id);
                parent_node->set_cell_neuron_position({ position });
                parent_node->set_rank(rank);
                return parent_node;
            }

            for (unsigned char idx = new_position_octant; idx == new_position_octant;) {
                /**
                 * Make node containing my octant a parent by
                 * adding neuron in this node as child node
                 */

                // Determine octant for neuron
                const auto& cell_own_position = parent_node->get_cell().get_neuron_position();
                RelearnException::check(cell_own_position.has_value(), "OctreeNode::insert: While building the octree, the cell doesn't have a position");

                idx = parent_node->get_cell().get_octant_for_position(cell_own_position.value());
                const auto parent_level = parent_node->get_level();
                auto* new_node = OctreeNode<AdditionalCellAttributes>::create();
                parent_node->set_child(new_node, idx);

                /**
                 * Init this new node properly
                 */
                const auto& [minimum_position, maximum_position] = parent_node->get_cell().get_size_for_octant(idx);

                new_node->set_cell_size(minimum_position, maximum_position);
                new_node->set_cell_neuron_position(cell_own_position);
                new_node->set_level(parent_level+1);

                // Neuron ID
                const auto prev_neuron_id = parent_node->get_cell_neuron_id();
                new_node->set_cell_neuron_id(prev_neuron_id);

                /**
                 * Set neuron ID of parent (inner node) to uninitialized.
                 * It is not used for inner nodes.
                 */
                parent_node->set_cell_neuron_id(NeuronID::virtual_id());
                parent_node->set_parent(); // Mark node as parent

                // MPI rank who owns this node
                new_node->set_rank(parent_node->get_rank());

                // Determine my octant
                new_position_octant = parent_node->get_cell().get_octant_for_position(position);

                if (new_position_octant == idx) {
                    parent_node = new_node;
                }
            }
        }

        OctreeNode* new_node_to_insert = OctreeNode<AdditionalCellAttributes>::create();
        RelearnException::check(new_node_to_insert != nullptr, "OctreeNode::insert: new_node_to_insert is nullptr");

        /**
         * Found my position in children array,
         * add myself to the array now
         */
        parent_node->set_child(new_node_to_insert, new_position_octant);
        const auto parent_level = parent_node->get_level();

        const auto& [minimum_position, maximum_position] = parent_node->get_cell().get_size_for_octant(new_position_octant);

        new_node_to_insert->set_cell_size(minimum_position, maximum_position);
        new_node_to_insert->set_cell_neuron_position({ position });
        new_node_to_insert->set_cell_neuron_id(neuron_id);
        new_node_to_insert->set_rank(rank);
        new_node_to_insert->set_level(parent_level+1);

        bool has_children = false;

        for (auto* child : children) {
            if (child != nullptr) {
                has_children = true;
                break;
            }
        }

        RelearnException::check(has_children, "OctreeNode::insert: the node didn't have children");

        return new_node_to_insert;
    }

    /**
     * @brief Sets the associated MPI rank
     * @param new_rank The associated MPI rank, >= 0
     * @exception Throws a RelearnException if new_rank < 0
     */
    void set_rank(const int new_rank) {
        RelearnException::check(new_rank >= 0, "OctreeNode::set_rank: new_rank is {}", new_rank);
        rank = new_rank;
    }

    /**
     * @brief Marks this node as a parent (an inner node)
     */
    void set_parent() noexcept {
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
     * @brief Set the level attribute.
     * @param value The new value of the level.
     */
    void set_level(const uint16_t value){
        level = value;
    }

    /**
     * @brief Get the level attribute.
     * @return The current value of level.
     */
    [[nodiscard]] uint16_t get_level() const noexcept{
        return level;
    }

    /**
     * @brief Resets the current object:
     *      (a) The cell is newly constructed
     *      (b) The children are newly constructed
     *      (c) parent is false
     *      (d) rank is -1
     */
    void reset() noexcept {
        cell = Cell<AdditionalCellAttributes>{};
        children = std::array<OctreeNodePtr, Constants::number_oct>{ nullptr };
        parent = false;
        rank = -1;
        level = 0;
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
        output_stream << "\n";

        output_stream << "  is_parent  : " << octree_node.is_parent() << "\n\n";
        output_stream << "  rank       : " << octree_node.get_rank() << "\n\n";
        output_stream << octree_node.get_cell();
        output_stream << "\n";

        return output_stream;
    }

private:
    std::array<OctreeNodePtr, Constants::number_oct> children{ nullptr };
    Cell<AdditionalCellAttributes> cell{};

    bool parent{ false };
    uint16_t level{0};

    int rank{ -1 }; // MPI rank who owns this octree node

public:
    /**
     * @brief Sets the optional position for both the excitatory and inhibitory positions in the associated cell
     * @param opt_position The optional position, can be empty
     */
    void set_cell_neuron_position(const std::optional<position_type>& opt_position) noexcept {
        cell.set_neuron_position(opt_position);
    }

    /**
     * @brief Sets the number of free excitatory and inhibitory dendrites in the associated cell
     * @param num_ex The number of free excitatory dendrites
     * @param num_in The number of free inhibitory dendrites
     */
    void set_cell_number_dendrites(const counter_type num_ex, const counter_type num_in) noexcept {
        cell.set_number_excitatory_dendrites(num_ex);
        cell.set_number_inhibitory_dendrites(num_in);
    }

    /**
     * @brief Sets the number of free excitatory and inhibitory axons in the associated cell
     * @param num_ex The number of free excitatory axons
     * @param num_in The number of free inhibitory axons
     */
    void set_cell_number_axons(const counter_type num_ex, const counter_type num_in) noexcept {
        cell.set_number_excitatory_axons(num_ex);
        cell.set_number_inhibitory_axons(num_in);
    }

    /**
     * @brief Sets the optional position for the excitatory dendrites position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    void set_cell_excitatory_dendrites_position(const std::optional<position_type>& opt_position) {
        cell.set_excitatory_dendrites_position(opt_position);
    }

    /**
     * @brief Sets the optional position for the inhibitory dendrites position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    void set_cell_inhibitory_dendrites_position(const std::optional<position_type>& opt_position) {
        cell.set_inhibitory_dendrites_position(opt_position);
    }

    /**
     * @brief Sets the optional position for the excitatory axons position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    void set_cell_excitatory_axons_position(const std::optional<position_type>& opt_position) {
        cell.set_excitatory_axons_position(opt_position);
    }

    /**
     * @brief Sets the optional position for the inhibitory axons position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    void set_cell_inhibitory_axons_position(const std::optional<position_type>& opt_position) {
        cell.set_inhibitory_axons_position(opt_position);
    }

    /**
     * @brief Returns the neuron id for the associated cell
     * @return The neuron id
     */
    [[nodiscard]] NeuronID get_cell_neuron_id() const noexcept {
        return cell.get_neuron_id();
    }

    /**
     * @brief Sets the neuron id for the associated cell
     * @param neuron_id The neuron id
     * @exception Throws a RelearnException if the neuron_id is not initialized
     */
    void set_cell_neuron_id(const NeuronID& neuron_id) {
        cell.set_neuron_id(neuron_id);
    }

    /**
     * @brief Sets the min and max of the associated cell
     * @param min The minimum boundary of the cell
     * @param max The maximum boundary of the cell
     * @exception Might throw a RelearnException
     */
    void set_cell_size(const box_size_type& min, const box_size_type& max) {
        cell.set_size(min, max);
    }

    /**
     * @brief Returns the size of the associated cell
     * @return The size of the cell
     */
    [[nodiscard]] std::tuple<box_size_type, box_size_type> get_size() const noexcept {
        return cell.get_size();
    }

};
