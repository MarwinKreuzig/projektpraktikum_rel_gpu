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

#include "Cell.h"
#include "../Config.h"
#include "../mpi/MPIWrapper.h"
#include "../mpi/MPI_RMA_MemAllocator.h"

#include <array>
#include <optional>
#include <ostream>

/**
 * This class serves as the basic building blocks of the Octree.
 * Each object has up to 8 children (can be nullptr) and a Cell which summarizes the relevant biological aspects.
 * Additionally, an object stores its level within the tree (0 being root), its MPI rank and whether or not it is an inner node.
 */
template <typename AdditionalCellAttributes>
class OctreeNode {
public:
    using OctreeNodePtr = OctreeNode<AdditionalCellAttributes>*;

    /**
     * @brief Returns the MPI rank to which this object belongs
     * @return The MPI rank to which this object belongs
     */
    [[nodiscard]] int get_rank() const noexcept {
        return rank;
    }

    /**
     * @brief Returns the level in the octree, 0 being root
     * @return The level in the octree
     */
    [[nodiscard]] size_t get_level() const noexcept {
        return level;
    }

    /**
     * @brief Returns a flag that indicates if this node is an inner node or a leaf node
     * @return True iff it is an inner node
     */
    [[nodiscard]] bool is_parent() const noexcept {
        return parent;
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
    [[nodiscard]] const OctreeNodePtr get_child(size_t idx) const {
        RelearnException::check(idx < Constants::number_oct, "In OctreeNode::get_child const, idx was: %u", idx);
        // NOLINTNEXTLINE
        return children[idx];
    }

    /**
     * @brief Returns the child node with the requested id (id calculation base on Cell::get_octant_for_position)
     * @exception Throws a RelearnException if idx >= Constants::number_oct
     * @return The associated child
     */
    [[nodiscard]] OctreeNodePtr get_child(size_t idx) {
        RelearnException::check(idx < Constants::number_oct, "In OctreeNode::get_child, idx was: %u", idx);
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
    [[nodiscard]] OctreeNodePtr insert(const Vec3d& position, const size_t neuron_id, int rank) {
        Vec3d cell_xyz_min;
        Vec3d cell_xyz_max;

        std::tie(cell_xyz_min, cell_xyz_max) = cell.get_size();

        RelearnException::check(cell_xyz_min.get_x() <= position.get_x() && position.get_x() <= cell_xyz_max.get_x(), "In OctreeNode::insert, x was not in range");
        RelearnException::check(cell_xyz_min.get_y() <= position.get_y() && position.get_y() <= cell_xyz_max.get_y(), "In OctreeNode::insert, y was not in range");
        RelearnException::check(cell_xyz_min.get_z() <= position.get_z() && position.get_z() <= cell_xyz_max.get_z(), "In OctreeNode::insert, z was not in range");

        RelearnException::check(rank >= 0, "In OctreeNode::insert, rank was smaller than 0");
        RelearnException::check(neuron_id <= Constants::uninitialized, "In OctreeNode::insert, neuron_id was too large");

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

        RelearnException::check(parent_node != nullptr, "parent_node is nullptr");

        /**
	     * Found my octant, but
	     * I'm the very first child of that node.
	     * I.e., the node is a leaf.
	     */
        if (!parent_node->is_parent()) {
            /**
             * The found parent node is virtual and can just be substituted,
             * i.e., it was constructed while constructing the upper part to the branch nodes.
             */
            if (parent_node->get_cell_neuron_id() == Constants::uninitialized && neuron_id != parent_node->get_cell_neuron_id()) {
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
                const auto& cell_own_position = parent_node->get_cell().get_dendrite_position();
                RelearnException::check(cell_own_position.has_value(), "While building the octree, the cell doesn't have a position");

                idx = parent_node->get_cell().get_octant_for_position(cell_own_position.value());
                auto* new_node = MPI_RMA_MemAllocator<BarnesHutCell>::new_octree_node();
                parent_node->set_child(new_node, idx);

                /**
			     * Init this new node properly
			     */
                Vec3d xyz_min;
                Vec3d xyz_max;
                std::tie(xyz_min, xyz_max) = parent_node->get_cell().get_size_for_octant(idx);

                new_node->set_cell_size(xyz_min, xyz_max);
                new_node->set_cell_neuron_position(cell_own_position);

                // Neuron ID
                const auto prev_neuron_id = parent_node->get_cell_neuron_id();
                new_node->set_cell_neuron_id(prev_neuron_id);

                /**
			     * Set neuron ID of parent (inner node) to uninitialized.
			     * It is not used for inner nodes.
			     */
                parent_node->set_cell_neuron_id(Constants::uninitialized);
                parent_node->set_parent(); // Mark node as parent

                // MPI rank who owns this node
                new_node->set_rank(parent_node->get_rank());

                // New node is one level below
                new_node->set_level(parent_node->get_level() + 1);

                // Determine my octant
                new_position_octant = parent_node->get_cell().get_octant_for_position(position);

                if (new_position_octant == idx) {
                    parent_node = new_node;
                }
            }
        }

        OctreeNode* new_node_to_insert = MPI_RMA_MemAllocator<BarnesHutCell>::new_octree_node();
        RelearnException::check(new_node_to_insert != nullptr, "new_node_to_insert is nullptr");

        /**
	     * Found my position in children array,
	     * add myself to the array now
	     */
        parent_node->set_child(new_node_to_insert, new_position_octant);
        new_node_to_insert->set_level(parent_node->get_level() + 1); // Now we know level of me

        Vec3d xyz_min;
        Vec3d xyz_max;
        std::tie(xyz_min, xyz_max) = parent_node->get_cell().get_size_for_octant(new_position_octant);

        new_node_to_insert->set_cell_size(xyz_min, xyz_max);
        new_node_to_insert->set_cell_neuron_position({ position });
        new_node_to_insert->set_cell_neuron_id(neuron_id);
        new_node_to_insert->set_rank(rank);

        bool has_children = false;

        for (auto* child : children) {
            if (child != nullptr) {
                has_children = true;
            }
        }

        RelearnException::check(has_children, "");

        return new_node_to_insert;
    }

    /**
     * @brief Sets the associated MPI rank
     * @param new_rank The associated MPI rank, >= 0
     * @exception Throws a RelearnException if new_rank < 0
     */
    void set_rank(int new_rank) {
        RelearnException::check(new_rank >= 0, "In OctreeNode::set_rank, new_rank was: %u", new_rank);
        rank = new_rank;
    }

    /**
     * @brief Sets the level in the octree
     * @param new_level The level in the octree, < Constants::uninitialized
     * @expection Throws a RelearnException if new_level is too large
     */
    void set_level(size_t new_level) {
        RelearnException::check(new_level < Constants::uninitialized, "In OctreeNode::set_level, new_level was: %u", new_level);
        level = new_level;
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
    void set_child(OctreeNodePtr node, size_t idx) {
        RelearnException::check(idx < Constants::number_oct, "In OctreeNode::set_child, idx was: %u", idx);
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
     * @brief Resets the current object:
     *      (a) The cell is newly constructed
     *      (b) The children are newly constructed
     *      (c) parent is false
     *      (d) rank is -1
     *      (e) level is Constants::uninitialized
     */
    void reset() noexcept {
        cell = Cell<AdditionalCellAttributes>{};
        children = std::array<OctreeNodePtr, Constants::number_oct>{ nullptr };
        parent = false;
        rank = -1;
        level = Constants::uninitialized;
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
        output_stream << "  rank       : " << octree_node.get_rank() << "\n";
        output_stream << "  level      : " << octree_node.get_level() << "\n\n";
        output_stream << octree_node.get_cell();
        output_stream << "\n";

        return output_stream;
    }

private:
    std::array<OctreeNodePtr, Constants::number_oct> children{ nullptr };
    Cell<AdditionalCellAttributes> cell{};

    bool parent{ false };

    int rank{ -1 }; // MPI rank who owns this octree node
    size_t level{ Constants::uninitialized }; // Level in the tree [0 (= root) ... depth of tree]

public:
    /**
     * @brief Sets the optional position for both the excitatory and inhibitory positions in the associated cell
     * @param opt_position The optional position, can be empty
     */
    void set_cell_neuron_position(const std::optional<Vec3d>& opt_position) noexcept {
        cell.set_dendrite_position(opt_position);
    }

    /**
     * @brief Sets the number of free excitatory and inhibitory dendrites in the associated cell
     * @param num_ex The number of free excitatory dendrites
     * @param num_in The number of free inhibitory dendrites
     */
    void set_cell_num_dendrites(unsigned int num_ex, unsigned int num_in) noexcept {
        cell.set_number_excitatory_dendrites(num_ex);
        cell.set_number_inhibitory_dendrites(num_in);
    }

    /**
     * @brief Sets the optional position for the excitatory position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    void set_cell_neuron_pos_exc(const std::optional<Vec3d>& opt_position) noexcept {
        cell.set_excitatory_dendrite_position(opt_position);
    }

    /**
     * @brief Sets the optional position for the inhibitory position in the associated cell
     * @param opt_position The optional position, can be empty
     */
    void set_cell_neuron_pos_inh(const std::optional<Vec3d>& opt_position) noexcept {
        cell.set_inhibitory_dendrite_position(opt_position);
    }

    /**
     * @brief Sets the neuron id in the associated cell
     * @param neuron_id The new neuron id
     */
    void set_cell_neuron_id(size_t neuron_id) noexcept {
        cell.set_neuron_id(neuron_id);
    }

    /**
     * @brief Returns the neuron id for the associated cell. Is Constants::uninitialized to indicate a virtual neuron aka an inner node in the Octree
     * @return The neuron id
     */
    [[nodiscard]] size_t get_cell_neuron_id() const noexcept {
        return cell.get_neuron_id();
    }

    /**
     * @brief Sets the min and max of the associated cell
     * @param min The minimum boundary of the cell
     * @param max The maximum boundary of the cell
     * @exception Throws a RelearnException if one ordinate of min is larger than the associated of max
     */
    void set_cell_size(const Vec3d& min, const Vec3d& max) {
        cell.set_size(min, max);
    }

    /**
     * @brief Returns the size of the associated cell
     * @return The size of the cell
     */
    [[nodiscard]] std::tuple<Vec3d, Vec3d> get_size() const noexcept {
        return cell.get_size();
    }
};
