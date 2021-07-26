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
#include "Cell.h"

#include <array>
#include <optional>

/**
 * This class serves as the basic building blocks of the Octree.
 * Each object has up to 8 children (can be nullptr) and a Cell which summarizes the relevant biological aspects.
 * Additionally, an object stores its level within the tree (0 being root), its MPI rank and whether or not it is an inner node.
 */
class OctreeNode {
    std::array<OctreeNode*, Constants::number_oct> children{ nullptr };
    Cell cell{};

    bool parent{ false };

    int rank{ -1 }; // MPI rank who owns this octree node
    size_t level{ Constants::uninitialized }; // Level in the tree [0 (= root) ... depth of tree]

public:
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
    [[nodiscard]] const std::array<OctreeNode*, Constants::number_oct>& get_children() const noexcept {
        return children;
    }

    /**
     * @brief Returns the child node with the requested id (id calculation base on Cell::get_octant_for_position)
     * @exception Throws a RelearnException if idx >= Constants::number_oct
     * @return The associated child
     */
    [[nodiscard]] const OctreeNode* get_child(size_t idx) const {
        RelearnException::check(idx < Constants::number_oct, "In OctreeNode::get_child const, idx was: %u", idx);
        // NOLINTNEXTLINE
        return children[idx];
    }

    /**
     * @brief Returns the child node with the requested id (id calculation base on Cell::get_octant_for_position)
     * @exception Throws a RelearnException if idx >= Constants::number_oct
     * @return The associated child
     */
    [[nodiscard]] OctreeNode* get_child(size_t idx) {
        RelearnException::check(idx < Constants::number_oct, "In OctreeNode::get_child, idx was: %u", idx);
        // NOLINTNEXTLINE
        return children[idx];
    }

    /**
     * @brief Returns a constant view on the associated cell. This reference is not invalidated by calls to other methods
     * @return A constant view on the associated cell
     */
    [[nodiscard]] const Cell& get_cell() const noexcept {
        return cell;
    }

    /**
     * @brief Returns a flag that indicates if this object belongs to the current MPI process. 
     *      Achieves so by calling MPIWrapper::get_my_rank()
     * @exception Throws a RelearnException if the MPIWrapper is not properly initialized
     * @return True iff this object belongs to the current MPI process
     */
    [[nodiscard]] bool is_local() const;

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
    [[nodiscard]] OctreeNode* insert(const Vec3d& position, const size_t neuron_id, int rank);

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
     * @brief Sets the neuron id in the associated cell
     * @param neuron_id The new neuron id
     */
    void set_cell_neuron_id(size_t neuron_id) noexcept {
        cell.set_neuron_id(neuron_id);
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
     * @brief Sets the node as the child with the given index and updates the parent flag accordingly
     * @param node The new child node (can be nullptr)
     * @param idx The index of the child which shall be set, < Constants::number_oct
     * @exception Throws a RelearnException if idx >= Constants::number_oct
     */
    void set_child(OctreeNode* node, size_t idx) {
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
     * @brief Prints the current octreenode to LogFiles::EventType::Cout
     */
    void print() const;

    /**
     * @brief Resets the current object:
     *      (a) The cell is newly constructed
     *      (b) The children are newly constructed
     *      (c) parent is false
     *      (d) rank is -1
     *      (e) level is Constants::uninitialized
     */
    void reset() noexcept {
        cell = Cell{};
        children = std::array<OctreeNode*, Constants::number_oct>{ nullptr };
        parent = false;
        rank = -1;
        level = Constants::uninitialized;
    }
};
