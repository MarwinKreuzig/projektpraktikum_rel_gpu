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
#include "../neurons/SignalType.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"

#include <optional>
#include <tuple>

/**
 * This class summarizes all 'octree-relevant' data from a neuron.
 * It contains a size in the octree (min and max), a neuron id (value Constants::uninitialized for virtual neurons, aka. inner nodes in the octree), 
 * and a position for the inhibitory and excitatory dendrites
 */
class Cell {
public:
    /**
     * @brief Sets the size of this cell
     * @param min The minimum x, y, and z of the sell
     * @param max The maximum x, y, and z of the sell
     * @exception Throws a RelearnException if one component of min is larger than the respective component of max
     */
    void set_size(const Vec3d& min, const Vec3d& max) {
        RelearnException::check(min.get_x() <= max.get_x(), "In Cell::set_size, x was not ok");
        RelearnException::check(min.get_y() <= max.get_y(), "In Cell::set_size, y was not ok");
        RelearnException::check(min.get_z() <= max.get_z(), "In Cell::set_size, z was not ok");

        xyz_min = min;
        xyz_max = max;
    }

    /**
     * @brief Returns the size of the cell
     * @return The size of the cell
     */
    [[nodiscard]] std::tuple<Vec3d, Vec3d> get_size() const noexcept {
        return std::make_tuple(xyz_min, xyz_max);
    }

    /**
	 * @brief Returns maximum edge length of the cell, i.e., ||max - min||_1
	 * @return The maximum edge length of the cell
	 */
    [[nodiscard]] double get_maximal_dimension_difference() const noexcept {
        const auto diff_vector = xyz_max - xyz_min;
        const auto diff = diff_vector.get_maximum();

        return diff;
    }

    /**
     * @brief Sets the dendrite position for both inhibitory and excitatory
     * @param opt_position The dendrite position, can be empty
     */
    void set_dendrite_position(const std::optional<Vec3d>& opt_position) noexcept {
        set_excitatory_dendrite_position(opt_position);
        set_inhibitory_dendrite_position(opt_position);
    }

    /**
     * @brief Returns the dendrite position, for which either both positions must be empty or equal
     * @exception Throws a RelearnException if one position is valid and the other one invalid or if both are valid with different values
     * @return The position of the dendrite, can be empty
     */
    [[nodiscard]] std::optional<Vec3d> get_dendrite_position() const;

    /**
     * @brief Returns the position of the excitatory dendrite
     * @return The position of the excitatory dendrite
     */
    [[nodiscard]] std::optional<Vec3d> get_excitatory_dendrite_position() const noexcept {
        return dendrites_ex.xyz_pos;
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param opt_position The new position of the excitatory dendrite
     * @exception Throws a RelearnException if the position is valid but not within the box
     */
    void set_excitatory_dendrite_position(const std::optional<Vec3d>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();

            RelearnException::check(xyz_min.get_x() <= position.get_x() && position.get_x() <= xyz_max.get_x(), "In Cell::set_neuron_position_exc, x was not in the box");
            RelearnException::check(xyz_min.get_y() <= position.get_y() && position.get_y() <= xyz_max.get_y(), "In Cell::set_neuron_position_exc, y was not in the box");
            RelearnException::check(xyz_min.get_z() <= position.get_z() && position.get_z() <= xyz_max.get_z(), "In Cell::set_neuron_position_exc, z was not in the box");
        }

        dendrites_ex.xyz_pos = opt_position;
    }

    /**
     * @brief Returns the position of the inhibitory dendrite
     * @return The position of the inhibitory dendrite
     */
    [[nodiscard]] std::optional<Vec3d> get_inhibitory_dendrite_position() const noexcept {
        return dendrites_in.xyz_pos;
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param opt_position The new position of the inhibitory dendrite
     * @exception Throws a RelearnException if the position is valid but not within the box
     */
    void set_inhibitory_dendrite_position(const std::optional<Vec3d>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();

            RelearnException::check(xyz_min.get_x() <= position.get_x() && position.get_x() <= xyz_max.get_x(), "In Cell::set_neuron_position_exc, x was not in the box");
            RelearnException::check(xyz_min.get_y() <= position.get_y() && position.get_y() <= xyz_max.get_y(), "In Cell::set_neuron_position_exc, y was not in the box");
            RelearnException::check(xyz_min.get_z() <= position.get_z() && position.get_z() <= xyz_max.get_z(), "In Cell::set_neuron_position_exc, z was not in the box");
        }

        dendrites_in.xyz_pos = opt_position;
    }

    /**
     * @brief Returns the position of the dendrite with the given signal type
     * @param dendrite_type The type of dendrite which's position should be returned
     * @return The position of the associated dendrite, can be empty
     */
    [[nodiscard]] std::optional<Vec3d> get_dendrite_position_for(SignalType dendrite_type) const noexcept {
        if (dendrite_type == SignalType::EXCITATORY) {
            return dendrites_ex.xyz_pos;
        }

        return dendrites_in.xyz_pos;
    }

    /**
     * @brief Sets the number of free excitatory dendrites in this cell
     * @param num_dendrites The number of free excitatory dendrites
     */
    void set_number_excitatory_dendrites(unsigned int num_dendrites) noexcept {
        dendrites_ex.num_dendrites = num_dendrites;
    }

    /**
     * @brief Returns the number of free excitatory dendrites in this cell
     * @return The number of free excitatory dendrites
     */
    [[nodiscard]] unsigned int get_number_excitatory_dendrites() const noexcept {
        return dendrites_ex.num_dendrites;
    }

    /**
     * @brief Sets the number of free inhibitory dendrites in this cell
     * @param num_dendrites The number of free inhibitory dendrites
     */
    void set_number_inhibitory_dendrites(unsigned int num_dendrites) noexcept {
        dendrites_in.num_dendrites = num_dendrites;
    }

    /**
     * @brief Returns the number of free inhibitory dendrites in this cell
     * @return The number of free inhibitory dendrites
     */
    [[nodiscard]] unsigned int get_number_inhibitory_dendrites() const noexcept {
        return dendrites_in.num_dendrites;
    }

    /**
     * @brief Returns the number of free dendrites for the associated type in this cell
     * @return The number of free dendrites for the associated type
     */
    [[nodiscard]] unsigned int get_number_dendrites_for(SignalType dendrite_type) const noexcept {
        if (dendrite_type == SignalType::EXCITATORY) {
            return dendrites_ex.num_dendrites;
        }

        return dendrites_in.num_dendrites;
    }

    /**
     * @brief Sets the neuron id for the associated cell. Can be set to Constants::uninitialized to indicate a virtual neuron aka an inner node in the Octree
     * @param neuron_id The neuron id, can be Constants::uninitialized
     */
    void set_neuron_id(size_t neuron_id) noexcept {
        this->neuron_id = neuron_id;
    }

    /**
     * @brief Returns the neuron id for the associated cell. Is Constants::uninitialized to indicate a virtual neuron aka an inner node in the Octree
     * @return The neuron id
     */
    [[nodiscard]] size_t get_neuron_id() const noexcept {
        return neuron_id;
    }

    /**
     * @brief Calculates the octant for the position.
     * @param pos The position inside the current cell which's octant position should be found
     * @exception Throws a RelearnException if the position is not within the current cell
     * @return A value from 0 to 7 that indicates which octant the position is
     * 
     * The binary numbering is computed as follows:
     * 
     * 		   110 ----- 111
	 *		   /|        /|
	 *		  / |       / |
	 *		 /  |      /  |
	 *	   010 ----- 011  |    y
	 *		|  100 ---|- 101   ^   z
	 *		|  /      |  /     |
	 *		| /       | /      | /
	 *		|/        |/       |/
	 *	   000 ----- 001       +-----> x
     */
    [[nodiscard]] unsigned char get_octant_for_position(const Vec3d& pos) const;

    /**
     * @brief Returns the size of the cell in the in the given octant
     * @param octant The octant, between 0 and 7
     * @exception Throws a RelearnException if octant > 7
     * @return A tuple with (min, max) for the cell in the given octant
     */
    [[nodiscard]] std::tuple<Vec3d, Vec3d> get_size_for_octant(unsigned char octant) const;

    /**
     * @brief Prints the cell to LogFiles::EventType::Cout
     */
    void print() const;

private:
    struct Dendrites {
        // All dendrites have the same position
        std::optional<Vec3d> xyz_pos{};
        unsigned int num_dendrites{ 0 };
        // TODO(future)
        // List colliding_axons;
    };

    // Two points describe size of cell
    Vec3d xyz_min{ Constants::uninitialized };
    Vec3d xyz_max{ Constants::uninitialized };

    /**
	 * Cell contains info for one neuron, which could be a "super" neuron
	 *
	 * Info about EXCITATORY dendrites: dendrites_ex
	 * Info about INHIBITORY dendrites: dendrites_in
	 */
    Dendrites dendrites_ex{};
    Dendrites dendrites_in{};

    /**
	 * ID of the neuron in the cell.
	 * This is only valid for cells that contain a normal neuron.
	 * For those with a super neuron, it has no meaning.
	 * This info is used to identify (return) the target neuron for a given axon
	 */
    size_t neuron_id{ Constants::uninitialized };
};
