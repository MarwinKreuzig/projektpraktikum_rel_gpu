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
#include "../algorithm/FastMultipoleMethodsCell.h"
#include "../neurons/ElementType.h"
#include "../neurons/SignalType.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"

#include <optional>
#include <ostream>
#include <tuple>

/**
 * This class summarizes all 'octree-relevant' data from a neuron.
 * It contains a size in the octree (min and max), a neuron id (value Constants::uninitialized for virtual neurons, aka. inner nodes in the octree), 
 * and a position for the inhibitory and excitatory dendrites
 */
template <typename AdditionalCellAttributes>
class Cell {
public:
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
     * @brief Returns the size of the cell as tuple of (1) min and (2) max
     * @return The size of the cell as tuple of (1) min and (2) max
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
    [[nodiscard]] unsigned char get_octant_for_position(const Vec3d& pos) const {
        unsigned char idx = 0;

        const auto& x = pos.get_x();
        const auto& y = pos.get_y();
        const auto& z = pos.get_z();

        /**
	     * Sanity check: Make sure that the position is within this cell
	     * This check returns false if negative coordinates are used.
	     * Thus make sure to use positions >=0.
	     */
        RelearnException::check(x >= xyz_min.get_x() && x <= xyz_max.get_x(), "x is bad");
        RelearnException::check(y >= xyz_min.get_y() && y <= xyz_max.get_y(), "y is bad");
        RelearnException::check(z >= xyz_min.get_z() && z <= xyz_max.get_z(), "z is bad");

        //NOLINTNEXTLINE
        idx = idx | ((x < (xyz_min.get_x() + xyz_max.get_x()) / 2.0) ? 0 : 1); // idx | (pos_x < midpoint_dim_x) ? 0 : 1

        //NOLINTNEXTLINE
        idx = idx | ((y < (xyz_min.get_y() + xyz_max.get_y()) / 2.0) ? 0 : 2); // idx | (pos_y < midpoint_dim_y) ? 0 : 2

        //NOLINTNEXTLINE
        idx = idx | ((z < (xyz_min.get_z() + xyz_max.get_z()) / 2.0) ? 0 : 4); // idx | (pos_z < midpoint_dim_z) ? 0 : 4

        RelearnException::check(idx < Constants::number_oct, "Octree octant must be smaller than 8");

        return idx;
    }

    /**
     * @brief Returns the size of the cell in the in the given octant
     * @param octant The octant, between 0 and 7
     * @exception Throws a RelearnException if octant > 7
     * @return A tuple with (min, max) for the cell in the given octant
     */
    [[nodiscard]] std::tuple<Vec3d, Vec3d> get_size_for_octant(unsigned char octant) const /*noexcept*/ {
        RelearnException::check(octant <= Constants::number_oct, "Octant was too large");

        const bool x_over_halfway_point = (octant & 1U) != 0;
        const bool y_over_halfway_point = (octant & 2U) != 0;
        const bool z_over_halfway_point = (octant & 4U) != 0;

        Vec3d octant_xyz_min = this->xyz_min;
        Vec3d octant_xyz_max = this->xyz_max;
        // NOLINTNEXTLINE
        Vec3d octant_xyz_middle = (octant_xyz_min + octant_xyz_max) / 2.0;

        if (x_over_halfway_point) {
            octant_xyz_min.set_x(octant_xyz_middle.get_x());
        } else {
            octant_xyz_max.set_x(octant_xyz_middle.get_x());
        }

        if (y_over_halfway_point) {
            octant_xyz_min.set_y(octant_xyz_middle.get_y());
        } else {
            octant_xyz_max.set_y(octant_xyz_middle.get_y());
        }

        if (z_over_halfway_point) {
            octant_xyz_min.set_z(octant_xyz_middle.get_z());
        } else {
            octant_xyz_max.set_z(octant_xyz_middle.get_z());
        }

        return std::make_tuple(octant_xyz_min, octant_xyz_max);
    }

    void set_inhibitory_dendrites_position(const std::optional<Vec3d>& opt_position) {
        additional_cell_attributes.set_inhibitory_dendrites_position(opt_position);
    }

    [[nodiscard]] std::optional<Vec3d> get_excitatory_dendrites_position() const noexcept {
        return additional_cell_attributes.get_excitatory_dendrites_position();
    }

    void set_excitatory_dendrites_position(const std::optional<Vec3d>& opt_position) {
        additional_cell_attributes.set_excitatory_dendrites_position(opt_position);
    }

    [[nodiscard]] std::optional<Vec3d> get_axons_position_for(SignalType axon_type) const {
        return additional_cell_attributes.get_axons_position_for(axon_type);
    }

    /**
     * @brief Sets the dendrite position for both inhibitory and excitatory
     * @param opt_position The dendrite position, can be empty
     */
    void set_dendrite_position(const std::optional<Vec3d>& opt_position) noexcept {
        set_excitatory_dendrite_position(opt_position);
        set_inhibitory_dendrite_position(opt_position);
    }

    [[nodiscard]] std::optional<Vec3d> get_excitatory_axons_position() const noexcept {
        return additional_cell_attributes.get_excitatory_axons_position();
    }

    void set_excitatory_axons_position(const std::optional<Vec3d>& opt_position) noexcept {
        additional_cell_attributes.set_excitatory_axons_position(opt_position);
    }

    [[nodiscard]] std::optional<Vec3d> get_inhibitory_axons_position() const noexcept {
        return additional_cell_attributes.get_inhibitory_axons_position();
    }

    [[nodiscard]] std::optional<Vec3d> get_dendrites_position_for(SignalType dendrite_type) const {
        return additional_cell_attributes.get_dendrites_position_for(dendrite_type);
    }

    void set_inhibitory_axons_position(const std::optional<Vec3d>& opt_position) noexcept {
        additional_cell_attributes.set_inhibitory_axons_position(opt_position);
    }

    /**
     * @brief Returns the dendrite position, for which either both positions must be empty or equal
     * @exception Throws a RelearnException if one position is valid and the other one invalid or if both are valid with different values
     * @return The position of the dendrite, can be empty
     */
    [[nodiscard]] std::optional<Vec3d> get_dendrite_position() const {
        const auto& excitatory_dendrite_position_opt = get_excitatory_dendrite_position();
        const auto& inhibitory_dendrite_position_opt = get_inhibitory_dendrites_position();

        const bool ex_valid = excitatory_dendrite_position_opt.has_value();
        const bool in_valid = inhibitory_dendrite_position_opt.has_value();
        if (!ex_valid && !in_valid) {
            return {};
        }

        if (ex_valid && in_valid) {
            const auto& pos_ex = excitatory_dendrite_position_opt.value();
            const auto& pos_in = inhibitory_dendrite_position_opt.value();

            const auto diff = pos_ex - pos_in;
            const bool exc_position_equals_inh_position = diff.get_x() == 0.0 && diff.get_y() == 0.0 && diff.get_z() == 0.0;
            RelearnException::check(exc_position_equals_inh_position, "In get neuron position, positions are unequal");

            return pos_ex;
        }

        RelearnException::fail("In Cell, one pos was valid and one was not");

        return {};
    }
    /**
     * @brief Returns the number of free dendrites for the associated type in this cell
     * @return The number of free dendrites for the associated type
     */
    [[nodiscard]] unsigned int get_number_dendrites_for(SignalType dendrite_type) const noexcept {
        if (dendrite_type == SignalType::EXCITATORY) {
            return get_number_excitatory_dendrites();
        }

        return get_number_inhibitory_dendrites();
    }

    /**
     * @brief Prints the cell to the output stream
     * @param output_stream The output stream
     * @param cell The cell to print
     * @return The output stream after printing the cell
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const Cell<AdditionalCellAttributes>& cell) {
        const auto number_excitatory_dendrites = cell.get_number_excitatory_dendrites();
        const auto number_inhibitory_dendrites = cell.get_number_inhibitory_dendrites();

        const auto& position_excitatory_dendrites_opt = cell.get_excitatory_dendrite_position();
        const auto& position_inhibitory_dendrites_opt = cell.get_inhibitory_dendrites_position();

        const auto& position_excitatory_dendrites = position_excitatory_dendrites_opt.value();
        const auto& position_inhibitory_dendrites = position_inhibitory_dendrites_opt.value();

        Vec3d xyz_min{};
        Vec3d xyz_max{};

        std::tie(xyz_min, xyz_max) = cell.get_size();

        output_stream << "  == Cell (" << reinterpret_cast<size_t>(&cell) << " ==\n";
        output_stream << "\tMin: " << xyz_min << "\n\tMax: " << xyz_max << '\n';
        output_stream << "\tnumber_excitatory_dendrites: " << number_excitatory_dendrites << "\tPosition: " << position_excitatory_dendrites << '\n';
        output_stream << "\tnumber_inhibitory_dendrites: " << number_inhibitory_dendrites << "\tPosition: " << position_inhibitory_dendrites << '\n';

        return output_stream;
    }

private:
    /**
	 * ID of the neuron in the cell.
	 * This is only valid for cells that contain a normal neuron.
	 * For those with a super neuron, it has no meaning.
	 * This info is used to identify (return) the target neuron for a given axon
	 */
    size_t neuron_id{ Constants::uninitialized };

    // Two points describe size of cell
    Vec3d xyz_min{ Constants::uninitialized };
    Vec3d xyz_max{ Constants::uninitialized };

    AdditionalCellAttributes additional_cell_attributes{};

public:
    /**
     * @brief Returns the position of the excitatory dendrite
     * @return The position of the excitatory dendrite
     */
    [[nodiscard]] std::optional<Vec3d> get_excitatory_dendrite_position() const noexcept {
        return additional_cell_attributes.get_excitatory_dendrites_position();
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

        additional_cell_attributes.set_excitatory_dendrites_position(opt_position);
    }

    /**
     * @brief Returns the position of the inhibitory dendrite
     * @return The position of the inhibitory dendrite
     */
    [[nodiscard]] std::optional<Vec3d> get_inhibitory_dendrites_position() const noexcept {
        return additional_cell_attributes.get_inhibitory_dendrites_position();
    }

    void set_number_excitatory_axons(unsigned int num_axons) noexcept {
        additional_cell_attributes.set_number_excitatory_axons(num_axons);
    }

    [[nodiscard]] unsigned int get_number_excitatory_axons() const noexcept {
        return additional_cell_attributes.get_number_excitatory_axons();
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

        additional_cell_attributes.set_inhibitory_dendrites_position(opt_position);
    }

    void set_number_inhibitory_axons(unsigned int num_axons) noexcept {
        additional_cell_attributes.set_number_inhibitory_axons(num_axons);
    }

    [[nodiscard]] unsigned int get_number_inhibitory_axons() const noexcept {
        return additional_cell_attributes.get_number_inhibitory_axons();
    }

    [[nodiscard]] unsigned int get_number_axons_for(SignalType axon_type) const noexcept {
        return additional_cell_attributes.get_number_axons_for(axon_type);
    }

    /**
     * @brief Returns the position of the dendrite with the given signal type
     * @param dendrite_type The type of dendrite which's position should be returned
     * @return The position of the associated dendrite, can be empty
     */
    [[nodiscard]] std::optional<Vec3d> get_dendrite_position_for(SignalType dendrite_type) const noexcept {
        if (dendrite_type == SignalType::EXCITATORY) {
            return additional_cell_attributes.get_excitatory_dendrites_position();
        }
        return additional_cell_attributes.get_inhibitory_dendrites_position();
    }

    /**
     * @brief Sets the number of free excitatory dendrites in this cell
     * @param num_dendrites The number of free excitatory dendrites
     */
    void set_number_excitatory_dendrites(unsigned int num_dendrites) noexcept {
        additional_cell_attributes.set_number_excitatory_dendrites(num_dendrites);
    }

    /**
     * @brief Returns the number of free excitatory dendrites in this cell
     * @return The number of free excitatory dendrites
     */
    [[nodiscard]] unsigned int get_number_excitatory_dendrites() const noexcept {
        return additional_cell_attributes.get_number_excitatory_dendrites();
    }

    /**
     * @brief Sets the number of free inhibitory dendrites in this cell
     * @param num_dendrites The number of free inhibitory dendrites
     */
    void set_number_inhibitory_dendrites(unsigned int num_dendrites) noexcept {
        additional_cell_attributes.set_number_inhibitory_dendrites(num_dendrites);
    }

    /**
     * @brief Returns the number of free inhibitory dendrites in this cell
     * @return The number of free inhibitory dendrites
     */
    [[nodiscard]] unsigned int get_number_inhibitory_dendrites() const noexcept {
        return additional_cell_attributes.get_number_inhibitory_dendrites();
    }

    void set_hermite_coef_ex(unsigned int x, double d) {
        additional_cell_attributes.set_hermite_coef_ex(x, d);
    }

    void set_hermite_coef_in(unsigned int x, double d) {
        additional_cell_attributes.set_hermite_coef_in(x, d);
    }

    void set_hermite_coef_for(unsigned int x, double d, SignalType needed) {
        additional_cell_attributes.set_hermite_coef_for(x, d, needed);
    }

    double get_hermite_coef_ex(unsigned int x) const {
        return additional_cell_attributes.get_hermite_coef_ex(x);
    }

    double get_hermite_coef_in(unsigned int x) const {
        return additional_cell_attributes.get_hermite_coef_in(x);
    }

    double get_hermite_coef_for(unsigned int x, SignalType needed) const {
        return additional_cell_attributes.get_hermite_coef_for(x, needed);
    }
};
