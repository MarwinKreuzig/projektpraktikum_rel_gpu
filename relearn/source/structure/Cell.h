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
#include "../neurons/ElementType.h"
#include "../neurons/SignalType.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"

#include <optional>
#include <ostream>
#include <tuple>

/**
 * This class summarizes all 'octree-relevant' data from a neuron.
 * It contains a size in the octree (min and max), a neuron id (value Constants::uninitialized for virtual neurons, aka. inner nodes in the octree).
 * Depending on the template type, it also stores dendrite and axon positions, as well as calculated HermiteCoefficients.
 * AdditionalCellAttributes should be BarnesHutCell or FastMultipoleCell
 */
template <typename AdditionalCellAttributes>
class Cell {
public:
    using position_type = typename AdditionalCellAttributes::position_type;
    using counter_type = typename AdditionalCellAttributes::counter_type;
    using box_size_type = RelearnTypes::box_size_type;

    /**
     * @brief Sets the neuron id for the associated cell. Can be set to Constants::uninitialized to indicate a virtual neuron aka an inner node in the Octree
     * @param neuron_id The neuron id, can be Constants::uninitialized
     */
    void set_neuron_id(const size_t neuron_id) noexcept {
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
     * @param min The minimum index, y, and z of the sell
     * @param max The maximum index, y, and z of the sell
     * @exception Throws a RelearnException if one component of min is larger than the respective component of max
     */
    void set_size(const box_size_type& min, const box_size_type& max) {
        RelearnException::check(min.get_x() <= max.get_x(), "Cell::set_size: x was not ok");
        RelearnException::check(min.get_y() <= max.get_y(), "Cell::set_size: y was not ok");
        RelearnException::check(min.get_z() <= max.get_z(), "Cell::set_size: z was not ok");

        minimum_position = min;
        maximum_position = max;
    }

    /**
     * @brief Returns the size of the cell as tuple of (1) min and (2) max
     * @return The size of the cell as tuple of (1) min and (2) max
     */
    [[nodiscard]] std::tuple<box_size_type, box_size_type> get_size() const noexcept {
        return std::make_tuple(minimum_position, maximum_position);
    }

    /**
	 * @brief Returns maximum edge length of the cell, i.e., ||max - min||_1
	 * @return The maximum edge length of the cell
	 */
    [[nodiscard]] double get_maximal_dimension_difference() const noexcept {
        const auto diff_vector = maximum_position - minimum_position;
        const auto diff = diff_vector.get_maximum();

        return diff;
    }

    /**
     * @brief Calculates the octant for the position.
     * @param position The position inside the current cell which's octant position should be found
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
    [[nodiscard]] unsigned char get_octant_for_position(const box_size_type& position) const {
        unsigned char idx = 0;

        const auto& x = position.get_x();
        const auto& y = position.get_y();
        const auto& z = position.get_z();

        /**
	     * Sanity check: Make sure that the position is within this cell
	     * This check returns false if negative coordinates are used.
	     * Thus make sure to use positions >= 0.
	     */
        const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
        RelearnException::check(is_in_box, "Cell::get_octant_for_position: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);

        //NOLINTNEXTLINE
        idx = idx | ((x < (minimum_position.get_x() + maximum_position.get_x()) / 2.0) ? 0 : 1); // idx | (pos_x < midpoint_dim_x) ? 0 : 1

        //NOLINTNEXTLINE
        idx = idx | ((y < (minimum_position.get_y() + maximum_position.get_y()) / 2.0) ? 0 : 2); // idx | (pos_y < midpoint_dim_y) ? 0 : 2

        //NOLINTNEXTLINE
        idx = idx | ((z < (minimum_position.get_z() + maximum_position.get_z()) / 2.0) ? 0 : 4); // idx | (pos_z < midpoint_dim_z) ? 0 : 4

        RelearnException::check(idx < Constants::number_oct, "Cell::get_octant_for_position: Calculated octant is too large: {}", idx);

        return idx;
    }

    /**
     * @brief Returns the size of the cell in the in the given octant
     * @param octant The octant, between 0 and 7
     * @exception Throws a RelearnException if octant > Constants::number_oct
     * @return A tuple with (min, max) for the cell in the given octant
     */
    [[nodiscard]] std::tuple<box_size_type, box_size_type> get_size_for_octant(const unsigned char octant) const {
        RelearnException::check(octant <= Constants::number_oct, "Cell::get_size_for_octant: Octant was too large: {}", octant);

        const bool x_over_halfway_point = (octant & 1U) != 0;
        const bool y_over_halfway_point = (octant & 2U) != 0;
        const bool z_over_halfway_point = (octant & 4U) != 0;

        auto octant_xyz_min = this->minimum_position;
        auto octant_xyz_max = this->maximum_position;
        // NOLINTNEXTLINE
        const auto& octant_xyz_middle = (octant_xyz_min + octant_xyz_max) / 2.0;

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

    /**
     * @brief Prints the cell to the output stream
     * @param output_stream The output stream
     * @param cell The cell to print
     * @return The output stream after printing the cell
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const Cell<AdditionalCellAttributes>& cell) {
        const auto number_excitatory_dendrites = cell.get_number_excitatory_dendrites();
        const auto number_inhibitory_dendrites = cell.get_number_inhibitory_dendrites();

        const auto& position_excitatory_dendrites_opt = cell.get_excitatory_dendrites_position();
        const auto& position_inhibitory_dendrites_opt = cell.get_inhibitory_dendrites_position();

        const auto& position_excitatory_dendrites = position_excitatory_dendrites_opt.value();
        const auto& position_inhibitory_dendrites = position_inhibitory_dendrites_opt.value();

        const auto& [minimum_position, maximum_position] = cell.get_size();

        // NOLINTNEXTLINE
        output_stream << "  == Cell (" << reinterpret_cast<size_t>(&cell) << " ==\n";
        output_stream << "\tMin: " << minimum_position << "\n\tMax: " << maximum_position << '\n';
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
    box_size_type minimum_position{ Constants::uninitialized };
    box_size_type maximum_position{ Constants::uninitialized };

    AdditionalCellAttributes additional_cell_attributes{};

public:
    /**
     * @brief Sets the number of free excitatory dendrites in this cell
     * @param num_dendrites The number of free excitatory dendrites
     */
    void set_number_excitatory_dendrites(const counter_type num_dendrites) noexcept {
        additional_cell_attributes.set_number_excitatory_dendrites(num_dendrites);
    }

    /**
     * @brief Returns the number of free excitatory dendrites in this cell
     * @return The number of free excitatory dendrites
     */
    [[nodiscard]] counter_type get_number_excitatory_dendrites() const noexcept {
        return additional_cell_attributes.get_number_excitatory_dendrites();
    }

    /**
     * @brief Sets the number of free inhibitory dendrites in this cell
     * @param num_dendrites The number of free inhibitory dendrites
     */
    void set_number_inhibitory_dendrites(const counter_type num_dendrites) noexcept {
        additional_cell_attributes.set_number_inhibitory_dendrites(num_dendrites);
    }

    /**
     * @brief Returns the number of free inhibitory dendrites in this cell
     * @return The number of free inhibitory dendrites
     */
    [[nodiscard]] counter_type get_number_inhibitory_dendrites() const noexcept {
        return additional_cell_attributes.get_number_inhibitory_dendrites();
    }

    /**
     * @brief Returns the number of free dendrites for the associated type in this cell
     * @param dendrite_type The requested dendrite type
     * @return The number of free dendrites for the associated type
     */
    [[nodiscard]] counter_type get_number_dendrites_for(const SignalType dendrite_type) const noexcept {
        return additional_cell_attributes.get_number_dendrites_for(dendrite_type);
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param opt_position The new position of the excitatory dendrite
     * @exception Throws a RelearnException if the position is valid but not within the box
     */
    void set_excitatory_dendrites_position(const std::optional<position_type>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();
            const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
            RelearnException::check(is_in_box, "Cell::set_excitatory_dendrites_position: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);
        }

        additional_cell_attributes.set_excitatory_dendrites_position(opt_position);
    }

    /**
     * @brief Returns the position of the excitatory dendrite
     * @return The position of the excitatory dendrite
     */
    [[nodiscard]] std::optional<position_type> get_excitatory_dendrites_position() const noexcept {
        return additional_cell_attributes.get_excitatory_dendrites_position();
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param opt_position The new position of the inhibitory dendrite
     * @exception Throws a RelearnException if the position is valid but not within the box
     */
    void set_inhibitory_dendrites_position(const std::optional<position_type>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();
            const auto is_in_box = position.check_in_box(minimum_position, maximum_position);
            RelearnException::check(is_in_box, "Cell::set_inhibitory_dendrites_position: position is not in box: {} in [{}, {}]", position, minimum_position, maximum_position);
        }

        additional_cell_attributes.set_inhibitory_dendrites_position(opt_position);
    }

    /**
     * @brief Returns the position of the inhibitory dendrite
     * @return The position of the inhibitory dendrite
     */
    [[nodiscard]] std::optional<position_type> get_inhibitory_dendrites_position() const noexcept {
        return additional_cell_attributes.get_inhibitory_dendrites_position();
    }

    /**
     * @brief Returns the position of the dendrite with the given signal type
     * @param dendrite_type The type of dendrite which's position should be returned
     * @return The position of the associated dendrite, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_dendrites_position_for(const SignalType dendrite_type) const noexcept {
        return additional_cell_attributes.get_dendrites_position_for(dendrite_type);
    }

    /**
     * @brief Sets the dendrite position for both inhibitory and excitatory
     * @param opt_position The dendrite position, can be empty
     */
    void set_dendrites_position(const std::optional<position_type>& opt_position) {
        set_excitatory_dendrites_position(opt_position);
        set_inhibitory_dendrites_position(opt_position);
    }

    /**
     * @brief Returns the dendrite position, for which either both positions must be empty or equal
     * @exception Throws a RelearnException if one position is valid and the other one invalid or if both are valid with different values
     * @return The position of the dendrite, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_dendrites_position() const {
        const auto& excitatory_dendrites_position_opt = get_excitatory_dendrites_position();
        const auto& inhibitory_dendrites_position_opt = get_inhibitory_dendrites_position();

        const bool ex_valid = excitatory_dendrites_position_opt.has_value();
        const bool in_valid = inhibitory_dendrites_position_opt.has_value();
        if (!ex_valid && !in_valid) {
            return {};
        }

        if (ex_valid && in_valid) {
            const auto& pos_ex = excitatory_dendrites_position_opt.value();
            const auto& pos_in = inhibitory_dendrites_position_opt.value();

            const auto diff = pos_ex - pos_in;
            const bool exc_position_equals_inh_position = diff.get_x() == 0.0 && diff.get_y() == 0.0 && diff.get_z() == 0.0;
            RelearnException::check(exc_position_equals_inh_position, "Cell::get_dendrites_positions: positions are unequal");

            return pos_ex;
        }

        RelearnException::fail("In Cell, one pos was valid and one was not");

        return {};
    }

    /**
     * @brief Sets the number of free excitatory axons in this cell
     * @param num_axons The number of free excitatory axons
     */
    void set_number_excitatory_axons(const counter_type num_axons) noexcept {
        additional_cell_attributes.set_number_excitatory_axons(num_axons);
    }

    /**
     * @brief Returns the number of free excitatory axons in this cell
     * @return The number of free excitatory axons
     */
    [[nodiscard]] counter_type get_number_excitatory_axons() const noexcept {
        return additional_cell_attributes.get_number_excitatory_axons();
    }

    /**
     * @brief Sets the number of free inhibitory axons in this cell
     * @param num_dendrites The number of free inhibitory axons
     */
    void set_number_inhibitory_axons(const counter_type num_axons) noexcept {
        additional_cell_attributes.set_number_inhibitory_axons(num_axons);
    }

    /**
     * @brief Returns the number of free inhibitory axons in this cell
     * @return The number of free inhibitory axons
     */
    [[nodiscard]] counter_type get_number_inhibitory_axons() const noexcept {
        return additional_cell_attributes.get_number_inhibitory_axons();
    }

    /**
     * @brief Returns the number of free axons for the associated type in this cell
     * @param axon_type The requested axons type
     * @return The number of free axons for the associated type
     */
    [[nodiscard]] counter_type get_number_axons_for(const SignalType axon_type) const noexcept {
        return additional_cell_attributes.get_number_axons_for(axon_type);
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param opt_position The new position of the excitatory axons
     * @exception Throws a RelearnException if the position is valid but not within the box
     */
    void set_excitatory_axons_position(const std::optional<position_type>& opt_position) noexcept {
        additional_cell_attributes.set_excitatory_axons_position(opt_position);
    }

    /**
     * @brief Returns the position of the excitatory axons
     * @return The position of the excitatory axons
     */
    [[nodiscard]] std::optional<position_type> get_excitatory_axons_position() const noexcept {
        return additional_cell_attributes.get_excitatory_axons_position();
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param opt_position The new position of the inhibitory axons
     * @exception Throws a RelearnException if the position is valid but not within the box
     */
    void set_inhibitory_axons_position(const std::optional<position_type>& opt_position) noexcept {
        additional_cell_attributes.set_inhibitory_axons_position(opt_position);
    }

    /**
     * @brief Returns the position of the inhibitory axons
     * @return The position of the inhibitory axons
     */
    [[nodiscard]] std::optional<position_type> get_inhibitory_axons_position() const noexcept {
        return additional_cell_attributes.get_inhibitory_axons_position();
    }

    /**
     * @brief Returns the position of the axons with the given signal type
     * @param dendrite_type The type of axons which's position should be returned
     * @return The position of the associated axons, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_axons_position_for(const SignalType axon_type) const {
        return additional_cell_attributes.get_axons_position_for(axon_type);
    }

    /**
     * @brief Returns the axons position, for which either both positions must be empty or equal
     * @exception Throws a RelearnException if one position is valid and the other one invalid or if both are valid with different values
     * @return The position of the axons, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_axons_position() const {
        const auto& excitatory_axons_position_opt = get_excitatory_axons_position();
        const auto& inhibitory_axons_position_opt = get_inhibitory_axons_position();

        const bool ex_valid = excitatory_axons_position_opt.has_value();
        const bool in_valid = inhibitory_axons_position_opt.has_value();

        if (!ex_valid && !in_valid) {
            return {};
        }

        if (ex_valid && in_valid) {
            const auto& pos_ex = excitatory_axons_position_opt.value();
            const auto& pos_in = inhibitory_axons_position_opt.value();

            const auto diff = pos_ex - pos_in;
            const bool exc_position_equals_inh_position = diff.get_x() == 0.0 && diff.get_y() == 0.0 && diff.get_z() == 0.0;
            RelearnException::check(exc_position_equals_inh_position, "Cell::get_axons_position: positions are unequal");

            return pos_ex;
        }

        RelearnException::fail("In Cell, one pos was valid and one was not");

        return {};
    }

    /**
     * @brief Sets the position of the neuron for every necessary part of the cell
     * @param opt_position The position, can be empty
     */
    void set_neuron_position(const std::optional<position_type>& opt_position) noexcept {
        additional_cell_attributes.set_neuron_position(opt_position);
    }
};
