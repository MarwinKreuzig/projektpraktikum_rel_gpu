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
#include <tuple>

template <typename AdditionalCellAttributes = FastMultipoleMethodsCell>
class Cell {
public:
    [[nodiscard]] size_t get_neuron_id() const noexcept {
        return neuron_id;
    }

    void set_neuron_id(size_t neuron_id) noexcept {
        this->neuron_id = neuron_id;
    }

    [[nodiscard]] std::tuple<Vec3d, Vec3d> get_size() const noexcept {
        return std::make_tuple(xyz_min, xyz_max);
    }

    void set_size(const Vec3d& min, const Vec3d& max) {
        RelearnException::check(min.get_x() <= max.get_x(), "In Cell::set_size, x was not ok");
        RelearnException::check(min.get_y() <= max.get_y(), "In Cell::set_size, y was not ok");
        RelearnException::check(min.get_z() <= max.get_z(), "In Cell::set_size, z was not ok");

        xyz_min = min;
        xyz_max = max;
    }

    [[nodiscard]] double get_maximal_dimension_difference() const noexcept {
        const auto diff_vector = xyz_max - xyz_min;
        const auto diff = diff_vector.get_maximum();

        return diff;
    }

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

        //RelearnException::check(x >= xyz_min.get_x() && x <= xyz_min.get_x() + xyz_max.get_x(), "x is bad");
        //RelearnException::check(y >= xyz_min.get_y() && y <= xyz_min.get_y() + xyz_max.get_y(), "y is bad");
        //RelearnException::check(z >= xyz_min.get_z() && z <= xyz_min.get_z() + xyz_max.get_z(), "z is bad");

        /**
	     * Figure below shows the binary numbering of the octants (subcells) in a cell.
	     * The binary number of an octant (subcell) corresponds to its index [0..7] in the
	     * children array of the cell.
	     *

			   110 ----- 111
			   /|        /|
			  / |       / |
			 /  |      /  |
		   010 ----- 011  |    y
			|  100 ---|- 101   ^   z
			|  /      |  /     |
			| /       | /      | /
			|/        |/       |/
		   000 ----- 001       +-----> x

		 */

        //NOLINTNEXTLINE
        idx = idx | ((x < (xyz_min.get_x() + xyz_max.get_x()) / 2.0) ? 0 : 1); // idx | (pos_x < midpoint_dim_x) ? 0 : 1

        //NOLINTNEXTLINE
        idx = idx | ((y < (xyz_min.get_y() + xyz_max.get_y()) / 2.0) ? 0 : 2); // idx | (pos_y < midpoint_dim_y) ? 0 : 2

        //NOLINTNEXTLINE
        idx = idx | ((z < (xyz_min.get_z() + xyz_max.get_z()) / 2.0) ? 0 : 4); // idx | (pos_z < midpoint_dim_z) ? 0 : 4

        RelearnException::check(idx < Constants::number_oct, "Octree octant must be smaller than 8");

        return idx;
    }

    [[nodiscard]] std::tuple<Vec3d, Vec3d> get_size_for_octant(unsigned char idx) const /*noexcept*/ {
        const bool x_over_halfway_point = (idx & 1U) != 0;
        const bool y_over_halfway_point = (idx & 2U) != 0;
        const bool z_over_halfway_point = (idx & 4U) != 0;

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

    [[nodiscard]] unsigned char get_neuron_octant() const {
        const std::optional<Vec3d>& pos = get_neuron_position();
        RelearnException::check(pos.has_value(), "position didn_t have a value");
        return get_octant_for_position(pos.value());
    }

    [[nodiscard]] std::optional<Vec3d> get_neuron_position() const {
        const auto& dend_ex_pos = get_excitatory_dendrites_position();
        const auto& dend_in_pos = get_inhibitory_dendrites_position();
        const auto& ax_ex_pos = get_excitatory_axons_position();
        const auto& ax_in_pos = get_inhibitory_axons_position();

        const bool dend_ex_valid = dend_ex_pos.has_value();
        const bool dend_in_valid = dend_in_pos.has_value();
        const bool ax_in_valid = ax_ex_pos.has_value();
        const bool ax_ex_valid = ax_in_pos.has_value();

        if (!dend_ex_valid && !dend_in_valid && !ax_ex_valid && ax_in_valid) {
            return {};
        }

        if (dend_ex_valid && dend_in_valid && ax_in_valid && ax_ex_valid) {
            const auto& dend_pos_ex = dend_ex_pos.value();
            const auto& dend_pos_in = dend_in_pos.value();
            const auto& ax_pos_ex = ax_ex_pos.value();
            const auto& ax_pos_in = ax_in_pos.value();

            const auto dend_diff = dend_pos_ex - dend_pos_in;
            const bool dend_exc_position_equals_dend_inh_position = dend_diff.get_x() == 0.0 && dend_diff.get_y() == 0.0 && dend_diff.get_z() == 0.0;
            RelearnException::check(dend_exc_position_equals_dend_inh_position, "In get neuron position, dendrite positions are unequal");

            const auto ax_diff = ax_pos_ex - ax_pos_in;
            const bool ax_exc_position_equals_ax_inh_position = ax_diff.get_x() == 0.0 && ax_diff.get_y() == 0.0 && ax_diff.get_z() == 0.0;
            RelearnException::check(ax_exc_position_equals_ax_inh_position, "In get neuron position, axon positions are unequal");

            const auto diff = ax_pos_ex - dend_pos_ex;
            const bool ax_position_equals_dend_position = diff.get_x() == 0.0 && diff.get_y() == 0.0 && diff.get_z() == 0.0;
            RelearnException::check(ax_position_equals_dend_position, "In get neuron position, axon positions are unequal to dendrite positions");

            return dend_pos_ex;
        }

        RelearnException::fail("In Cell, one pos was valid and one was not");

        return {};
    }

    [[nodiscard]] std::optional<Vec3d> get_neuron_position_for(SignalType signal_type, ElementType element_type) const {
        if (element_type == ElementType::AXON) {
            return additional_attributes.get_axons_position_for(signal_type);
        }

        return additional_attributes.get_dendrites_position_for(signal_type);
    }

    void set_neuron_position(const std::optional<Vec3d>& opt_position) noexcept {
        set_excitatory_dendrites_position(opt_position);
        set_inhibitory_dendrites_position(opt_position);
        set_excitatory_axons_position(opt_position);
        set_inhibitory_axons_position(opt_position);
    }

    [[nodiscard]] std::optional<Vec3d> get_excitatory_dendrites_position() const noexcept {
        return additional_attributes.get_excitatory_dendrites_position();
    }

    void set_excitatory_dendrites_position(const std::optional<Vec3d>& opt_position) {
        additional_attributes.set_excitatory_dendrites_position(opt_position);
    }

    [[nodiscard]] std::optional<Vec3d> get_inhibitory_dendrites_position() const noexcept {
        return additional_attributes.get_inhibitory_dendrites_position();
    }

    void set_inhibitory_dendrites_position(const std::optional<Vec3d>& opt_position) {
        additional_attributes.set_inhibitory_dendrites_position(opt_position);
    }

    [[nodiscard]] std::optional<Vec3d> get_axons_position_for(SignalType axon_type) const {
        return additional_attributes.get_axons_position_for(axon_type);
    }

    [[nodiscard]] std::optional<Vec3d> get_excitatory_axons_position() const noexcept {
        return additional_attributes.get_excitatory_axons_position();
    }

    void set_excitatory_axons_position(const std::optional<Vec3d>& opt_position) noexcept {
        additional_attributes.set_excitatory_axons_position(opt_position);
    }

    [[nodiscard]] std::optional<Vec3d> get_inhibitory_axons_position() const noexcept {
        return additional_attributes.get_inhibitory_axons_position();
    }

    void set_inhibitory_axons_position(const std::optional<Vec3d>& opt_position) noexcept {
        additional_attributes.set_inhibitory_axons_position(opt_position);
    }

    [[nodiscard]] std::optional<Vec3d> get_dendrites_position_for(SignalType dendrite_type) const {
        return additional_attributes.get_dendrites_position_for(dendrite_type);
    }

    void set_number_excitatory_dendrites(unsigned int num_dendrites) noexcept {
        additional_attributes.set_number_excitatory_dendrites(num_dendrites);
    }

    [[nodiscard]] unsigned int get_number_excitatory_dendrites() const noexcept {
        return additional_attributes.get_number_excitatory_dendrites();
    }

    void set_number_inhibitory_dendrites(unsigned int num_dendrites) noexcept {
        additional_attributes.set_number_inhibitory_dendrites(num_dendrites);
    }

    [[nodiscard]] unsigned int get_number_inhibitory_dendrites() const noexcept {
        return additional_attributes.get_number_inhibitory_dendrites();
    }

    [[nodiscard]] unsigned int get_number_dendrites_for(SignalType dendrite_type) const noexcept {
        return additional_attributes.get_number_dendrites_for(dendrite_type);
    }

    void set_number_excitatory_axons(unsigned int num_axons) noexcept {
        additional_attributes.set_number_excitatory_axons(num_axons);
    }

    [[nodiscard]] unsigned int get_number_excitatory_axons() const noexcept {
        return additional_attributes.get_number_excitatory_axons();
    }

    void set_number_inhibitory_axons(unsigned int num_axons) noexcept {
        additional_attributes.set_number_inhibitory_axons(num_axons);
    }

    [[nodiscard]] unsigned int get_number_inhibitory_axons() const noexcept {
        return additional_attributes.get_number_inhibitory_axons();
    }

    [[nodiscard]] unsigned int get_number_axons_for(SignalType axon_type) const noexcept {
        return additional_attributes.get_number_axons_for(axon_type);
    }

private:
    // Two points describe size of cell
    Vec3d xyz_min{ Constants::uninitialized };
    Vec3d xyz_max{ Constants::uninitialized };

    AdditionalCellAttributes additional_attributes{};

    /**
	 * ID of the neuron in the cell.
	 * This is only valid for cells that contain a normal neuron.
	 * For those with a super neuron, it has no meaning.
	 * This info is used to identify (return) the target neuron for a given axon
	 */
    size_t neuron_id{ Constants::uninitialized };
};
