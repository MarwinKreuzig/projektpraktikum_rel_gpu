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
#include "../io/LogFiles.h"

#include <sstream>

[[nodiscard]] std::optional<Vec3d> Cell::get_neuron_position() const {
    const bool dend_ex_valid = dendrites_ex.xyz_pos.has_value();
    const bool dend_in_valid = dendrites_in.xyz_pos.has_value();
    const bool ax_in_valid = axons_in.xyz_pos.has_value();
    const bool ax_ex_valid = axons_ex.xyz_pos.has_value();
    
    

    if (!dend_ex_valid && !dend_in_valid && !ax_ex_valid && ax_in_valid) {
        return {};
    }

    if (dend_ex_valid && dend_in_valid && ax_in_valid && ax_ex_valid) {
        const auto& dend_pos_ex = dendrites_ex.xyz_pos.value();
        const auto& dend_pos_in = dendrites_in.xyz_pos.value();
        const auto& ax_pos_ex = axons_ex.xyz_pos.value();
        const auto& ax_pos_in = axons_in.xyz_pos.value();

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

[[nodiscard]] std::optional<Vec3d> Cell::get_neuron_position_for(SignalType dendrite_type) const {
    if (dendrite_type == SignalType::EXCITATORY) {
        return dendrites_ex.xyz_pos;
    }

    return dendrites_in.xyz_pos;
}

[[nodiscard]] std::optional<Vec3d> Cell::get_neuron_dendrite_position_for(SignalType dendrite_type) const {
    if (dendrite_type == SignalType::EXCITATORY) {
        return dendrites_ex.xyz_pos;
    }

    return dendrites_in.xyz_pos;
}

[[nodiscard]] std::optional<Vec3d> Cell::get_neuron_axon_position_for(SignalType dendrite_type) const {
    if (dendrite_type == SignalType::EXCITATORY) {
        return axons_ex.xyz_pos;
    }

    return axons_in.xyz_pos;
}

[[nodiscard]] unsigned char Cell::get_octant_for_position(const Vec3d& pos) const {
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

[[nodiscard]] std::tuple<Vec3d, Vec3d> Cell::get_size_for_octant(unsigned char idx) const /*noexcept*/ {
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

void Cell::print() const {
    std::stringstream ss;

    ss << "  == Cell (" << this << ") ==\n";

    ss << "    xyz_min[3]: ";
    ss << xyz_min.get_x() << " ";
    ss << xyz_min.get_y() << " ";
    ss << xyz_min.get_z() << " ";
    ss << "\n";

    ss << "    xyz_max[3]: ";
    ss << xyz_max.get_x() << " ";
    ss << xyz_max.get_y() << " ";
    ss << xyz_max.get_z() << " ";
    ss << "\n";

    ss << "    dendrites_ex.num_dendrites: " << dendrites_ex.num_dendrites;
    ss << "    dendrites_ex.xyz_pos[3]   : ";
    ss << dendrites_ex.xyz_pos.value().get_x() << " ";
    ss << dendrites_ex.xyz_pos.value().get_y() << " ";
    ss << dendrites_ex.xyz_pos.value().get_z() << " ";
    ss << "\n";

    ss << "    dendrites_in.num_dendrites: " << dendrites_in.num_dendrites;
    ss << "    dendrites_in.xyz_pos[3]   : ";
    ss << dendrites_in.xyz_pos.value().get_x() << " ";
    ss << dendrites_in.xyz_pos.value().get_y() << " ";
    ss << dendrites_in.xyz_pos.value().get_z() << " ";
    ss << "\n";

    LogFiles::write_to_file(LogFiles::EventType::Cout, ss.str(), true);
}
