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

#include <iostream>

[[nodiscard]] std::tuple<Vec3d, bool> Cell::get_neuron_position() const {
    const auto diff = dendrites_in.xyz_pos - dendrites_ex.xyz_pos;

    const bool exc_position_equals_inh_position = diff.x == 0.0 && diff.y == 0.0 && diff.z == 0.0;

    RelearnException::check(exc_position_equals_inh_position);
    RelearnException::check(dendrites_ex.xyz_pos_valid == dendrites_in.xyz_pos_valid);

    const auto position = dendrites_ex.xyz_pos;
    const auto valid = dendrites_ex.xyz_pos_valid;

    return std::make_tuple(position, valid);
}

[[nodiscard]] std::tuple<Vec3d, bool> Cell::get_neuron_position_for(DendriteType dendrite_type) const {
    if (dendrite_type == DendriteType::EXCITATORY) {
        const auto position = dendrites_ex.xyz_pos;
        const auto valid = dendrites_ex.xyz_pos_valid;

        return std::make_tuple(position, valid);
    }

    RelearnException::check(dendrite_type == DendriteType::INHIBITORY);

    const auto position = dendrites_in.xyz_pos;
    const auto valid = dendrites_in.xyz_pos_valid;

    return std::make_tuple(position, valid);
}

[[nodiscard]] unsigned char Cell::get_neuron_octant() const {
    const auto diff = dendrites_in.xyz_pos - dendrites_ex.xyz_pos;

    const auto exc_position_equals_inh_position = diff.x == 0.0 && diff.y == 0.0 && diff.z == 0.0;
    RelearnException::check(exc_position_equals_inh_position);

    return get_octant_for_position(dendrites_in.xyz_pos);
}

[[nodiscard]] unsigned char Cell::get_octant_for_position(const Vec3d& pos) const {
    unsigned char idx = 0;

    const auto& x = pos.x;
    const auto& y = pos.y;
    const auto& z = pos.z;

    /**
	* Sanity check: Make sure that the position is within this cell
	* This check returns false if negative coordinates are used.
	* Thus make sure to use positions >=0.
	*/
    RelearnException::check(x >= xyz_min.x && x <= xyz_min.x + xyz_max.x);
    RelearnException::check(y >= xyz_min.y && y <= xyz_min.y + xyz_max.y);
    RelearnException::check(z >= xyz_min.z && z <= xyz_min.z + xyz_max.z);

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
    idx = idx | ((x < (xyz_min.x + xyz_max.x) / 2.0) ? 0 : 1); // idx | (pos_x < midpoint_dim_x) ? 0 : 1

    //NOLINTNEXTLINE
    idx = idx | ((y < (xyz_min.y + xyz_max.y) / 2.0) ? 0 : 2); // idx | (pos_y < midpoint_dim_y) ? 0 : 2

    //NOLINTNEXTLINE
    idx = idx | ((z < (xyz_min.z + xyz_max.z) / 2.0) ? 0 : 4); // idx | (pos_z < midpoint_dim_z) ? 0 : 4

    RelearnException::check(idx < Constants::number_oct, "Octree octant must be smaller than 8");

    return idx;
}

[[nodiscard]] std::tuple<Vec3d, Vec3d> Cell::get_size_for_octant(unsigned char idx) const /*noexcept*/ {
    const bool x_over_halfway_point = (idx & 1) != 0;
    const bool y_over_halfway_point = (idx & 2) != 0;
    const bool z_over_halfway_point = (idx & 4) != 0;

    Vec3d octant_xyz_min = this->xyz_min;
    Vec3d octant_xyz_max = this->xyz_max;
    // NOLINTNEXTLINE
    Vec3d octant_xyz_middle = (octant_xyz_min + octant_xyz_max) / 2.0;

    if (x_over_halfway_point) {
        octant_xyz_min.x = octant_xyz_middle.x;
    } else {
        octant_xyz_max.x = octant_xyz_middle.x;
    }

    if (y_over_halfway_point) {
        octant_xyz_min.y = octant_xyz_middle.y;
    } else {
        octant_xyz_max.y = octant_xyz_middle.y;
    }

    if (z_over_halfway_point) {
        octant_xyz_min.z = octant_xyz_middle.z;
    } else {
        octant_xyz_max.z = octant_xyz_middle.z;
    }

    return std::make_tuple(octant_xyz_min, octant_xyz_max);
}

void Cell::print() const {
    std::cout << "  == Cell (" << this << ") ==\n";

    std::cout << "    xyz_min[3]: ";
    for (int i = 0; i < 3; i++) {
        std::cout << xyz_min[i] << " ";
    }
    std::cout << "\n";

    std::cout << "    xyz_max[3]: ";
    for (int i = 0; i < 3; i++) {
        std::cout << xyz_max[i] << " ";
    }
    std::cout << "\n";

    std::cout << "    dendrites_ex.num_dendrites: " << dendrites_ex.num_dendrites;
    std::cout << "    dendrites_ex.xyz_pos[3]   : ";
    for (int i = 0; i < 3; i++) {
        std::cout << dendrites_ex.xyz_pos[i] << " ";
    }
    std::cout << "\n";

    std::cout << "    dendrites_in.num_dendrites: " << dendrites_in.num_dendrites;
    std::cout << "    dendrites_in.xyz_pos[3]   : ";
    for (int i = 0; i < 3; i++) {
        std::cout << dendrites_in.xyz_pos[i] << " ";
    }
    std::cout << "\n";
}
