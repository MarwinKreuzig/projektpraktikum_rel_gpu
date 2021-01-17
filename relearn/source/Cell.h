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

#include "Commons.h"
#include "RelearnException.h"
#include "Vec3.h"

#include <iostream>
#include <tuple>

class Cell {
public:
    enum class DendriteType : char {
        EXCITATORY,
        INHIBITORY
    };

    struct Dendrites {
        // All dendrites have the same position
        Vec3d xyz_pos {};
        // Mark if xyz_pos[] values are valid and can be used
        bool xyz_pos_valid = false;
        unsigned int num_dendrites = 0;
        // TODO(future)
        // List colliding_axons;
    };

    Cell() = default;
    ~Cell() = default;

    Cell(const Cell& other) = default;
    Cell(Cell&& other) = default;

    Cell& operator=(const Cell& other) = default;
    Cell& operator=(Cell&& other) = default;

    void set_size(const Vec3d& min, const Vec3d& max) noexcept {
        xyz_min = min;
        xyz_max = max;
    }

    [[nodiscard]] std::tuple<Vec3d, Vec3d> get_size() const noexcept {
        return std::make_tuple(xyz_min, xyz_max);
    }

    /**
	 * Returns edge length of the cell
	 * Assumes that the cell is cubic
	 */
    [[nodiscard]] double get_length() const noexcept {
        const auto diff_vector = xyz_max - xyz_min;
        const auto diff = diff_vector.get_maximum();

        return diff;
    }

    void set_neuron_position(const Vec3d& pos, bool valid) noexcept {
        set_neuron_position_exc(pos, valid);
        set_neuron_position_inh(pos, valid);
    }

    [[nodiscard]] std::tuple<Vec3d, bool> get_neuron_position() const {
        const auto diff = dendrites_in.xyz_pos - dendrites_ex.xyz_pos;

        const bool exc_position_equals_inh_position = diff.x == 0.0 && diff.y == 0.0 && diff.z == 0.0;

        RelearnException::check(exc_position_equals_inh_position);
        RelearnException::check(dendrites_ex.xyz_pos_valid == dendrites_in.xyz_pos_valid);

        const auto position = dendrites_ex.xyz_pos;
        const auto valid = dendrites_ex.xyz_pos_valid;

        return std::make_tuple(position, valid);
    }

    [[nodiscard]] std::tuple<Vec3d, bool> get_neuron_position_exc() const noexcept {
        const auto position = dendrites_ex.xyz_pos;
        const auto valid = dendrites_ex.xyz_pos_valid;

        return std::make_tuple(position, valid);
    }

    void set_neuron_position_exc(const Vec3d& position, bool valid) noexcept {
        dendrites_ex.xyz_pos = position;
        dendrites_ex.xyz_pos_valid = valid;
    }

    [[nodiscard]] std::tuple<Vec3d, bool> get_neuron_position_inh() const noexcept {
        const auto position = dendrites_in.xyz_pos;
        const auto valid = dendrites_in.xyz_pos_valid;

        return std::make_tuple(position, valid);
    }

    void set_neuron_position_inh(const Vec3d& position, bool valid) noexcept {
        dendrites_in.xyz_pos = position;
        dendrites_in.xyz_pos_valid = valid;
    }

    [[nodiscard]] std::tuple<Vec3d, bool> get_neuron_position_for(DendriteType dendrite_type) const {
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

    void set_neuron_num_dendrites_exc(unsigned int num_dendrites) noexcept {
        dendrites_ex.num_dendrites = num_dendrites;
    }

    [[nodiscard]] unsigned int get_neuron_num_dendrites_exc() const noexcept {
        return dendrites_ex.num_dendrites;
    }

    void set_neuron_num_dendrites_inh(unsigned int num_dendrites) noexcept {
        dendrites_in.num_dendrites = num_dendrites;
    }

    [[nodiscard]] unsigned int get_neuron_num_dendrites_inh() const noexcept {
        return dendrites_in.num_dendrites;
    }

    [[nodiscard]] unsigned int get_neuron_num_dendrites_for(DendriteType dendrite_type) const {
        if (dendrite_type == DendriteType::EXCITATORY) {
            return dendrites_ex.num_dendrites;
        }

        RelearnException::check(dendrite_type == DendriteType::INHIBITORY);

        return dendrites_in.num_dendrites;
    }

    [[nodiscard]] size_t get_neuron_id() const noexcept {
        return neuron_id;
    }

    void set_neuron_id(size_t neuron_id) noexcept {
        this->neuron_id = neuron_id;
    }

    [[nodiscard]] unsigned char get_neuron_octant() const {
        const auto diff = dendrites_in.xyz_pos - dendrites_ex.xyz_pos;

        const auto exc_position_equals_inh_position = diff.x == 0.0 && diff.y == 0.0 && diff.z == 0.0;
        RelearnException::check(exc_position_equals_inh_position);

        return get_octant_for_position(dendrites_in.xyz_pos);
    }

    [[nodiscard]] unsigned char get_octant_for_position(const Vec3d& pos) const {
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

    [[nodiscard]] std::tuple<Vec3d, Vec3d> get_size_for_octant(unsigned char idx) const /*noexcept*/ {
        Vec3d xyz_min;
        Vec3d xyz_max;
        unsigned char mask = 1;

        // Check whether 2nd or 1st octant for each dimension
        for (auto i = 0; i < 3; i++) {
            // Use bit mask "mask" to see which bit is set for idx
            if ((mask & idx) != 0) {
                xyz_min[i] = (this->xyz_min[i] + this->xyz_max[i]) / 2.0;
                xyz_max[i] = this->xyz_max[i];
            } else {
                xyz_min[i] = this->xyz_min[i];
                xyz_max[i] = (this->xyz_min[i] + this->xyz_max[i]) / 2.0;
            }

            mask <<= 1U;
        }

        return std::make_tuple(xyz_min, xyz_max);
    }

    void print() const {
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

private:
    // Two points describe size of cell
    Vec3d xyz_min;
    Vec3d xyz_max;

    /**
	 * Cell contains info for one neuron, which could be a "super" neuron
	 *
	 * Info about EXCITATORY dendrites: dendrites_ex
	 * Info about INHIBITORY dendrites: dendrites_in
	 */
    Dendrites dendrites_ex;
    Dendrites dendrites_in;

    /**
	 * ID of the neuron in the cell.
	 * This is only valid for cells that contain a normal neuron.
	 * For those with a super neuron, it has no meaning.
	 * This info is used to identify (return) the target neuron for a given axon
	 */
    size_t neuron_id { Constants::uninitialized };
};
