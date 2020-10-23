/*
 * File:   Cell.h
 * Author: rinke
 *
 * Created on Oct 28, 2014
 */

#ifndef CELL_H
#define CELL_H

#include <cassert>
#include <iostream>

#include "Vec3.h"

class Cell {
public:
	enum DendriteType : int { EXCITATORY = 0, INHIBITORY = 1 };

	struct Dendrites {
		// All dendrites have the same position
		Vec3d xyz_pos{};
		// Mark if xyz_pos[] values are valid and can be used
		bool xyz_pos_valid = false;
		unsigned int num_dendrites = 0;
		// TODO
		// List colliding_axons;
	};

	Cell() noexcept : neuron_id(1111222233334444) {
	}

	~Cell() = default;
	 
	Cell(const Cell& other) = default;
	Cell(Cell&& other) = default;

	Cell& operator=(const Cell& other) = default;
	Cell& operator=(Cell&& other) = default;

	void set_size(const Vec3d& min, const Vec3d& max) noexcept {
		xyz_min = min;
		xyz_max = max;
	}

	void get_size(Vec3d& min, Vec3d& max) const noexcept {
		min = xyz_min;
		max = xyz_max;
	}

	/**
	 * Returns edge length of the cell
	 * Assumes that the cell is cubic
	 */
	double get_length() const noexcept {
		const auto diff_vector = xyz_max - xyz_min;
		const auto diff = diff_vector.get_maximum();
		//assert(diff.x == diff.y && diff.y == diff.z);
		return diff;
	}

	void set_neuron_position(const Vec3d& pos, bool valid) noexcept {
		set_neuron_position_exc(pos, valid);
		set_neuron_position_inh(pos, valid);
	}

	void get_neuron_position(Vec3d& pos, bool& valid) const noexcept {
		const auto diff = dendrites[INHIBITORY].xyz_pos - dendrites[EXCITATORY].xyz_pos;

		const bool exc_position_equals_inh_position = diff.x == 0.0 && diff.y == 0.0 && diff.z == 0.0;
		assert(exc_position_equals_inh_position);
		assert(dendrites[EXCITATORY].xyz_pos_valid == dendrites[INHIBITORY].xyz_pos_valid);

		pos = dendrites[EXCITATORY].xyz_pos;
		valid = dendrites[EXCITATORY].xyz_pos_valid;
	}

	void get_neuron_position_exc(Vec3d& position, bool& valid) const noexcept {
		position = dendrites[EXCITATORY].xyz_pos;
		valid = dendrites[EXCITATORY].xyz_pos_valid;
	}

	void set_neuron_position_exc(const Vec3d& position, bool valid) noexcept {
		dendrites[EXCITATORY].xyz_pos = position;
		dendrites[EXCITATORY].xyz_pos_valid = valid;
	}

	void get_neuron_position_inh(Vec3d& position, bool& valid) const noexcept {
		position = dendrites[INHIBITORY].xyz_pos;
		valid = dendrites[INHIBITORY].xyz_pos_valid;
	}

	void set_neuron_position_inh(const Vec3d& position, bool valid) noexcept {
		dendrites[INHIBITORY].xyz_pos = position;
		dendrites[INHIBITORY].xyz_pos_valid = valid;
	}

	void get_neuron_position_for(const DendriteType dendrite_type, Vec3d& xyz, bool& valid) const noexcept {
		// Use dendrite_type as index into array
		xyz = dendrites[dendrite_type].xyz_pos;
		valid = dendrites[dendrite_type].xyz_pos_valid;
	}

	void set_neuron_num_dendrites_exc(const unsigned int num_dendrites) noexcept {
		dendrites[EXCITATORY].num_dendrites = num_dendrites;
	}

	unsigned int get_neuron_num_dendrites_exc() const noexcept {
		return dendrites[EXCITATORY].num_dendrites;
	}

	void set_neuron_num_dendrites_inh(const unsigned int num_dendrites) noexcept {
		dendrites[INHIBITORY].num_dendrites = num_dendrites;
	}

	unsigned int get_neuron_num_dendrites_inh() const noexcept {
		return dendrites[INHIBITORY].num_dendrites;
	}

	unsigned int get_neuron_num_dendrites_for(const DendriteType dendrite_type) const noexcept {
		return dendrites[dendrite_type].num_dendrites;
	}

	size_t get_neuron_id() const noexcept {
		return neuron_id;
	}

	void set_neuron_id(size_t neuron_id) noexcept {
		this->neuron_id = neuron_id;
	}

	unsigned char get_neuron_octant() const noexcept {
		const auto diff = dendrites[INHIBITORY].xyz_pos - dendrites[EXCITATORY].xyz_pos;

		const auto exc_position_equals_inh_position = diff.x == 0.0 && diff.y == 0.0 && diff.z == 0.0;
		assert(exc_position_equals_inh_position);

		return get_octant_for_position(dendrites[INHIBITORY].xyz_pos);
	}

	unsigned char get_octant_for_position(const Vec3d& pos) const noexcept {
		unsigned char idx = 0;

		const auto& x = pos.x;
		const auto& y = pos.y;
		const auto& z = pos.z;

		/**
		 * Sanity check: Make sure that the position is within this cell
		 * This check returns false if negative coordinates are used.
		 * Thus make sure to use positions >=0.
		 */
		assert(x >= xyz_min.x && x <= xyz_min.x + xyz_max.x);
		assert(y >= xyz_min.y && y <= xyz_min.y + xyz_max.y);
		assert(z >= xyz_min.z && z <= xyz_min.z + xyz_max.z);

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

		idx = idx | ((x < (xyz_min.x + xyz_max.x) / 2.0) ? 0 : 1);  // idx | (pos_x < midpoint_dim_x) ? 0 : 1
		idx = idx | ((y < (xyz_min.y + xyz_max.y) / 2.0) ? 0 : 2);  // idx | (pos_y < midpoint_dim_y) ? 0 : 2
		idx = idx | ((z < (xyz_min.z + xyz_max.z) / 2.0) ? 0 : 4);  // idx | (pos_z < midpoint_dim_z) ? 0 : 4

		return idx;
	}

	void get_size_for_octant(unsigned char idx, Vec3d& xyz_min, Vec3d& xyz_max) const noexcept {
		unsigned char mask = 1;

		// Check whether 2nd or 1st octant for each dimension
		for (auto i = 0; i < 3; i++) {
			// Use bit mask "mask" to see which bit is set for idx
			if (mask & idx) {
				xyz_min[i] = (this->xyz_min[i] + this->xyz_max[i]) / 2.0;
				xyz_max[i] = this->xyz_max[i];
			}
			else {
				xyz_min[i] = this->xyz_min[i];
				xyz_max[i] = (this->xyz_min[i] + this->xyz_max[i]) / 2.0;
			}

			mask <<= 1;
		}
	}

	void print() const {
		using namespace std;

		cout << "  == Cell (" << this << ") ==\n";

		cout << "    xyz_min[3]: ";
		for (int i = 0; i < 3; i++) {
			cout << xyz_min[i] << " ";
		}
		cout << "\n";

		cout << "    xyz_max[3]: ";
		for (int i = 0; i < 3; i++) {
			cout << xyz_max[i] << " ";
		}
		cout << "\n";

		cout << "    dendrites[EXCITATORY].num_dendrites: " << dendrites[EXCITATORY].num_dendrites;
		cout << "    dendrites[EXCITATORY].xyz_pos[3]   : ";
		for (int i = 0; i < 3; i++) {
			cout << dendrites[EXCITATORY].xyz_pos[i] << " ";
		}
		cout << "\n";

		cout << "    dendrites[INHIBITORY].num_dendrites: " << dendrites[INHIBITORY].num_dendrites;
		cout << "    dendrites[INHIBITORY].xyz_pos[3]   : ";
		for (int i = 0; i < 3; i++) {
			cout << dendrites[INHIBITORY].xyz_pos[i] << " ";
		}
		cout << "\n";
	}

private:
	// Two points describe size of cell
	Vec3d xyz_min;
	Vec3d xyz_max;

	/**
	 * Cell contains info for one neuron, which could be a "super" neuron
	 *
	 * Info about EXCITATORY dendrites at dendrites[0]
	 * Info about INHIBITORY dendrites at dendrites[1]
	 *
	 * Type DendriteType (see declaration) is used as indices to access the array elements
	 */
	Dendrites dendrites[2];

	/**
	 * ID of the neuron in the cell.
	 * This is only valid for cells that contain a normal neuron.
	 * For those with a super neuron, it has no meaning.
	 * This info is used to identify (return) the target neuron for a given axon
	 */
	size_t neuron_id;
};

#endif /* CELL_H */
