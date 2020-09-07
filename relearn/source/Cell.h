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

class Cell {
public:
	enum DendriteType : int { EXCITATORY = 0, INHIBITORY = 1 };

	struct Dendrites {
		// All dendrites have the same position
		double xyz_pos[3];
		// Mark if xyz_pos[] values are valid and can be used
		bool xyz_pos_valid;
		unsigned int num_dendrites;
		// TODO
		// List colliding_axons;
	};

	Cell();
	~Cell();

	void set_size(double x_min, double y_min, double z_min,
		double x_max, double y_max, double z_max) {
		xyz_min[0] = x_min;
		xyz_min[1] = y_min;
		xyz_min[2] = z_min;
		xyz_max[0] = x_max;
		xyz_max[1] = y_max;
		xyz_max[2] = z_max;
	}

	void get_size(double* x_min, double* y_min, double* z_min,
		double* x_max, double* y_max, double* z_max) {
		*x_min = xyz_min[0];
		*y_min = xyz_min[1];
		*z_min = xyz_min[2];
		*x_max = xyz_max[0];
		*y_max = xyz_max[1];
		*z_max = xyz_max[2];
	}

	/**
	 * Returns edge length of the cell
	 * Assumes that the cell is cubic
	 */
	double get_length() {
		return xyz_max[0] - xyz_min[0];
	}

	void set_neuron_position(double x, double y, double z, bool valid) {
		dendrites[EXCITATORY].xyz_pos[0] = dendrites[INHIBITORY].xyz_pos[0] = x;
		dendrites[EXCITATORY].xyz_pos[1] = dendrites[INHIBITORY].xyz_pos[1] = y;
		dendrites[EXCITATORY].xyz_pos[2] = dendrites[INHIBITORY].xyz_pos[2] = z;
		dendrites[EXCITATORY].xyz_pos_valid = dendrites[INHIBITORY].xyz_pos_valid = valid;
	}

	void get_neuron_position(double* x, double* y, double* z, bool* valid) const {
		bool exc_position_equals_inh_position = dendrites[EXCITATORY].xyz_pos[0] == dendrites[INHIBITORY].xyz_pos[0] &&
			dendrites[EXCITATORY].xyz_pos[1] == dendrites[INHIBITORY].xyz_pos[1] &&
			dendrites[EXCITATORY].xyz_pos[2] == dendrites[INHIBITORY].xyz_pos[2];

		assert(exc_position_equals_inh_position);
		*x = dendrites[EXCITATORY].xyz_pos[0];
		*y = dendrites[EXCITATORY].xyz_pos[1];
		*z = dendrites[EXCITATORY].xyz_pos[2];
		assert(dendrites[EXCITATORY].xyz_pos_valid == dendrites[INHIBITORY].xyz_pos_valid);
		*valid = dendrites[EXCITATORY].xyz_pos_valid;
	}

	void get_neuron_position_exc(double* x, double* y, double* z, bool* valid) {
		*x = dendrites[EXCITATORY].xyz_pos[0];
		*y = dendrites[EXCITATORY].xyz_pos[1];
		*z = dendrites[EXCITATORY].xyz_pos[2];
		*valid = dendrites[EXCITATORY].xyz_pos_valid;
	}

	void set_neuron_position_exc(double x, double y, double z, bool valid) {
		dendrites[EXCITATORY].xyz_pos[0] = x;
		dendrites[EXCITATORY].xyz_pos[1] = y;
		dendrites[EXCITATORY].xyz_pos[2] = z;
		dendrites[EXCITATORY].xyz_pos_valid = valid;
	}

	void get_neuron_position_inh(double* x, double* y, double* z, bool* valid) {
		*x = dendrites[INHIBITORY].xyz_pos[0];
		*y = dendrites[INHIBITORY].xyz_pos[1];
		*z = dendrites[INHIBITORY].xyz_pos[2];
		*valid = dendrites[INHIBITORY].xyz_pos_valid;
	}

	void set_neuron_position_inh(double x, double y, double z, bool valid) {
		dendrites[INHIBITORY].xyz_pos[0] = x;
		dendrites[INHIBITORY].xyz_pos[1] = y;
		dendrites[INHIBITORY].xyz_pos[2] = z;
		dendrites[INHIBITORY].xyz_pos_valid = valid;
	}

	void get_neuron_position_for(DendriteType dendrite_type, double* xyz, bool* valid) const {
		for (int i = 0; i < 3; i++) {
			// Use dendrite_type as index into array
			xyz[i] = dendrites[dendrite_type].xyz_pos[i];
		}
		*valid = dendrites[dendrite_type].xyz_pos_valid;
	}

	void set_neuron_num_dendrites_exc(unsigned int num_dendrites) {
		dendrites[EXCITATORY].num_dendrites = num_dendrites;
	}

	unsigned int get_neuron_num_dendrites_exc() {
		return dendrites[EXCITATORY].num_dendrites;
	}

	void set_neuron_num_dendrites_inh(unsigned int num_dendrites) {
		dendrites[INHIBITORY].num_dendrites = num_dendrites;
	}

	unsigned int get_neuron_num_dendrites_inh() {
		return dendrites[INHIBITORY].num_dendrites;
	}

	unsigned int get_neuron_num_dendrites_for(DendriteType dendrite_type) const {
		return dendrites[dendrite_type].num_dendrites;
	}

	size_t get_neuron_id() const {
		return neuron_id;
	}

	void set_neuron_id(size_t neuron_id) {
		this->neuron_id = neuron_id;
	}

	unsigned char get_neuron_octant() {
		bool exc_position_equals_inh_position = dendrites[INHIBITORY].xyz_pos[0] == dendrites[EXCITATORY].xyz_pos[0] &&
			dendrites[INHIBITORY].xyz_pos[1] == dendrites[EXCITATORY].xyz_pos[1] &&
			dendrites[INHIBITORY].xyz_pos[2] == dendrites[EXCITATORY].xyz_pos[2];

		assert(exc_position_equals_inh_position);
		return get_octant_for_position(dendrites[INHIBITORY].xyz_pos[0], dendrites[INHIBITORY].xyz_pos[1], dendrites[INHIBITORY].xyz_pos[2]);
	}

	unsigned char get_octant_for_position(double x, double y, double z) {
		unsigned char idx = 0;

		/**
		 * Sanity check: Make sure that the position is within this cell
		 * This check returns false if negative coordinates are used.
		 * Thus make sure to use positions >=0.
		 */
		assert(x >= xyz_min[0] && x <= xyz_min[0] + xyz_max[0]);
		assert(y >= xyz_min[1] && y <= xyz_min[1] + xyz_max[1]);
		assert(z >= xyz_min[2] && z <= xyz_min[2] + xyz_max[2]);

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

		idx = idx | ((x < (xyz_min[0] + xyz_max[0]) / 2.0) ? 0 : 1);  // idx | (pos_x < midpoint_dim_x) ? 0 : 1
		idx = idx | ((y < (xyz_min[1] + xyz_max[1]) / 2.0) ? 0 : 2);  // idx | (pos_y < midpoint_dim_y) ? 0 : 2
		idx = idx | ((z < (xyz_min[2] + xyz_max[2]) / 2.0) ? 0 : 4);  // idx | (pos_z < midpoint_dim_z) ? 0 : 4

		return idx;
	}

	void get_size_for_octant(unsigned char idx, double xyz_min[3], double xyz_max[3]) {
		int i;
		unsigned char mask = 1;

		// Check whether 2nd or 1st octant for each dimension
		for (i = 0; i < 3; i++) {
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
	double xyz_min[3];
	double xyz_max[3];

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
