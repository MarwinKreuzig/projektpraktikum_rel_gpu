/*
 * File:   SynapticElements.h
 * Author: naveau
 *
 * Created on September 26, 2014, 2:33 PM
 */

#ifndef SYNAPTICELEMENT_H
#define	SYNAPTICELEMENT_H

#include <assert.h>
#include <iostream>
#include <cstddef>
#include <cmath>
#include "MPIInfos.h"

class SynapticElements {

public:
	enum ElementType : int { AXON = 0, DENDRITE = 1 };
	enum SignalType : int { EXCITATORY = 0, INHIBITORY = 1 };

	SynapticElements(ElementType, size_t, double, double, double, double);
	~SynapticElements();

	double* get_cnts() const { return cnts; }
	double* get_connected_cnts() const { return connected_cnts; }
	double* get_delta_cnts() const { return delta_cnts; }
	SignalType* get_signal_types() { return signal_types; }

	double get_cnt(size_t neuron_id) { return cnts[neuron_id]; }
	double get_connected_cnt(size_t neuron_id) { return connected_cnts[neuron_id]; }
	double get_delta_cnt(size_t neuron_id) { return delta_cnts[neuron_id]; }
	enum SignalType get_signal_type(size_t neuron_id) { return signal_types[neuron_id]; }
	void set_signal_type(size_t neuron_id, SignalType signal_type) { signal_types[neuron_id] = signal_type; }
	ElementType get_element_type() { return type; }

	/**
	 * Updates the number of synaptic elements for neuron "neuron_id"
	 * Returns the number of synapses to be deleted as a consequence of deleting synaptic elements
	 *
	 * Synaptic elements are deleted based on "delta_cnts" in the following way:
	 * 1. Delete vacant elements
	 * 2. Delete bound elements
	 */
	inline unsigned int update_number_elements(size_t neuron_id) {
		assert(neuron_id < size);
		assert(cnts[neuron_id] >= 0);
		assert(connected_cnts[neuron_id] >= 0);
		assert(cnts[neuron_id] >= connected_cnts[neuron_id]);

		double num_vacant, num_vacant_plus_delta;
		double connected_cnt_old, connected_cnt_floor;
		unsigned int num_delete_connected;

		num_vacant = cnts[neuron_id] - connected_cnts[neuron_id];
		num_vacant_plus_delta = num_vacant + delta_cnts[neuron_id];

		// No deletion of bound synaptic elements required
		if (num_vacant_plus_delta >= 0) {
			cnts[neuron_id] = (1 - vacant_retract_ratio) * num_vacant_plus_delta + connected_cnts[neuron_id];
			num_delete_connected = 0;
		}
		// Delete bound synaptic elements if available
		else {
			connected_cnt_old = connected_cnts[neuron_id];

			/**
			 * More bound elements should be deleted than are available.
			 * Now, neither vacant (see if branch above) nor bound elements are left.
			 */
			if ((connected_cnts[neuron_id] + num_vacant_plus_delta) < 0) {
				connected_cnts[neuron_id] = 0;
				cnts[neuron_id] = 0;
			}
			else {
				connected_cnts[neuron_id] += num_vacant_plus_delta;             // Result is >= 0
				connected_cnt_floor = floor(connected_cnts[neuron_id]);  // Round down for integer value
				num_vacant = connected_cnts[neuron_id] - connected_cnt_floor;   // Amount lost by rounding down
				assert(num_vacant >= 0);

				connected_cnts[neuron_id] = connected_cnt_floor;
				cnts[neuron_id] = connected_cnts[neuron_id] + num_vacant;
			}
			if (connected_cnt_old < connected_cnts[neuron_id]) {
				std::cout << "connected_cnt_old: " << connected_cnt_old << "\n"
					<< "connected_cnts[neuron_id]: " << connected_cnts[neuron_id] << "\n";
			}
			assert(connected_cnt_old >= connected_cnts[neuron_id]);
			num_delete_connected = static_cast<unsigned int>(connected_cnt_old - connected_cnts[neuron_id]);
		}

		// Reset delta counts
		delta_cnts[neuron_id] = 0;

		assert(cnts[neuron_id] >= connected_cnts[neuron_id]);

		return num_delete_connected;
	}

	void update_number_elements_delta(double* calcium) {
		double inc;

		//std::cout << __func__ << ": ";

		// For my neurons
		for (size_t i = 0; i < this->size; ++i) {
			inc = gaussian_growth_curve(calcium[i], min_C_level_to_grow, C_target, nu);
			delta_cnts[i] += inc;
			//std::cout << delta_cnts[i] << " ";
		}
		//std::cout << std::endl;
		//std::cout << __func__ << "...[OK]" << std::endl;
	}

private:
	double gaussian_growth_curve(double Ca, double eta, double epsilon, double growth_rate) {
		/**
		 * gaussian_growth_curve generates a gaussian curve that is compressed by
		 * growth-factor nu and intersects the x-axis at
		 * eta (left intersection) and epsilon (right intersection).
		 * xi and zeta are helper variables that directly follow from eta and epsilon.
		 * See Butz and van Ooyen, 2013 PloS Comp Biol, Equation 4.
		 */

		double xi;
		double zeta;
		double dz;

		xi = (eta + epsilon) / 2;
		zeta = (eta - epsilon) / (2 * sqrt(-log(0.5)));

		dz = growth_rate * (2 * exp(-pow((Ca - xi) / zeta, 2)) - 1);
		//std::cout << "Ca: " << Ca << " dz: " << dz << " eta: " << eta << " epsilon: " << epsilon << " xi: " << xi << " zeta: " << zeta << "\n";
		return dz;
	}

	ElementType type;            // Denotes the type of all synaptic elements, which is AXON or DENDRITE
	size_t size;
	double* cnts;
	double* delta_cnts;          // Keeps track of changes in number of elements until those changes are applied in next connectivity update
	double* connected_cnts;
	SignalType* signal_types;    // Signal type of synaptic elements, i.e., EXCITATORY or INHIBITORY.
								 // Note: Given that currently exc. and inh. dendrites are in different objects, this would only be needed for axons.
								 //       A more memory-efficient solution would be to use a different class for axons which has the signal_types array.

	// Parameters
	double min_C_level_to_grow;   // Minimum level of calcium needed for elements to grow
	double C_target;              // Desired calcium level (possible extension of the model: Give all neurons individual C_target values!)
	double nu;                    // Growth rate for synaptic elements in ms^-1. Needs to be much smaller than 1 to separate activity and structural dynamics.
	double vacant_retract_ratio;  // Percentage of how many vacant synaptic elements should be deleted during each connectivity update
};

#endif	/* SYNAPTICELEMENT_H */

