/*
 * File:   SynapticElements.h
 * Author: naveau
 *
 * Created on September 26, 2014, 2:33 PM
 */

#ifndef SYNAPTICELEMENT_H
#define	SYNAPTICELEMENT_H

#include <iostream>
#include <cstddef>
#include <cmath>
#include <vector>

#include "MPIInfos.h"
#include "RelearnException.h"

class NeuronMonitor;

class SynapticElements {
	friend class NeuronMonitor;

public:
	enum ElementType : int { AXON = 0, DENDRITE = 1 };
	enum SignalType : int { EXCITATORY = 0, INHIBITORY = 1 };

	SynapticElements(ElementType type, size_t s, double min_C_level_to_grow, double C_target, double nu, double vacant_retract_ratio) :
		type(type),
		size(s),
		min_C_level_to_grow(min_C_level_to_grow),
		C_target(C_target),
		nu(nu),
		vacant_retract_ratio(vacant_retract_ratio),
		cnts(size, 0.0), connected_cnts(size, 0.0), delta_cnts(size, 0.0), signal_types(size) {
	}

	SynapticElements(const SynapticElements& other) = delete;
	SynapticElements(SynapticElements&& other) = default;

	SynapticElements& operator = (const SynapticElements& other) = delete;
	SynapticElements& operator = (SynapticElements&& other) = default;

	~SynapticElements() = default;

	const double* get_cnts() const noexcept { return cnts.data(); }
	const double* get_connected_cnts() const noexcept { return connected_cnts.data(); }
	const double* get_delta_cnts() const noexcept { return delta_cnts.data(); }
	SignalType* get_signal_types() noexcept { return signal_types.data(); }

	void update_cnt(size_t neuron_id, double delta) {
		cnts[neuron_id] += delta;
		RelearnException::check(cnts[neuron_id] >= 0.0);
	}

	void update_conn_cnt(size_t neuron_id, double delta) {
		if (neuron_id == 94) {
			std::cerr << "94 update: type: " << type << " now: " << connected_cnts[neuron_id] << " delta: " << delta;
		}
		connected_cnts[neuron_id] += delta;
		if (neuron_id == 94) {
			std::cerr << " and now: " << connected_cnts[neuron_id] << std::endl;
		}
		RelearnException::check(connected_cnts[neuron_id] >= 0.0);
	}

	void update_delta_cnt(size_t neuron_id, double delta) {
		delta_cnts[neuron_id] += delta;
		RelearnException::check(delta_cnts[neuron_id] >= 0.0);
	}

	double get_cnt(size_t neuron_id) const noexcept { return cnts[neuron_id]; }
	double get_connected_cnt(size_t neuron_id) const noexcept { return connected_cnts[neuron_id]; }
	double get_delta_cnt(size_t neuron_id) const noexcept { return delta_cnts[neuron_id]; }
	enum SignalType get_signal_type(size_t neuron_id) const noexcept { return signal_types[neuron_id]; }
	void set_signal_type(size_t neuron_id, SignalType signal_type) noexcept { signal_types[neuron_id] = signal_type; }
	ElementType get_element_type() const noexcept { return type; }

	/**
	 * Updates the number of synaptic elements for neuron "neuron_id"
	 * Returns the number of synapses to be deleted as a consequence of deleting synaptic elements
	 *
	 * Synaptic elements are deleted based on "delta_cnts" in the following way:
	 * 1. Delete vacant elements
	 * 2. Delete bound elements
	 */
	unsigned int update_number_elements(size_t neuron_id) {
		RelearnException::check(neuron_id < size);
		RelearnException::check(cnts[neuron_id] >= 0, std::to_string(cnts[neuron_id]));
		RelearnException::check(connected_cnts[neuron_id] >= 0, std::to_string(connected_cnts[neuron_id]));
		RelearnException::check(cnts[neuron_id] >= connected_cnts[neuron_id], std::to_string(cnts[neuron_id] - connected_cnts[neuron_id]));

		unsigned int num_delete_connected = 0;

		double num_vacant = cnts[neuron_id] - connected_cnts[neuron_id];
		const double num_vacant_plus_delta = num_vacant + delta_cnts[neuron_id];

		// No deletion of bound synaptic elements required
		if (num_vacant_plus_delta >= 0) {
			cnts[neuron_id] = (1 - vacant_retract_ratio) * num_vacant_plus_delta + connected_cnts[neuron_id];
			num_delete_connected = 0;
		}
		// Delete bound synaptic elements if available
		else {
			const double connected_cnt_old = connected_cnts[neuron_id];

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
				const double connected_cnt_floor = floor(connected_cnts[neuron_id]);  // Round down for integer value
				num_vacant = connected_cnts[neuron_id] - connected_cnt_floor;   // Amount lost by rounding down

				RelearnException::check(num_vacant >= 0);

				connected_cnts[neuron_id] = connected_cnt_floor;
				cnts[neuron_id] = connected_cnts[neuron_id] + num_vacant;
			}
			if (connected_cnt_old < connected_cnts[neuron_id]) {
				std::cout << "connected_cnt_old: " << connected_cnt_old << "\n"
					<< "connected_cnts[neuron_id]: " << connected_cnts[neuron_id] << "\n";
			}

			RelearnException::check(connected_cnt_old >= connected_cnts[neuron_id]);
			num_delete_connected = static_cast<unsigned int>(connected_cnt_old - connected_cnts[neuron_id]);
		}

		// Reset delta counts
		delta_cnts[neuron_id] = 0;

		RelearnException::check(cnts[neuron_id] >= connected_cnts[neuron_id]);

		return num_delete_connected;
	}

	void update_number_elements_delta(const std::vector<double>& calcium) noexcept {
		// For my neurons
		for (size_t i = 0; i < this->size; ++i) {
			const auto inc = gaussian_growth_curve(calcium[i], min_C_level_to_grow, C_target, nu);
			delta_cnts[i] += inc;
		}
	}

private:
	double gaussian_growth_curve(double Ca, double eta, double epsilon, double growth_rate) const noexcept {
		/**
		 * gaussian_growth_curve generates a gaussian curve that is compressed by
		 * growth-factor nu and intersects the x-axis at
		 * eta (left intersection) and epsilon (right intersection).
		 * xi and zeta are helper variables that directly follow from eta and epsilon.
		 * See Butz and van Ooyen, 2013 PloS Comp Biol, Equation 4.
		 */

		const auto xi = (eta + epsilon) / 2;
		const auto zeta = (eta - epsilon) / (2 * sqrt(-log(0.5)));

		const auto dz = growth_rate * (2 * exp(-pow((Ca - xi) / zeta, 2)) - 1);
		return dz;
	}

	ElementType type;            // Denotes the type of all synaptic elements, which is AXON or DENDRITE
	size_t size;
	std::vector<double> cnts;
	std::vector<double> delta_cnts;          // Keeps track of changes in number of elements until those changes are applied in next connectivity update
	std::vector<double> connected_cnts;
	std::vector<SignalType> signal_types;    // Signal type of synaptic elements, i.e., EXCITATORY or INHIBITORY.
								 // Note: Given that currently exc. and inh. dendrites are in different objects, this would only be needed for axons.
								 //       A more memory-efficient solution would be to use a different class for axons which has the signal_types array.

	// Parameters
	double min_C_level_to_grow;   // Minimum level of calcium needed for elements to grow
	double C_target;              // Desired calcium level (possible extension of the model: Give all neurons individual C_target values!)
	double nu;                    // Growth rate for synaptic elements in ms^-1. Needs to be much smaller than 1 to separate activity and structural dynamics.
	double vacant_retract_ratio;  // Percentage of how many vacant synaptic elements should be deleted during each connectivity update
};

#endif	/* SYNAPTICELEMENT_H */

