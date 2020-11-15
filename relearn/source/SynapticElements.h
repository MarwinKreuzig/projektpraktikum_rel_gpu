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

#include "MPIInfos.h"
#include "RelearnException.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

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

	void update_conn_cnt(size_t neuron_id, double delta, const char* mess) {
		if (neuron_id == 94) {
			std::cerr << "94 update: type: " << mess << " now: " << connected_cnts[neuron_id] << " delta: " << delta;
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

		const double current_count = cnts[neuron_id];
		const double current_connected_count = connected_cnts[neuron_id];
		const double current_vacant = current_count - current_connected_count;
		const double current_delta = delta_cnts[neuron_id];

		RelearnException::check(current_count >= 0.0, std::to_string(current_count));
		RelearnException::check(current_connected_count >= 0.0, std::to_string(current_connected_count));
		RelearnException::check(current_vacant >= 0.0, std::to_string(current_count - current_connected_count));

		// The vacant portion after caring for the delta
		const double new_vacant = current_vacant + current_delta;

		// No deletion of bound synaptic elements required, connected_cnts stays the same
		if (new_vacant >= 0.0) {
			const double new_count = (1 - vacant_retract_ratio) * new_vacant + current_connected_count;
			RelearnException::check(new_count >= current_connected_count);

			cnts[neuron_id] = new_count;
			delta_cnts[neuron_id] = 0.0;
			return 0;
		}

		/**
		 * More bound elements should be deleted than are available.
		 * Now, neither vacant (see if branch above) nor bound elements are left.
		 */
		if (current_count + current_delta < 0.0) {
			connected_cnts[neuron_id] = 0.0;
			cnts[neuron_id] = 0.0;
			delta_cnts[neuron_id] = 0.0;

			unsigned int num_delete_connected = static_cast<unsigned int>(current_connected_count);
			return num_delete_connected;
		}

		const double new_cnts = current_count + current_delta;
		const double new_connected_cnt = floor(new_cnts);
		const auto num_vacant = new_cnts - new_connected_cnt;

		RelearnException::check(num_vacant >= 0);

		connected_cnts[neuron_id] = new_connected_cnt;
		cnts[neuron_id] = new_cnts;
		delta_cnts[neuron_id] = 0.0;

		const double deleted_cnts = current_connected_count - new_connected_cnt;

		RelearnException::check(deleted_cnts >= 0.0);
		unsigned int num_delete_connected = static_cast<unsigned int>(deleted_cnts);

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
