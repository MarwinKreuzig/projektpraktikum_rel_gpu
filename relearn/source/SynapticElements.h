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

#include "MPIWrapper.h"
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

	const std::vector<double>& get_cnts() const noexcept {
		return cnts;
	}

	const std::vector<double>& get_connected_cnts() const noexcept {
		return connected_cnts; 
	}
	
	const std::vector<double>& get_delta_cnts() const noexcept {
		return delta_cnts;
	}
	
	const std::vector<SignalType>& get_signal_types() const noexcept {
		return signal_types; 
	}

	void update_cnt(size_t neuron_id, double delta) {
		cnts[neuron_id] += delta;
		RelearnException::check(cnts[neuron_id] >= 0.0);
	}

	void update_conn_cnt(size_t neuron_id, double delta, const char* mess) {
		connected_cnts[neuron_id] += delta;
		RelearnException::check(connected_cnts[neuron_id] >= 0.0, mess);
	}

	void update_delta_cnt(size_t neuron_id, double delta) {
		delta_cnts[neuron_id] += delta;
		RelearnException::check(delta_cnts[neuron_id] >= 0.0);
	}

	void set_signal_type(size_t neuron_id, SignalType type) noexcept {
		signal_types[neuron_id] = type;
	}

	double get_cnt(size_t neuron_id) const noexcept {
		return cnts[neuron_id]; 
	}
	
	double get_connected_cnt(size_t neuron_id) const noexcept { 
		return connected_cnts[neuron_id];
	}
	
	double get_delta_cnt(size_t neuron_id) const noexcept { 
		return delta_cnts[neuron_id];
	}

	SignalType get_signal_type(size_t neuron_id) const noexcept { 
		return signal_types[neuron_id]; 
	}

	ElementType get_element_type() const noexcept { 
		return type; 
	}

	/**
	 * Updates the number of synaptic elements for neuron "neuron_id"
	 * Returns the number of synapses to be deleted as a consequence of deleting synaptic elements
	 *
	 * Synaptic elements are deleted based on "delta_cnts" in the following way:
	 * 1. Delete vacant elements
	 * 2. Delete bound elements
	 */
	unsigned int update_number_elements(size_t neuron_id);

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
