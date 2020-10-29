/*
 * File:   NeuronModels.h
 * Author: naveau
 *
 * Created on September 26, 2014, 9:31 PM
 */

#ifndef NEURONMODELS_H
#define	NEURONMODELS_H

#include <cstddef>
#include <random>
#include <algorithm>
#include <vector>
#include "NetworkGraph.h"
#include "MPIInfos.h"
#include "LogMessages.h"
#include "Timers.h"

class NeuronMonitor;

class NeuronModels {
	friend class NeuronMonitor;

public:
	/**
	 * Type for firing neuron ids which are used with MPI
	 */
	class FiringNeuronIds {
	public:
		// Return size
		size_t size() const noexcept { return neuron_ids.size(); }

		// Resize the number of neuron ids
		void resize(size_t size) { neuron_ids.resize(size); }

		// Append neuron id
		//
		// NOTE: This function asks the user to guarantee
		// that elements are appended in increasing/decreasing order.
		// That is they must be sorted. Otherwise, behavior is undefined.
		void append_if_not_found_sorted(size_t neuron_id) {
			// Neuron id not included yet
			const bool found = find(neuron_id);
			if (!found) {
				neuron_ids.push_back(neuron_id);
			}
		}

		// Test if "neuron_id" exists
		bool find(size_t neuron_id) const {
			return std::binary_search(neuron_ids.begin(), neuron_ids.end(), neuron_id);
		}

		// Get neuron id at index "neuron_id_index"
		size_t get_neuron_id(size_t neuron_id_index) const noexcept { return neuron_ids[neuron_id_index]; }

		// Get pointer to data
		size_t* get_neuron_ids() noexcept { return neuron_ids.data(); }

		const size_t* get_neuron_ids() const noexcept { return neuron_ids.data(); }

		size_t get_neuron_ids_size_in_bytes() const noexcept { return neuron_ids.size() * sizeof(size_t); }

	private:
		std::vector<size_t> neuron_ids;  // Firing neuron ids
										 // This vector is used as MPI communication buffer
	};

	/**
	 * Map of (MPI rank; FiringNeuronIds)
	 * The MPI rank specifies the corresponding process
	 */
	typedef std::map<int, FiringNeuronIds> MapFiringNeuronIds;

	NeuronModels(size_t num_neurons, double x_0, double tau_x, double k, double tau_C, double beta, int h, double refrac_time);

public:
	~NeuronModels() = default;

	NeuronModels(const NeuronModels& other) = delete;
	NeuronModels& operator=(const NeuronModels& other) = delete;

	NeuronModels(NeuronModels&& other) = default;
	//	: random_number_generator(other.random_number_generator)
	//{
	//	my_num_neurons = other.my_num_neurons;
	//	x_0 = other.x_0;
	//	tau_x = other.tau_x;
	//	k = other.k;
	//	tau_C = other.tau_C;
	//	beta = other.beta;
	//	refrac_time = other.refrac_time;
	//	h = other.h;
	//	x = std::move(other.x);
	//	fired = std::move(other.fired);
	//	refrac = std::move(other.refrac);
	//	I_syn = std::move(other.I_syn);
	//	random_number_distribution = std::move(other.random_number_distribution);
	//}
	NeuronModels& operator=(NeuronModels&& other) = default;

	double get_beta() const noexcept {
		return beta;
	}

	int get_fired(size_t i) const noexcept {
		return fired[i];
	}

	double get_x(size_t i) const noexcept {
		return this->x[i];
	}

	const std::vector<double>& get_x() const noexcept {
		return x;
	}

	int get_refrac(size_t i) const noexcept {
		return refrac[i];
	}

	/* Performs one iteration step of update in electrical activity */
	void update_electrical_activity(const NetworkGraph& network_graph, std::vector<double>& C);

private:
	int Theta(double x) {
		// 1: fire, 0: inactive
		const double threshold = random_number_distribution(random_number_generator);
		return (x >= threshold);
	}

	// My local number of neurons
	size_t my_num_neurons;

	// Model parameters for all neurons
	double x_0;         // Background or resting activity
	double tau_x;       // Decay time of firing rate in msec
	double k;           // Proportionality factor for synapses in Hz
	double tau_C;       // Decay time of calcium
	double beta;        // Increase in calcium each time a neuron fires
	double refrac_time; // Length of refractory period in msec. After an action potential a neuron cannot fire for this time
	int    h;           // Precision for Euler integration

	// Variables for each neuron where the array index denotes the neuron ID
	std::vector<double> x;      // Firing rate in Hz
	std::vector<int> fired;  // 1: neuron has fired, 0: neuron is inactive
	std::vector<int> refrac; // Remaining refractory time, prevents neurons from firing, 0: neuron can fire, >0: neuron cannot fire
	std::vector<double> I_syn;  // Synaptic input

	// Randpm number generator for this class (C++11)
	std::mt19937& random_number_generator;
	// Random number distribution used together with "random_number_generator" (C++11)
	// Uniform distribution for interval [0, 1]
	std::uniform_real_distribution<double> random_number_distribution;
};

#endif	/* NEURONMODELS_H */
