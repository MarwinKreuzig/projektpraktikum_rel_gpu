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
#include <mpi.h>
#include "NetworkGraph.h"
#include "MPIInfos.h"
#include "LogMessages.h"
#include "Timers.h"

class NeuronModels {
public:
	/**
	 * Type for firing neuron ids which are used with MPI
	 */
	class FiringNeuronIds {
	public:
		// Return size
		size_t size() { return neuron_ids.size(); }

		// Resize the number of neuron ids
		void resize(size_t size) { neuron_ids.resize(size); }

		// Append neuron id
		//
		// NOTE: This function asks the user to guarantee
		// that elements are appended in increasing/decreasing order.
		// That is they must be sorted. Otherwise, behavior is undefined.
		void append_if_not_found_sorted(size_t neuron_id) {
			// Neuron id not included yet
			if (!std::binary_search(neuron_ids.begin(), neuron_ids.end(), neuron_id)) {
				neuron_ids.push_back(neuron_id);
			}
		}

		// Test if "neuron_id" exists
		bool find(size_t neuron_id) const {
			return std::binary_search(neuron_ids.begin(), neuron_ids.end(), neuron_id);
		}

		// Get neuron id at index "neuron_id_index"
		size_t get_neuron_id(size_t neuron_id_index) { return neuron_ids[neuron_id_index]; }

		// Get pointer to data
		size_t* get_neuron_ids() { return neuron_ids.data(); }

		size_t  get_neuron_ids_size_in_bytes() { return neuron_ids.size() * sizeof(size_t); }

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

	// No copy constructor and no assignment operator are available
	NeuronModels(const NeuronModels&) = delete;
	NeuronModels& operator=(const NeuronModels&) = delete;

public:
	~NeuronModels();

	double get_beta() {
		return beta;
	}

	int get_fired(size_t i) {
		return fired[i];
	}

	double get_x(size_t i) {
		return this->x[i];
	}

	const double* get_x() {
		return x;
	}

	int get_refrac(size_t i) {
		return refrac[i];
	}

	/* Performs one iteration step of update in electrical activity */
	void update_electrical_activity(NetworkGraph& network_graph, double* C) {
		MapFiringNeuronIds map_firing_neuron_ids_outgoing, map_firing_neuron_ids_incoming;
		typename MapFiringNeuronIds::iterator map_it;
		std::vector<size_t> num_firing_neuron_ids_for_ranks(MPIInfos::num_ranks, 0);
		std::vector<size_t> num_firing_neuron_ids_from_ranks(MPIInfos::num_ranks, 112233);
		size_t neuron_id, num_neuron_ids, target_neuron_id;
		int mpi_requests_index, rank, target_rank, size_in_bytes;
		void* buffer;

		/**
		 * Check which of my neurons fired and determine which ranks need to know about it.
		 * That is, they contain the neurons connecting the axons of my firing neurons.
		 */
		GlobalTimers::timers.start(TimerRegion::PREPARE_SENDING_SPIKES);
		// For my neurons
		for (neuron_id = 0; neuron_id < my_num_neurons; ++neuron_id) {
			// My neuron fired
			if (fired[neuron_id]) {
				const NetworkGraph::Edges& out_edges = network_graph.get_out_edges(neuron_id);
				NetworkGraph::Edges::const_iterator it_out_edge;

				// Find all target neurons which should receive the signal fired.
				// That is, neurons which connect axons from neuron "neuron_id"
				for (it_out_edge = out_edges.begin(); it_out_edge != out_edges.end(); ++it_out_edge) {
					target_neuron_id = it_out_edge->first.second;
					target_rank = it_out_edge->first.first;

					// Don't send firing neuron id to myself as I already have this info
					if (target_rank != MPIInfos::my_rank) {
						// Function expects to insert neuron ids in sorted order
						// Append if it is not already in
						map_firing_neuron_ids_outgoing[target_rank].
							append_if_not_found_sorted(neuron_id);
					}
				}
			} // My neuron fired
		} // For my neurons
		GlobalTimers::timers.stop_and_add(TimerRegion::PREPARE_SENDING_SPIKES);

		GlobalTimers::timers.start(TimerRegion::PREPARE_NUM_NEURON_IDS);
		/**
		 * Send to every rank the number of firing neuron ids it should prepare for from me.
		 * Likewise, receive the number of firing neuron ids that I should prepare for from every rank.
		 */
		 // Fill vector with my number of firing neuron ids for every rank (excluding me)
		for (map_it = map_firing_neuron_ids_outgoing.begin(); map_it != map_firing_neuron_ids_outgoing.end(); ++map_it) {
			rank = map_it->first;
			num_neuron_ids = (map_it->second).size();

			num_firing_neuron_ids_for_ranks[rank] = num_neuron_ids;
		}
		GlobalTimers::timers.stop_and_add(TimerRegion::PREPARE_NUM_NEURON_IDS);

		GlobalTimers::timers.start(TimerRegion::ALL_TO_ALL);
		// Send and receive the number of firing neuron ids
		MPI_Alltoall((char*)num_firing_neuron_ids_for_ranks.data(), sizeof(size_t), MPI_CHAR,
			(char*)num_firing_neuron_ids_from_ranks.data(), sizeof(size_t), MPI_CHAR,
			MPI_COMM_WORLD);
		GlobalTimers::timers.stop_and_add(TimerRegion::ALL_TO_ALL);

		GlobalTimers::timers.start(TimerRegion::ALLOC_MEM_FOR_NEURON_IDS);
		// Now I know how many neuron ids I will get from every rank.
		// Allocate memory for all incoming neuron ids.
		for (rank = 0; rank < MPIInfos::num_ranks; ++rank) {
			num_neuron_ids = num_firing_neuron_ids_from_ranks[rank];
			if (0 != num_neuron_ids) { // Only create key-value pair in map for "rank" if necessary
				map_firing_neuron_ids_incoming[rank].resize(num_neuron_ids);
			}
		}
		GlobalTimers::timers.stop_and_add(TimerRegion::ALLOC_MEM_FOR_NEURON_IDS);

		GlobalTimers::timers.start(TimerRegion::EXCHANGE_NEURON_IDS);
		std::vector<MPI_Request>
			mpi_requests(map_firing_neuron_ids_outgoing.size() + map_firing_neuron_ids_incoming.size());

		/**
		 * Send and receive actual neuron ids
		 */
		mpi_requests_index = 0;

		// Receive actual neuron ids
		for (map_it = map_firing_neuron_ids_incoming.begin(); map_it != map_firing_neuron_ids_incoming.end(); ++map_it) {
			rank = map_it->first;
			buffer = (map_it->second).get_neuron_ids();
			size_in_bytes = (int)((map_it->second).get_neuron_ids_size_in_bytes());

			MPI_Irecv(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
			++mpi_requests_index;
		}
		// Send actual neuron ids
		for (map_it = map_firing_neuron_ids_outgoing.begin(); map_it != map_firing_neuron_ids_outgoing.end(); ++map_it) {
			rank = map_it->first;
			buffer = (map_it->second).get_neuron_ids();
			size_in_bytes = (int)((map_it->second).get_neuron_ids_size_in_bytes());

			MPI_Isend(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
			++mpi_requests_index;
		}
		// Wait for all sends and receives to complete
		MPI_Waitall(mpi_requests_index, mpi_requests.data(), MPI_STATUSES_IGNORE);
		GlobalTimers::timers.stop_and_add(TimerRegion::EXCHANGE_NEURON_IDS);

		/**
		 * Now the fired[] array contains spikes only from my own neurons
		 * (spikes from local neurons)
		 *
		 * The incoming spikes of neurons from other ranks are in map_firing_neuron_ids_incoming
		 * (spikes from neurons from other ranks)
		 */
		GlobalTimers::timers.start(TimerRegion::CALC_SYNAPTIC_INPUT);
		// For my neurons
		int spike;
		size_t src_neuron_id;
		for (neuron_id = 0; neuron_id < my_num_neurons; ++neuron_id) {
			I_syn[neuron_id] = 0.0;

			/**
			 * Determine synaptic input from neurons connected to me
			 */
			 // Walk through in-edges of my neuron
			const NetworkGraph::Edges& in_edges = network_graph.get_in_edges(neuron_id);
			NetworkGraph::Edges::const_iterator it_in_edge;

			for (it_in_edge = in_edges.begin(); it_in_edge != in_edges.end(); ++it_in_edge) {
				rank = it_in_edge->first.first;
				src_neuron_id = it_in_edge->first.second;

				if (rank == MPIInfos::my_rank) {
					spike = fired[src_neuron_id];
				}
				else {
					MapFiringNeuronIds::const_iterator it = map_firing_neuron_ids_incoming.find(rank);
					spike = (it != map_firing_neuron_ids_incoming.end()) && (it->second.find(src_neuron_id));
				}
				I_syn[neuron_id] += k * (it_in_edge->second) * (double)spike;
			}
		}
		GlobalTimers::timers.stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);

		GlobalTimers::timers.start(TimerRegion::CALC_ACTIVITY);
		// For my neurons
		for (size_t i = 0; i < my_num_neurons; i++) {
			for (int integration_steps = 0; integration_steps < h; integration_steps++) {
				// Update the membrane potential
				x[i] += (1 / (double)h) * ((x_0 - x[i]) / tau_x + I_syn[i]);
			}

			// Neuron ready to fire again
			if (refrac[i] == 0) {
				fired[i] = Theta(x[i]);             // Decide whether a neuron fires depending on its firing rate
				refrac[i] = static_cast<int>(fired[i] * refrac_time);  // After having fired, a neuron is in a refractory state
			}
			// Neuron now/still in refractory state
			else {
				fired[i] = 0;  // Set neuron inactive
				--refrac[i];   // Decrease refractory time
			}

			for (int integration_steps = 0; integration_steps < h; integration_steps++) {
				// Update calcium depending on the firing
				C[i] += (1 / (double)h) * (-C[i] / tau_C + beta * fired[i]);
			}
		}

		GlobalTimers::timers.stop_and_add(TimerRegion::CALC_ACTIVITY);
	}

private:
	int Theta(double x) {
		// 1: fire, 0: inactive
		return (x >= random_number_distribution(random_number_generator));
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
	double* x;      // Firing rate in Hz
	int* fired;  // 1: neuron has fired, 0: neuron is inactive
	int* refrac; // Remaining refractory time, prevents neurons from firing
					// 0: neuron can fire, >0: neuron cannot fire
	double* I_syn;  // Synaptic input

	// Randpm number generator for this class (C++11)
	std::mt19937& random_number_generator;
	// Random number distribution used together with "random_number_generator" (C++11)
	// Uniform distribution for interval [0, 1]
	std::uniform_real_distribution<double> random_number_distribution;
};

#endif	/* NEURONMODELS_H */
