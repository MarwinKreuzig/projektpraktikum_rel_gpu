/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronModels.h"

#include "MPIInfos.h"
#include "Random.h"

#include <mpi.h>

NeuronModels::NeuronModels(size_t num_neurons, double k, double tau_C, double beta, int h)
	: my_num_neurons(num_neurons),
	k(k),
	tau_C(tau_C),
	beta(beta),
	h(h),
	x(num_neurons, 0),
	fired(num_neurons, 0),
	I_syn(num_neurons, 0) {
}

/* Performs one iteration step of update in electrical activity */
void NeuronModels::update_electrical_activity(const NetworkGraph& network_graph, std::vector<double>& C) {
	MapFiringNeuronIds map_firing_neuron_ids_outgoing;
	/**
	* Check which of my neurons fired and determine which ranks need to know about it.
	* That is, they contain the neurons connecting the axons of my firing neurons.
	*/
	GlobalTimers::timers.start(TimerRegion::PREPARE_SENDING_SPIKES);
	// For my neurons
	for (auto neuron_id = 0; neuron_id < my_num_neurons; ++neuron_id) {
		// My neuron fired
		if (static_cast<bool>(fired[neuron_id])) {
			const NetworkGraph::Edges& out_edges = network_graph.get_out_edges(neuron_id);

			// Find all target neurons which should receive the signal fired.
			// That is, neurons which connect axons from neuron "neuron_id"
			for (const auto& it_out_edge : out_edges) {
				//target_neuron_id = it_out_edge->first.second;
				auto target_rank = it_out_edge.first.first;

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
	std::vector<size_t> num_firing_neuron_ids_for_ranks(MPIInfos::num_ranks, 0);
	std::vector<size_t> num_firing_neuron_ids_from_ranks(MPIInfos::num_ranks, 112233);

	// Fill vector with my number of firing neuron ids for every rank (excluding me)
	for (const auto& map_it : map_firing_neuron_ids_outgoing) {
		auto rank = map_it.first;
		auto num_neuron_ids = map_it.second.size();

		num_firing_neuron_ids_for_ranks[rank] = num_neuron_ids;
	}
	GlobalTimers::timers.stop_and_add(TimerRegion::PREPARE_NUM_NEURON_IDS);

	GlobalTimers::timers.start(TimerRegion::ALL_TO_ALL);
	// Send and receive the number of firing neuron ids
	MPI_Alltoall(num_firing_neuron_ids_for_ranks.data(), sizeof(size_t), MPI_CHAR,
		num_firing_neuron_ids_from_ranks.data(), sizeof(size_t), MPI_CHAR,
		MPI_COMM_WORLD);
	GlobalTimers::timers.stop_and_add(TimerRegion::ALL_TO_ALL);

	GlobalTimers::timers.start(TimerRegion::ALLOC_MEM_FOR_NEURON_IDS);
	// Now I know how many neuron ids I will get from every rank.
	// Allocate memory for all incoming neuron ids.
	MapFiringNeuronIds map_firing_neuron_ids_incoming;
	for (auto rank = 0; rank < MPIInfos::num_ranks; ++rank) {
		auto num_neuron_ids = num_firing_neuron_ids_from_ranks[rank];
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
	auto mpi_requests_index = 0;

	// Receive actual neuron ids
	for (auto& map_it : map_firing_neuron_ids_incoming) {
		auto rank = map_it.first;
		auto buffer = map_it.second.get_neuron_ids();
		const auto size_in_bytes = static_cast<int>(map_it.second.get_neuron_ids_size_in_bytes());

		MPI_Irecv(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
		++mpi_requests_index;
	}

	// Send actual neuron ids
	for (const auto& map_it : map_firing_neuron_ids_outgoing) {
		auto rank = map_it.first;
		const auto buffer = map_it.second.get_neuron_ids();
		const auto size_in_bytes = static_cast<int>(map_it.second.get_neuron_ids_size_in_bytes());

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
	for (auto neuron_id = 0; neuron_id < my_num_neurons; ++neuron_id) {
		I_syn[neuron_id] = 0.0;

		/**
		 * Determine synaptic input from neurons connected to me
		 */
		 // Walk through in-edges of my neuron
		const NetworkGraph::Edges& in_edges = network_graph.get_in_edges(neuron_id);

		for (const auto& it_in_edge : in_edges) {
			auto rank = it_in_edge.first.first;
			auto src_neuron_id = it_in_edge.first.second;

			bool spike{ false };
			if (rank == MPIInfos::my_rank) {
				spike = static_cast<bool>(fired[src_neuron_id]);
			}
			else {
				MapFiringNeuronIds::const_iterator it = map_firing_neuron_ids_incoming.find(rank);
				spike = (it != map_firing_neuron_ids_incoming.end()) && (it->second.find(src_neuron_id));
			}
			I_syn[neuron_id] += k * (it_in_edge.second) * static_cast<double>(spike);
		}
	}
	GlobalTimers::timers.stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);

	GlobalTimers::timers.start(TimerRegion::CALC_ACTIVITY);

	// For my neurons
	for (size_t i = 0; i < my_num_neurons; ++i) {
		update_activity(i);

		for (int integration_steps = 0; integration_steps < h; ++integration_steps) {
			// Update calcium depending on the firing
			C[i] += (1 / static_cast<double>(h)) * (-C[i] / tau_C + beta * static_cast<double>(fired[i]));
		}
	}

	GlobalTimers::timers.stop_and_add(TimerRegion::CALC_ACTIVITY);
}
