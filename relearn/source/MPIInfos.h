/*
 * File:   MPIInfos.h
 * Author: rinke
 *
 * Created on Apr 17, 2016
 */

#ifndef MPIINFOS_H
#define MPIINFOS_H

#include <string>

namespace MPIInfos {
	/**
	 * Global variables
	 */
	extern int num_ranks;                     // Number of ranks in MPI_COMM_WORLD
	extern int my_rank;                       // My rank in MPI_COMM_WORLD

	extern size_t num_neurons;                // Total number of neurons
	extern int    my_num_neurons;             // My number of neurons I'm responsible for
	extern int    my_neuron_id_start;         // ID of my first neuron
	extern int    my_neuron_id_end;           // ID of my last neuron

	// Needed for Allgatherv
	extern int* num_neurons_of_ranks;         // Number of neurons that each rank is responsible for
	extern int* num_neurons_of_ranks_displs;  // Displacements based on "num_neurons_of_ranks" (exclusive prefix sums, i.e. Exscan)

	extern int thread_level_provided;         // Thread level provided by MPI

	extern std::string my_rank_str;

	/**
	 * Functions
	 */
	void init(int argc, char** argv);
	void init_neurons(const size_t num_neurons);
	void finalize();
	void print_infos_rank(int rank);

	//
	// TODO: Not needed anymore
	//
		// Return the rank holding neuron "neuron_id"
	inline int neuron_id_to_rank(size_t neuron_id) {
		int rest = (int)num_neurons % num_ranks;
		int block_size = (int)num_neurons / num_ranks;
		int block_size_with_rest = block_size + 1;
		int max_neuron_id_with_rest = rest * block_size_with_rest - 1;
		int rank;

		if ((int)neuron_id <= max_neuron_id_with_rest) {
			rank = (int)(neuron_id / (size_t)block_size_with_rest);
		}
		else {
			neuron_id -= (size_t)rest * block_size_with_rest;
			/*
			 * "rest" is first rank after all the ranks where each of them has "block_size_with_rest" neurons
			 */
			rank = rest + (int)(neuron_id / (size_t)block_size);
		}

		return rank;
	}

	//
	// TODO: Not needed anymore
	//
		// Return true if "neuron_id" is one of my neurons
	inline bool neuron_id_is_mine(size_t neuron_id) {
		return (neuron_id >= MPIInfos::my_neuron_id_start && neuron_id <= MPIInfos::my_neuron_id_end);
	}
}

#endif /* MPIINFOS_H */
