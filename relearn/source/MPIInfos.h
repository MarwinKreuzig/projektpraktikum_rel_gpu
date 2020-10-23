/*
 * File:   MPIInfos.h
 * Author: rinke
 *
 * Created on Apr 17, 2016
 */

#ifndef MPIINFOS_H
#define MPIINFOS_H

#include <string>

#include "OctreeNode.h"
#include "MPI_RMA_MemAllocator.h"

class Octree;

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

	extern MPI_RMA_MemAllocator<OctreeNode> mpi_rma_mem_allocator;

	/**
	 * Functions
	 */
	void init(int argc, char** argv);
	void init_neurons(const size_t num_neurons);
	void finalize() noexcept;
	void print_infos_rank(int rank);
}

#endif /* MPIINFOS_H */
