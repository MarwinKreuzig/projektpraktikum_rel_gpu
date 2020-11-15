/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MPIInfos.h"

#include "LogMessages.h"
#include "MPI_RMA_MemAllocator.h"
#include "Utility.h"

#include <mpi.h>

#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace MPIInfos {
	/**
	 * IMPORTANT: MPI expects int array with receive counts and displacements for vector operations
	 *
	 * The int receive count limits the number of neurons per rank to ~2e9 (2^31 - 1).
	 * The int displacement limits the *total* number of neurons to ~2e9 (2^31 - 1) which is a problem.
	 * A way to solve this problem is to use communication operations without displacement.
	 *
	 * Solving the problem is future work. Until it is solved the total number of neurons is limited to 2^31-1.
	 */


	 /**
	  * Global variables
	  */
	int num_ranks;                     // Number of ranks in MPI_COMM_WORLD
	int my_rank;                       // My rank in MPI_COMM_WORLD

	size_t num_neurons;                // Total number of neurons
	int    my_num_neurons;             // My number of neurons I'm responsible for
	int    my_neuron_id_start;         // ID of my first neuron
	int    my_neuron_id_end;           // ID of my last neuron

	std::vector<int> num_neurons_of_ranks;         // Number of neurons that each rank is responsible for
	std::vector<int> num_neurons_of_ranks_displs;  // Displacements based on "num_neurons_of_ranks" (exclusive prefix sums, i.e. Exscan)

	int thread_level_provided;         // Thread level provided by MPI

	std::string my_rank_str;

	MPI_RMA_MemAllocator<OctreeNode> mpi_rma_mem_allocator;

	RMABufferOctreeNodes rma_buffer_branch_nodes;

	/**
	 * Functions
	 */
	void init(int argc, char** argv) {
		MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &MPIInfos::thread_level_provided);

		// NOLINTNEXTLINE
		MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
		// NOLINTNEXTLINE
		MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

		num_neurons_of_ranks.resize(num_ranks);
		num_neurons_of_ranks_displs.resize(num_ranks);

		const int num_digits = Util::num_digits(MPIInfos::num_ranks - 1);

		std::stringstream sstring;
		sstring << std::setw(num_digits) << std::setfill('0') << MPIInfos::my_rank;

		my_rank_str = sstring.str();
	}

	void init_neurons(size_t num_neurons) {
		/**
		 * Sanity check for use of MPI
		 *
		 * Check if num_neurons fits in int value (see IMPORTANT notice above)
		 */
		if (num_neurons > std::numeric_limits<int>::max()) {
			LogMessages::print_error("init_neurons: num_neurons does not fit in \"int\" data type");
			exit(EXIT_FAILURE);
		}

		MPIInfos::num_neurons = num_neurons;

		/*
		 * Info about how much to receive (num_neurons) from every process and where to store it (displs)
		 */
		int displ = 0;
		const int rest = static_cast<int>(num_neurons) % num_ranks;
		const int block_size = static_cast<int>(num_neurons) / num_ranks;
		for (int i = 0; i < num_ranks; i++) {
			num_neurons_of_ranks[i] = block_size;
			num_neurons_of_ranks[i] += (i < rest) ? 1 : 0;

			num_neurons_of_ranks_displs[i] = displ;
			displ += num_neurons_of_ranks[i];
		}

		/*
		 * Calc which neurons this MPI process is responsible for
		 */
		my_num_neurons = num_neurons_of_ranks[my_rank];
		my_neuron_id_start = 0;
		for (int i = 0; i < my_rank; i++) {
			my_neuron_id_start += num_neurons_of_ranks[i];
		}
		my_neuron_id_end = my_neuron_id_start + (my_num_neurons - 1);
	}

	void init_mem_allocator(size_t mem_size) {
		mpi_rma_mem_allocator.set_size_requested(mem_size);
		mpi_rma_mem_allocator.allocate_rma_mem();
		mpi_rma_mem_allocator.create_rma_window();  // collective
		mpi_rma_mem_allocator.gather_rma_window_base_pointers();
	}

	void init_buffer_octree(size_t num_partitions) {
		rma_buffer_branch_nodes.num_nodes = num_partitions;
		rma_buffer_branch_nodes.ptr = mpi_rma_mem_allocator.get_block_of_objects_memory(num_partitions);

		mpi_rma_mem_allocator.init_free_object_list();
	}

	void lock_window(int rank, MPI_Locktype lock_type) {
		int lock_type_int = static_cast<int>(lock_type);
		MPI_Win_lock(lock_type_int, rank, MPI_MODE_NOCHECK, mpi_rma_mem_allocator.mpi_window);
	}

	void unlock_window(int rank) {
		MPI_Win_unlock(rank, mpi_rma_mem_allocator.mpi_window);
	}

	void finalize() /*noexcept*/ {
		// Free RMA window (MPI collective)
		// mpi_rma_mem_allocator.free_rma_window();
		// mpi_rma_mem_allocator.deallocate_rma_mem();

		MPI_Finalize();
	}

	// Print which neurons "rank" is responsible for
	void print_infos_rank(int rank) {
		if (rank == my_rank || rank == -1) {
			std::cout << "Number ranks: " << num_ranks << "\n";
			std::cout << "Partitioning based on number neurons would be: Rank " << my_rank << ": my_num_neurons: " << my_num_neurons
				<< " [start_id,end_id]: [" << my_neuron_id_start << "," << my_neuron_id_end << "]\n";
		}
	}
} // namespace MPIInfos
