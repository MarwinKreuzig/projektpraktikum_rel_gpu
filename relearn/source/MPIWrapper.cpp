/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MPIWrapper.h"

#include "LogMessages.h"
#include "MPI_RMA_MemAllocator.h"
#include "RelearnException.h"
#include "Utility.h"

#include <bitset>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

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
MPI_Op MPIWrapper::minsummax;

int MPIWrapper::num_ranks = -1; // Number of ranks in MPI_COMM_WORLD
int MPIWrapper::my_rank = -1; // My rank in MPI_COMM_WORLD

size_t MPIWrapper::num_neurons; // Total number of neurons
size_t MPIWrapper::my_num_neurons; // My number of neurons I'm responsible for
size_t MPIWrapper::my_neuron_id_start; // ID of my first neuron
size_t MPIWrapper::my_neuron_id_end; // ID of my last neuron

std::vector<size_t> MPIWrapper::num_neurons_of_ranks; // Number of neurons that each rank is responsible for
std::vector<size_t> MPIWrapper::num_neurons_of_ranks_displs; // Displacements based on "num_neurons_of_ranks" (exclusive prefix sums, i.e. Exscan)

int MPIWrapper::thread_level_provided; // Thread level provided by MPI

std::string MPIWrapper::my_rank_str;

MPI_RMA_MemAllocator MPIWrapper::mpi_rma_mem_allocator;

MPIWrapper::RMABufferOctreeNodes MPIWrapper::rma_buffer_branch_nodes;

/**
 * Functions
 */
void MPIWrapper::init(int argc, char** argv) {
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &thread_level_provided);

    init_globals();

    // Number of ranks must be 2^n so that
    // the connectivity update works correctly
    const std::bitset<sizeof(int) * 8> bitset_num_ranks(num_ranks);
    if (1 != bitset_num_ranks.count() && (0 == my_rank)) {
        RelearnException::fail("Number of ranks must be of the form 2^n");
    }

    register_custom_function();

    num_neurons_of_ranks.resize(num_ranks);
    num_neurons_of_ranks_displs.resize(num_ranks);

    const unsigned int num_digits = Util::num_digits(num_ranks - 1);

    std::stringstream sstring;
    sstring << std::setw(static_cast<std::streamsize>(num_digits)) << std::setfill('0') << my_rank;

    my_rank_str = sstring.str();
}

void MPIWrapper::init_globals() {
    // NOLINTNEXTLINE
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // NOLINTNEXTLINE
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
}

void MPIWrapper::init_neurons(size_t num_neurons) {
    /**
	 * Sanity check for use of MPI
	 *
	 * Check if num_neurons fits in int value (see IMPORTANT notice above)
	 */
    if (num_neurons > std::numeric_limits<int>::max()) {
        LogMessages::print_error("init_neurons: num_neurons does not fit in \"int\" data type");

        // NOLINTNEXTLINE
        exit(EXIT_FAILURE);
    }

    MPIWrapper::num_neurons = num_neurons;

    /*
	 * Info about how much to receive (num_neurons) from every process and where to store it (displs)
	 */
    size_t displ = 0;
    const size_t rest = num_neurons % num_ranks;
    const size_t block_size = num_neurons / num_ranks;
    for (size_t i = 0; i < num_ranks; i++) {
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

void MPIWrapper::init_buffer_octree(size_t num_partitions) {
    mpi_rma_mem_allocator.init(Constants::mpi_alloc_mem);

    rma_buffer_branch_nodes.num_nodes = num_partitions;
    rma_buffer_branch_nodes.ptr = mpi_rma_mem_allocator.get_root_nodes_for_local_trees(num_partitions);
}

void MPIWrapper::barrier(Scope scope) {
    MPI_Comm mpi_scope = translate_scope(scope);

    const int errorcode = MPI_Barrier(mpi_scope);
    RelearnException::check(errorcode == 0, "Error in barrier");
}

[[nodiscard]] double MPIWrapper::reduce(double value, ReduceFunction function, int root_rank, Scope scope) {
    MPI_Comm mpi_scope = translate_scope(scope);
    MPI_Op mpi_reduce_function = translate_reduce_function(function);

    double result = 0.0;
    // NOLINTNEXTLINE
    const int errorcode = MPI_Reduce(&value, &result, 1, MPI_DOUBLE, mpi_reduce_function, root_rank, mpi_scope);
    RelearnException::check(errorcode == 0, "Error in reduce");

    return result;
}

[[nodiscard]] double MPIWrapper::all_reduce(double value, ReduceFunction function, Scope scope) {
    MPI_Comm mpi_scope = translate_scope(scope);
    MPI_Op mpi_reduce_function = translate_reduce_function(function);

    double result = 0.0;
    // NOLINTNEXTLINE
    const int errorcode = MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, mpi_reduce_function, mpi_scope);
    RelearnException::check(errorcode == 0, "Error in all reduce");

    return result;
}

void MPIWrapper::all_to_all(const std::vector<size_t>& src, std::vector<size_t>& dst, Scope scope) {
    size_t count_src = src.size();
    size_t count_dst = dst.size();

    RelearnException::check(count_src == count_dst, "Error in all to all: size");

    MPI_Comm mpi_scope = translate_scope(scope);

    // NOLINTNEXTLINE
    const int errorcode = MPI_Alltoall(src.data(), sizeof(size_t), MPI_CHAR, dst.data(), sizeof(size_t), MPI_CHAR, mpi_scope);
    RelearnException::check(errorcode == 0, "Error in all to all, mpi");
}

[[nodiscard]] MPI_Aint MPIWrapper::get_ptr_displacement(int target_rank, const OctreeNode* ptr) {
    const std::vector<MPI_Aint>& base_ptrs = mpi_rma_mem_allocator.get_base_pointers();
    const auto displacement = MPI_Aint(ptr) - MPI_Aint(base_ptrs[target_rank]);
    return displacement;
}

[[nodiscard]] OctreeNode* MPIWrapper::new_octree_node() {
    return mpi_rma_mem_allocator.new_octree_node();
}

[[nodiscard]] int MPIWrapper::get_num_ranks() {
    RelearnException::check(num_ranks >= 0, "MPIWrapper is not initialized");
    return num_ranks;
}

[[nodiscard]] int MPIWrapper::get_my_rank() {
    RelearnException::check(my_rank >= 0, "MPIWrapper is not initialized");
    return my_rank;
}

[[nodiscard]] size_t MPIWrapper::get_num_neurons() {
    return num_neurons;
}

[[nodiscard]] size_t MPIWrapper::get_my_num_neurons() {
    return my_num_neurons;
}

[[nodiscard]] size_t MPIWrapper::get_my_neuron_id_start() {
    return my_neuron_id_start;
}

[[nodiscard]] size_t MPIWrapper::get_my_neuron_id_end() {
    return my_neuron_id_end;
}

[[nodiscard]] size_t MPIWrapper::get_num_avail_objects() {
    return mpi_rma_mem_allocator.get_min_num_avail_objects();
}

[[nodiscard]] OctreeNode* MPIWrapper::get_buffer_octree_nodes() {
    return rma_buffer_branch_nodes.ptr;
}

[[nodiscard]] size_t MPIWrapper::get_num_buffer_octree_nodes() {
    return rma_buffer_branch_nodes.num_nodes;
}

[[nodiscard]] std::string MPIWrapper::get_my_rank_str() {
    return my_rank_str;
}

void MPIWrapper::delete_octree_node(OctreeNode* ptr) {
    mpi_rma_mem_allocator.delete_octree_node(ptr);
}

void MPIWrapper::wait_request(AsyncToken& request) {
    // NOLINTNEXTLINE
    if (MPI_REQUEST_NULL != request) {
        // NOLINTNEXTLINE
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
}

[[nodiscard]] MPIWrapper::AsyncToken MPIWrapper::get_non_null_request() {
    // NOLINTNEXTLINE
    return (AsyncToken)(!MPI_REQUEST_NULL);
}

[[nodiscard]] MPIWrapper::AsyncToken MPIWrapper::get_null_request() {
    // NOLINTNEXTLINE
    return (AsyncToken)(MPI_REQUEST_NULL);
}

void MPIWrapper::all_gather_v(size_t total_num_neurons, std::vector<double>& xyz_pos, std::vector<int>& recvcounts, std::vector<int>& displs) {
    // Create MPI data type for three doubles
    // NOLINTNEXTLINE
    MPI_Datatype type;

    // NOLINTNEXTLINE
    const int errorcode_1 = MPI_Type_contiguous(3, MPI_DOUBLE, &type);
    RelearnException::check(errorcode_1 == 0, "Error in all to all, mpi");

    const int errorcode_2 = MPI_Type_commit(&type);
    RelearnException::check(errorcode_2 == 0, "Error in all to all, mpi");

    barrier(Scope::global);

    // Receive all neuron positions as xyz-triples

    // NOLINTNEXTLINE
    const int errorcode_3 = MPI_Allgatherv(MPI_IN_PLACE, static_cast<int>(total_num_neurons), type, xyz_pos.data(), recvcounts.data(), displs.data(), type, MPI_COMM_WORLD);
    RelearnException::check(errorcode_3 == 0, "Error in all to all, mpi");

    const int errorcode_4 = MPI_Type_free(&type);
    RelearnException::check(errorcode_4 == 0, "Error in all to all, mpi");
}

void MPIWrapper::wait_all_tokens(std::vector<AsyncToken>& tokens) {
    const int size = static_cast<int>(tokens.size());
    // NOLINTNEXTLINE
    MPI_Waitall(size, tokens.data(), MPI_STATUSES_IGNORE);
}

[[nodiscard]] MPI_Op MPIWrapper::translate_reduce_function(ReduceFunction rf) {
    // NOLINTNEXTLINE
    auto mpi_reduce_function = MPI_Op(0);

    switch (rf) {
    case ReduceFunction::min:
        // NOLINTNEXTLINE
        mpi_reduce_function = MPI_MIN;
        break;
    case ReduceFunction::max:
        // NOLINTNEXTLINE
        mpi_reduce_function = MPI_MAX;
        break;
    case ReduceFunction::sum:
        // NOLINTNEXTLINE
        mpi_reduce_function = MPI_SUM;
        break;
    case ReduceFunction::minsummax:
        mpi_reduce_function = minsummax;
        break;
    default:
        RelearnException::fail("In reduce, got wrong function");
        break;
    }

    return mpi_reduce_function;
}

[[nodiscard]] MPI_Comm MPIWrapper::translate_scope(Scope scope) {
    // NOLINTNEXTLINE
    auto mpi_scope = MPI_Comm(0);

    switch (scope) {
    case Scope::global:
        // NOLINTNEXTLINE
        mpi_scope = MPI_COMM_WORLD;
        break;
    default:
        RelearnException::fail("In barrier, got wrong scope");
        break;
    }

    return mpi_scope;
}

void MPIWrapper::register_custom_function() {
    // NOLINTNEXTLINE
    MPI_Op_create((MPI_User_function*)MPIUserDefinedOperation::min_sum_max, 1, &minsummax);
}

void MPIWrapper::free_custom_function() {
    MPI_Op_free(&minsummax);
}

void MPIWrapper::lock_window(int rank, MPI_Locktype lock_type) {
    RelearnException::check(rank >= 0);
    const auto lock_type_int = static_cast<int>(lock_type);

    // NOLINTNEXTLINE
    const int errorcode = MPI_Win_lock(lock_type_int, rank, MPI_MODE_NOCHECK, mpi_rma_mem_allocator.mpi_window);
    RelearnException::check(errorcode == 0, "Error in lock window");
}

void MPIWrapper::unlock_window(int rank) {
    RelearnException::check(rank >= 0);
    const int errorcode = MPI_Win_unlock(rank, mpi_rma_mem_allocator.mpi_window);
    RelearnException::check(errorcode == 0, "Error in unlock window");
}

void MPIWrapper::finalize() /*noexcept*/ {
    free_custom_function();

    // Free RMA window (MPI collective)
    mpi_rma_mem_allocator.free_rma_window();
    mpi_rma_mem_allocator.deallocate_rma_mem();

    const int errorcode = MPI_Finalize();
    RelearnException::check(errorcode == 0, "Error in finalize");
}

// Print which neurons "rank" is responsible for
void MPIWrapper::print_infos_rank(int rank) {
    if (rank == my_rank || rank == -1) {
        std::cout << "Number ranks: " << num_ranks << "\n";
        std::cout << "Partitioning based on number neurons would be: Rank " << my_rank << ": my_num_neurons: " << my_num_neurons
                  << " [start_id,end_id]: [" << my_neuron_id_start << "," << my_neuron_id_end << "]\n";
    }
}

// This combination function assumes that it's called with the correct MPI datatype
void MPIUserDefinedOperation::min_sum_max(const int* invec, int* inoutvec, const int* const len, [[maybe_unused]] MPI_Datatype* dtype) /*noexcept*/ {
    const auto real_length = *len / sizeof(double) / 3;

    // NOLINTNEXTLINE
    const auto in = reinterpret_cast<const double*>(invec);
    // NOLINTNEXTLINE
    auto inout = reinterpret_cast<double*>(inoutvec);

    for (int i = 0; i < real_length; i++) {
        // NOLINTNEXTLINE
        inout[3 * i] = std::min(in[3 * i], inout[3 * i]);

        // NOLINTNEXTLINE
        inout[3 * i + 1] += in[3 * i + 1];

        // NOLINTNEXTLINE
        inout[3 * i + 2] = std::max(in[3 * i + 2], inout[3 * i + 2]);
    }
}
