/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "../mpi/MPIWrapper.h"
#include "spdlog/fmt/bundled/core.h"

#if MPI_FOUND

#include "../Config.h"
#include "../io/LogFiles.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"
#include "../util/Utility.h"
#include "MPI_RMA_MemAllocator.h"

#include <mpi.h>

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

    const unsigned int num_digits = Util::num_digits(num_ranks - 1);
    my_rank_str = fmt::format("{1:0>{0}}", num_digits, my_rank);
}

void MPIWrapper::init_globals() {
    // NOLINTNEXTLINE
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // NOLINTNEXTLINE
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
}

void MPIWrapper::init_buffer_octree(size_t num_partitions) {
    MPI_RMA_MemAllocator::init(Constants::mpi_alloc_mem, num_partitions);

    rma_buffer_branch_nodes.num_nodes = num_partitions;
    rma_buffer_branch_nodes.ptr = MPI_RMA_MemAllocator::get_branch_nodes();
}

void MPIWrapper::barrier(Scope scope) {
    const MPI_Comm mpi_scope = translate_scope(scope);

    const int errorcode = MPI_Barrier(mpi_scope);
    RelearnException::check(errorcode == 0, "Error in barrier");
}

double MPIWrapper::reduce(double value, ReduceFunction function, int root_rank, Scope scope) {
    RelearnException::check(root_rank >= 0, "In MPIWrapper::reduce, root_rank was negative");
    const MPI_Comm mpi_scope = translate_scope(scope);
    const MPI_Op mpi_reduce_function = translate_reduce_function(function);

    double result = 0.0;
    // NOLINTNEXTLINE
    const int errorcode = MPI_Reduce(&value, &result, 1, MPI_DOUBLE, mpi_reduce_function, root_rank, mpi_scope);
    RelearnException::check(errorcode == 0, "Error in reduce");

    return result;
}

double MPIWrapper::all_reduce(double value, ReduceFunction function, Scope scope) {
    const MPI_Comm mpi_scope = translate_scope(scope);
    const MPI_Op mpi_reduce_function = translate_reduce_function(function);

    double result = 0.0;
    // NOLINTNEXTLINE
    const int errorcode = MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, mpi_reduce_function, mpi_scope);
    RelearnException::check(errorcode == 0, "Error in all reduce");

    return result;
}

void MPIWrapper::all_to_all(const std::vector<size_t>& src, std::vector<size_t>& dst, Scope scope) {
    const size_t count_src = src.size();
    const size_t count_dst = dst.size();

    RelearnException::check(count_src == count_dst, "Error in all to all: size");

    const MPI_Comm mpi_scope = translate_scope(scope);

    // NOLINTNEXTLINE
    const int errorcode = MPI_Alltoall(src.data(), sizeof(size_t), MPI_CHAR, dst.data(), sizeof(size_t), MPI_CHAR, mpi_scope);
    RelearnException::check(errorcode == 0, "Error in all to all, mpi");
}

void MPIWrapper::async_s(const void* buffer, int count, int rank, Scope scope, AsyncToken& token) {
    RelearnException::check(rank >= 0, "Error in async s, rank is <= 0");
    const MPI_Comm mpi_scope = translate_scope(scope);
    // NOLINTNEXTLINE
    const int errorcode = MPI_Isend(buffer, count, MPI_CHAR, rank, 0, mpi_scope, &token);
    RelearnException::check(errorcode == 0, "Error in async send");
}

void MPIWrapper::async_recv(void* buffer, int count, int rank, Scope scope, AsyncToken& token) {
    RelearnException::check(rank >= 0, "Error in async recv, rank is <= 0");
    const MPI_Comm mpi_scope = translate_scope(scope);
    // NOLINTNEXTLINE
    const int errorcode = MPI_Irecv(buffer, count, MPI_CHAR, rank, 0, mpi_scope, &token);
    RelearnException::check(errorcode == 0, "Error in async receive");
}

void MPIWrapper::reduce(const void* src, void* dst, int size, ReduceFunction function, int root_rank, Scope scope) {
    const MPI_Comm mpi_scope = translate_scope(scope);
    const MPI_Op mpi_reduce_function = translate_reduce_function(function);

    // NOLINTNEXTLINE
    const int errorcode = MPI_Reduce(src, dst, size, MPI_CHAR, mpi_reduce_function, root_rank, mpi_scope);
    RelearnException::check(errorcode == 0, "Error in reduce: %d", errorcode);
}

void MPIWrapper::all_gather(const void* own_data, void* buffer, int size, Scope scope) {
    const MPI_Comm mpi_scope = translate_scope(scope);

    // NOLINTNEXTLINE
    const int errorcode = MPI_Allgather(own_data, size, MPI_CHAR, buffer, size, MPI_CHAR, mpi_scope);
    RelearnException::check(errorcode == 0, "Error in all gather");
}

void MPIWrapper::all_gather_inl(void* ptr, int count, Scope scope) {
    RelearnException::check(count > 0, "Error in all gather , count is not greater than 0");
    const MPI_Comm mpi_scope = translate_scope(scope);

    // NOLINTNEXTLINE
    const int errorcode = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, ptr, count, MPI_CHAR, mpi_scope);
    RelearnException::check(errorcode == 0, "Error in all gather ");
}

void MPIWrapper::download_octree_node(OctreeNode* dst, int target_rank, const OctreeNode* src) {
    RelearnException::check(target_rank >= 0, "target rank is negative in download_octree_nodet");
    const auto& base_ptrs = MPI_RMA_MemAllocator::get_base_pointers();
    RelearnException::check(target_rank < base_ptrs.size(), "target rank is greater than the base pointers");
    const auto displacement = int64_t(src) - base_ptrs[target_rank];

    const MPI_Aint displacement_mpi(displacement);
    // NOLINTNEXTLINE
    const int errorcode = MPI_Get(dst, sizeof(OctreeNode), MPI_CHAR, target_rank, displacement_mpi, sizeof(OctreeNode), MPI_CHAR, MPI_RMA_MemAllocator::mpi_window);
    RelearnException::check(errorcode == 0, "Error in get");
}

OctreeNode* MPIWrapper::new_octree_node() {
    return MPI_RMA_MemAllocator::new_octree_node();
}

int MPIWrapper::get_num_ranks() {
    RelearnException::check(num_ranks >= 0, "MPIWrapper is not initialized");
    return num_ranks;
}

int MPIWrapper::get_my_rank() {
    RelearnException::check(my_rank >= 0, "MPIWrapper is not initialized");
    return my_rank;
}

size_t MPIWrapper::get_num_avail_objects() {
    return MPI_RMA_MemAllocator::get_num_avail_objects();
}

OctreeNode* MPIWrapper::get_buffer_octree_nodes() {
    return rma_buffer_branch_nodes.ptr;
}

size_t MPIWrapper::get_num_buffer_octree_nodes() {
    return rma_buffer_branch_nodes.num_nodes;
}

std::string MPIWrapper::get_my_rank_str() {
    RelearnException::check(my_rank >= 0, "MPIWrapper is not initialized");
    return my_rank_str;
}

void MPIWrapper::delete_octree_node(OctreeNode* ptr) {
    MPI_RMA_MemAllocator::delete_octree_node(ptr);
}

void MPIWrapper::wait_request(AsyncToken& request) {
    // NOLINTNEXTLINE
    if (MPI_REQUEST_NULL != request) {
        // NOLINTNEXTLINE
        const int errorcode = MPI_Wait(&request, MPI_STATUS_IGNORE);
        RelearnException::check(errorcode == 0, "Error in wait_request ");
    }
}

void MPIWrapper::wait_all_tokens(std::vector<AsyncToken>& tokens) {
    const int size = static_cast<int>(tokens.size());
    // NOLINTNEXTLINE
    const int errorcode = MPI_Waitall(size, tokens.data(), MPI_STATUSES_IGNORE);
    RelearnException::check(errorcode == 0, "Error in wait_all_tokens");
}

MPI_Op MPIWrapper::translate_reduce_function(ReduceFunction rf) {
    switch (rf) {
    case ReduceFunction::min:
        // NOLINTNEXTLINE
        return MPI_MIN;

    case ReduceFunction::max:
        // NOLINTNEXTLINE
        return MPI_MAX;

    case ReduceFunction::sum:
        // NOLINTNEXTLINE
        return MPI_SUM;

    case ReduceFunction::minsummax:
        return minsummax;

    default:
        RelearnException::fail("In reduce, got wrong function");
        // NOLINTNEXTLINE
        return 0;
    }
}

MPI_Comm MPIWrapper::translate_scope(Scope scope) {
    switch (scope) {
    case Scope::global:
        // NOLINTNEXTLINE
        return MPI_COMM_WORLD;
    default:
        RelearnException::fail("In barrier, got wrong scope");
        // NOLINTNEXTLINE
        return 0;
    }
}

void MPIWrapper::register_custom_function() {
    // NOLINTNEXTLINE
    MPI_Op_create((MPI_User_function*)MPIUserDefinedOperation::min_sum_max, 1, &minsummax);
}

void MPIWrapper::free_custom_function() {
    MPI_Op_free(&minsummax);
}

void MPIWrapper::make_all_mem_available() {
    MPI_RMA_MemAllocator::make_all_available();
}

void MPIWrapper::lock_window(int rank, MPI_Locktype lock_type) {
    RelearnException::check(rank >= 0, "rank was: %d", rank);
    const auto lock_type_int = static_cast<int>(lock_type);

    // NOLINTNEXTLINE
    const int errorcode = MPI_Win_lock(lock_type_int, rank, MPI_MODE_NOCHECK, MPI_RMA_MemAllocator::mpi_window);
    RelearnException::check(errorcode == 0, "Error in lock window");
}

void MPIWrapper::unlock_window(int rank) {
    RelearnException::check(rank >= 0, "rank was: %d", rank);
    const int errorcode = MPI_Win_unlock(rank, MPI_RMA_MemAllocator::mpi_window);
    RelearnException::check(errorcode == 0, "Error in unlock window");
}

void MPIWrapper::finalize() /*noexcept*/ {
    free_custom_function();

    MPI_RMA_MemAllocator::finalize();

    const int errorcode = MPI_Finalize();
    RelearnException::check(errorcode == 0, "Error in finalize");
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

#endif
