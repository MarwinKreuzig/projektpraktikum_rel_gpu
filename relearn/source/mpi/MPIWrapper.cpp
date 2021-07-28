/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "../algorithm/BarnesHutCell.h"
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
#include <map>
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

std::map<MPIWrapper::AsyncToken, MPI_Request> translation_map{};
size_t current_token{ 0 };

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

void MPIWrapper::init_buffer_octree() {
    MPI_RMA_MemAllocator<BarnesHutCell>::init(Constants::mpi_alloc_mem);
}

void MPIWrapper::barrier() {
    const int errorcode = MPI_Barrier(MPI_COMM_WORLD);
    RelearnException::check(errorcode == 0, "Error in barrier");
}

double MPIWrapper::reduce(double value, ReduceFunction function, int root_rank) {
    RelearnException::check(root_rank >= 0, "In MPIWrapper::reduce, root_rank was negative");
    const MPI_Op* mpi_reduce_function = (MPI_Op*)translate_reduce_function(function);

    double result = 0.0;
    // NOLINTNEXTLINE
    const int errorcode = MPI_Reduce(&value, &result, 1, MPI_DOUBLE, *mpi_reduce_function, root_rank, MPI_COMM_WORLD);
    RelearnException::check(errorcode == 0, "Error in reduce");

    delete mpi_reduce_function;

    return result;
}

double MPIWrapper::all_reduce(double value, ReduceFunction function) {
    const MPI_Op* mpi_reduce_function = (MPI_Op*)translate_reduce_function(function);

    double result = 0.0;
    // NOLINTNEXTLINE
    const int errorcode = MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, *mpi_reduce_function, MPI_COMM_WORLD);
    RelearnException::check(errorcode == 0, "Error in all reduce");

    delete mpi_reduce_function;

    return result;
}

void MPIWrapper::reduce_double(const double* src, double* dst, size_t size, ReduceFunction function, int root_rank) {
    const MPI_Op* mpi_reduce_function = (MPI_Op*)translate_reduce_function(function);

    // NOLINTNEXTLINE
    const int errorcode = MPI_Reduce(src, dst, size, MPI_DOUBLE, *mpi_reduce_function, root_rank, MPI_COMM_WORLD);
    RelearnException::check(errorcode == 0, "Error in reduce: %d", errorcode);

    delete mpi_reduce_function;
}

void MPIWrapper::reduce_int64(const int64_t* src, int64_t* dst, size_t size, ReduceFunction function, int root_rank) {
    const MPI_Op* mpi_reduce_function = (MPI_Op*)translate_reduce_function(function);

    // NOLINTNEXTLINE
    const int errorcode = MPI_Reduce(src, dst, size, MPI_INT64_T, *mpi_reduce_function, root_rank, MPI_COMM_WORLD);
    RelearnException::check(errorcode == 0, "Error in reduce: %d", errorcode);

    delete mpi_reduce_function;
}

void MPIWrapper::all_to_all(const std::vector<size_t>& src, std::vector<size_t>& dst) {
    const size_t count_src = src.size();
    const size_t count_dst = dst.size();

    RelearnException::check(count_src == count_dst, "Error in all to all: size");

    // NOLINTNEXTLINE
    const int errorcode = MPI_Alltoall(src.data(), sizeof(size_t), MPI_CHAR, dst.data(), sizeof(size_t), MPI_CHAR, MPI_COMM_WORLD);
    RelearnException::check(errorcode == 0, "Error in all to all, mpi");
}

void MPIWrapper::async_s(const void* buffer, int count, int rank, AsyncToken& token) {
    RelearnException::check(rank >= 0, "Error in async s, rank is <= 0");

    token = current_token++;
    MPI_Request& translated_token = translation_map[token];

    // NOLINTNEXTLINE
    const int errorcode = MPI_Isend(buffer, count, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &translated_token);
    RelearnException::check(errorcode == 0, "Error in async send");
}

void MPIWrapper::async_recv(void* buffer, int count, int rank, AsyncToken& token) {
    RelearnException::check(rank >= 0, "Error in async recv, rank is <= 0");

    token = current_token++;
    MPI_Request& translated_token = translation_map[token];

    // NOLINTNEXTLINE
    const int errorcode = MPI_Irecv(buffer, count, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &translated_token);
    RelearnException::check(errorcode == 0, "Error in async receive");
}

int MPIWrapper::translate_lock_type(MPI_Locktype lock_type) {
    switch (lock_type) {
    case MPI_Locktype::exclusive:
        return MPI_LOCK_EXCLUSIVE;
    case MPI_Locktype::shared:
        return MPI_LOCK_SHARED;
    }

    return 0;
}

void MPIWrapper::reduce(const void* src, void* dst, int size, ReduceFunction function, int root_rank) {
    const MPI_Op* mpi_reduce_function = (MPI_Op*)translate_reduce_function(function);

    const auto* s_ptr = reinterpret_cast<const int64_t*>(src);

    // NOLINTNEXTLINE
    const int errorcode = MPI_Reduce(src, dst, size, MPI_CHAR, *mpi_reduce_function, root_rank, MPI_COMM_WORLD);
    RelearnException::check(errorcode == 0, "Error in reduce: %d", errorcode);

    delete mpi_reduce_function;
}

void MPIWrapper::all_gather(const void* own_data, void* buffer, int size) {
    // NOLINTNEXTLINE
    const int errorcode = MPI_Allgather(own_data, size, MPI_CHAR, buffer, size, MPI_CHAR, MPI_COMM_WORLD);
    RelearnException::check(errorcode == 0, "Error in all gather");
}

void MPIWrapper::all_gather_inl(void* ptr, int count) {
    RelearnException::check(count > 0, "Error in all gather , count is not greater than 0");

    // NOLINTNEXTLINE
    const int errorcode = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, ptr, count, MPI_CHAR, MPI_COMM_WORLD);
    RelearnException::check(errorcode == 0, "Error in all gather ");
}

void MPIWrapper::get(void* origin, size_t size, int target_rank, int64_t displacement) {
    
    const MPI_Aint displacement_mpi(displacement);
    const auto mpi_window = *(MPI_Win*)(MPI_RMA_MemAllocator<BarnesHutCell>::mpi_window);

    const int errorcode = MPI_Get(origin, size, MPI_CHAR, target_rank, displacement_mpi, size, MPI_CHAR, mpi_window);
    RelearnException::check(errorcode == 0, "Error in get");
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
    return MPI_RMA_MemAllocator<BarnesHutCell>::get_num_avail_objects();
}

std::string MPIWrapper::get_my_rank_str() {
    RelearnException::check(my_rank >= 0, "MPIWrapper is not initialized");
    return my_rank_str;
}

void MPIWrapper::wait_request(AsyncToken& request) {
    MPI_Request translated_token = translation_map[request];
    translation_map.erase(request);

    // NOLINTNEXTLINE
    const int errorcode = MPI_Wait(&translated_token, MPI_STATUS_IGNORE);
    RelearnException::check(errorcode == 0, "Error in wait_request ");
}

void MPIWrapper::wait_all_tokens(std::vector<AsyncToken>& tokens) {
    const int size = static_cast<int>(tokens.size());
    // NOLINTNEXTLINE

    std::vector<MPI_Request> requests(size);

    for (auto i = 0; i < size; i++) {
        const auto& request = tokens[i];
        MPI_Request translated_token = translation_map[request];
        translation_map.erase(request);
        requests[i] = translated_token;
    }

    const int errorcode = MPI_Waitall(size, requests.data(), MPI_STATUSES_IGNORE);
    RelearnException::check(errorcode == 0, "Error in wait_all_tokens");
}

void* MPIWrapper::translate_reduce_function(ReduceFunction rf) {
    MPI_Op* op = new MPI_Op;

    switch (rf) {
    case ReduceFunction::min:
        // NOLINTNEXTLINE
        *op = MPI_MIN;
        break;

    case ReduceFunction::max:
        // NOLINTNEXTLINE
        *op = MPI_MAX;
        break;

    case ReduceFunction::sum:
        // NOLINTNEXTLINE
        *op = MPI_SUM;
        break;

    case ReduceFunction::minsummax:
        *op = *(MPI_Op*)minsummax;
        break;

    default:
        RelearnException::fail("In reduce, got wrong function");
        // NOLINTNEXTLINE
        return nullptr;
    }

    return op;
}

void MPIWrapper::register_custom_function() {
    minsummax = new MPI_Op;
    // NOLINTNEXTLINE
    MPI_Op_create((MPI_User_function*)MPIUserDefinedOperation::min_sum_max, 1, (MPI_Op*)minsummax);
}

void MPIWrapper::free_custom_function() {
    MPI_Op_free((MPI_Op*)minsummax);
    delete minsummax;
}

void MPIWrapper::make_all_mem_available() {
    MPI_RMA_MemAllocator<BarnesHutCell>::make_all_available();
}

void MPIWrapper::lock_window(int rank, MPI_Locktype lock_type) {
    RelearnException::check(rank >= 0, "rank was: %d", rank);
    const auto lock_type_int = translate_lock_type(lock_type);

    // NOLINTNEXTLINE
    const auto mpi_window = *(MPI_Win*)(MPI_RMA_MemAllocator<BarnesHutCell>::mpi_window);
    const int errorcode = MPI_Win_lock(lock_type_int, rank, MPI_MODE_NOCHECK, mpi_window);
    RelearnException::check(errorcode == 0, "Error in lock window");
}

void MPIWrapper::unlock_window(int rank) {
    RelearnException::check(rank >= 0, "rank was: %d", rank);
    const auto mpi_window = *(MPI_Win*)(MPI_RMA_MemAllocator<BarnesHutCell>::mpi_window);
    const int errorcode = MPI_Win_unlock(rank, mpi_window);
    RelearnException::check(errorcode == 0, "Error in unlock window");
}

void MPIWrapper::finalize() /*noexcept*/ {
    free_custom_function();

    MPI_RMA_MemAllocator<BarnesHutCell>::finalize();

    const int errorcode = MPI_Finalize();
    RelearnException::check(errorcode == 0, "Error in finalize");
}

// This combination function assumes that it's called with the correct MPI datatype
void MPIUserDefinedOperation::min_sum_max(const int* invec, int* inoutvec, const int* const len, [[maybe_unused]] void* dtype) /*noexcept*/ {
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
