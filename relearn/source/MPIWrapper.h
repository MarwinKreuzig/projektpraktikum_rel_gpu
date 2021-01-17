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

#include "MPI_RMA_MemAllocator.h"
#include "OctreeNode.h"

#include <mpi.h>

#include <array>
#include <string>

class Octree;

enum class MPI_Locktype : int {
    exclusive = 234,
    shared = 235,
};

namespace MPIUserDefinedOperation {
// This combination function assumes that it's called with the
// correct MPI datatype
void min_sum_max(const int* invec, int* inoutvec, const int* len, MPI_Datatype* dtype);
} // namespace MPIUserDefinedOperation

class MPIWrapper {
    struct RMABufferOctreeNodes {
        OctreeNode* ptr;
        size_t num_nodes;
    };

public:
    enum class Scope : char {
        global = 0,
        none = 1
    };

    enum class ReduceFunction : char {
        min = 0,
        max = 1,
        avg = 2,
        sum = 3,
        none = 4,
        minsummax = 100
    };

    using AsyncToken = MPI_Request;

private:
    static MPI_Op minsummax;

    static MPI_Op translate_reduce_function(ReduceFunction rf);

    static MPI_Comm translate_scope(Scope scope);

    static void register_custom_function();

    static void free_custom_function();

    static MPI_RMA_MemAllocator mpi_rma_mem_allocator;
    static RMABufferOctreeNodes rma_buffer_branch_nodes;

    static size_t num_ranks; // Number of ranks in MPI_COMM_WORLD
    static size_t my_rank; // My rank in MPI_COMM_WORLD

    static size_t num_neurons; // Total number of neurons
    static size_t my_num_neurons; // My number of neurons I'm responsible for
    static size_t my_neuron_id_start; // ID of my first neuron
    static size_t my_neuron_id_end; // ID of my last neuron

    // Needed for Allgatherv
    static std::vector<size_t> num_neurons_of_ranks; // Number of neurons that each rank is responsible for
    static std::vector<size_t> num_neurons_of_ranks_displs; // Displacements based on "num_neurons_of_ranks" (exclusive prefix sums, i.e. Exscan)

    static int thread_level_provided; // Thread level provided by MPI

    static std::string my_rank_str;

public:
    static void init(int argc, char** argv);

    static void init_globals();

    static void init_neurons(size_t num_neurons);

    static void init_buffer_octree(size_t num_partitions);

    static void barrier(Scope scope);

    [[nodiscard]] static double reduce(double value, ReduceFunction function, int root_rank, Scope scope);

    [[nodiscard]] static double all_reduce(double value, ReduceFunction function, Scope scope);

    static void all_to_all(const std::vector<size_t>& src, std::vector<size_t>& dst, Scope scope);

    template <typename T>
    static void async_send(const T* buffer, size_t size_in_bytes, int rank, Scope scope, AsyncToken& token) {
        MPI_Comm mpi_scope = translate_scope(scope);

        // NOLINTNEXTLINE
        const int errorcode = MPI_Isend(buffer, size_in_bytes, MPI_CHAR, rank, 0, mpi_scope, &token);
        RelearnException::check(errorcode == 0, "Error in async send");
    }

    template <typename T>
    static void async_receive(T* buffer, size_t size_in_bytes, int rank, Scope scope, AsyncToken& token) {
        MPI_Comm mpi_scope = translate_scope(scope);

        // NOLINTNEXTLINE
        const int errorcode = MPI_Irecv(buffer, size_in_bytes, MPI_CHAR, rank, 0, mpi_scope, &token);
        RelearnException::check(errorcode == 0, "Error in async receive");
    }

    template <typename T, size_t size>
    static void reduce(const std::array<T, size>& src, std::array<T, size>& dst, ReduceFunction function, int root_rank, Scope scope) {
        RelearnException::check(src.size() == dst.size(), "Sizes of vectors don't match");

        MPI_Comm mpi_scope = translate_scope(scope);
        MPI_Op mpi_reduce_function = translate_reduce_function(function);

        // NOLINTNEXTLINE
        const int errorcode = MPI_Reduce(src.data(), dst.data(), sizeof(T) * src.size(), MPI_CHAR, mpi_reduce_function, root_rank, mpi_scope);
        RelearnException::check(errorcode == 0, "Error in reduce: " + std::to_string(errorcode));
    }

    template <typename T>
    static void all_gather(T own_data, std::vector<T>& results, Scope scope) {
        MPI_Comm mpi_scope = translate_scope(scope);

        // NOLINTNEXTLINE
        const int errorcode = MPI_Allgather(&own_data, sizeof(T), MPI_CHAR, results.data(), sizeof(T), MPI_CHAR, mpi_scope);
        RelearnException::check(errorcode == 0, "Error in all gather");
    }

    template <typename T>
    static void all_gather_inline(T* ptr, int count, Scope scope) {
        MPI_Comm mpi_scope = translate_scope(scope);

        // NOLINTNEXTLINE
        const int errorcode = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, ptr, count * sizeof(T), MPI_CHAR, MPI_COMM_WORLD);
        RelearnException::check(errorcode == 0, "Error in all gather inline");
    }

    template <typename T>
    static void get(T* ptr, int target_rank, MPI_Aint target_display) {

        // NOLINTNEXTLINE
        const int errorcode = MPI_Get(ptr, sizeof(T), MPI_CHAR, target_rank, target_display, sizeof(T), MPI_CHAR, mpi_rma_mem_allocator.mpi_window);
        RelearnException::check(errorcode == 0, "Error in get");
    }

    [[nodiscard]] static MPI_Aint get_ptr_displacement(int target_rank, const OctreeNode* ptr);

    [[nodiscard]] static OctreeNode* new_octree_node();

    [[nodiscard]] static size_t get_num_ranks();

    [[nodiscard]] static size_t get_my_rank();

    [[nodiscard]] static size_t get_num_neurons();

    [[nodiscard]] static size_t get_my_num_neurons();

    [[nodiscard]] static size_t get_my_neuron_id_start();

    [[nodiscard]] static size_t get_my_neuron_id_end();

    [[nodiscard]] static size_t get_num_avail_objects();

    [[nodiscard]] static OctreeNode* get_buffer_octree_nodes();

    [[nodiscard]] static size_t get_num_buffer_octree_nodes();

    [[nodiscard]] static std::string get_my_rank_str();

    static void delete_octree_node(OctreeNode* ptr);

    static void wait_request(AsyncToken& request);

    [[nodiscard]] static AsyncToken get_non_null_request();

    [[nodiscard]] static AsyncToken get_null_request();

    static void all_gather_v(size_t total_num_neurons, std::vector<double>& xyz_pos, std::vector<int>& recvcounts, std::vector<int>& displs);

    static void wait_all_tokens(std::vector<AsyncToken>& tokens);

    static void lock_window(size_t rank, MPI_Locktype lock_type);

    static void unlock_window(size_t rank);

    static void finalize() /*noexcept*/;

    static void print_infos_rank(size_t rank);
};
