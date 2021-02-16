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

#include "MPITypes.h"
#include "RelearnException.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

class Octree;
class OctreeNode;

enum class MPI_Locktype : int {
    exclusive = MPI_LOCK_EXCLUSIVE,
    shared = MPI_LOCK_SHARED,
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
    MPIWrapper() = default;

    static MPI_Op minsummax;

    static MPI_Op translate_reduce_function(ReduceFunction rf);

    static MPI_Comm translate_scope(Scope scope);

    static void register_custom_function();

    static void free_custom_function();

    static RMABufferOctreeNodes rma_buffer_branch_nodes;

    static int num_ranks; // Number of ranks in MPI_COMM_WORLD
    static int my_rank; // My rank in MPI_COMM_WORLD

    static int thread_level_provided; // Thread level provided by MPI

    static std::string my_rank_str;

    static void get(void* ptr, int size, int target_rank, int64_t target_display);

    static void all_gather(const void* own_data, void* buffer, int size, Scope scope);

    static void all_gather_inl(void* ptr, int count, Scope scope);

    static void reduce(const void* src, void* dst, int size, ReduceFunction function, int root_rank, Scope scope);

    // NOLINTNEXTLINE
    static void async_s(const void* buffer, int count, int rank, Scope scope, AsyncToken& token);

    // NOLINTNEXTLINE
    static void async_recv(void* buffer, int count, int rank, Scope scope, AsyncToken& token);

public:
    static void init(int argc, char** argv);

    static void init_globals();

    static void init_buffer_octree(size_t num_partitions);

    static void barrier(Scope scope);

    [[nodiscard]] static double reduce(double value, ReduceFunction function, int root_rank, Scope scope);

    [[nodiscard]] static double all_reduce(double value, ReduceFunction function, Scope scope);

    // NOLINTNEXTLINE
    static void all_to_all(const std::vector<size_t>& src, std::vector<size_t>& dst, Scope scope);

    template <typename T>
    // NOLINTNEXTLINE
    static void async_send(const T* buffer, size_t size_in_bytes, int rank, Scope scope, AsyncToken& token) {
        async_s(buffer, static_cast<int>(size_in_bytes), rank, scope, token);
    }

    template <typename T>
    // NOLINTNEXTLINE
    static void async_receive(T* buffer, size_t size_in_bytes, int rank, Scope scope, AsyncToken& token) {
        async_recv(buffer, static_cast<int>(size_in_bytes), rank, scope, token);
    }

    template <typename T, size_t size>
    static void reduce(const std::array<T, size>& src, std::array<T, size>& dst, ReduceFunction function, int root_rank, Scope scope) {
        RelearnException::check(src.size() == dst.size(), "Sizes of vectors don't match");

        const auto count = static_cast<int>(src.size() * sizeof(T));
        reduce(src.data(), dst.data(), count, function, root_rank, scope);
    }

    template <typename T>
    static void all_gather(T own_data, std::vector<T>& results, Scope scope) {
        all_gather(&own_data, results.data(), sizeof(T), scope);
    }

    template <typename T>
    static void get(T* ptr, int target_rank, int64_t target_display) {
        get(ptr, sizeof(T), target_rank, target_display);
    }

    template <typename T>
    static void all_gather_inline(T* ptr, int count, Scope scope) {
        all_gather_inl(ptr, count * sizeof(T), scope);
    }

    [[nodiscard]] static int64_t get_ptr_displacement(int target_rank, const OctreeNode* ptr);

    [[nodiscard]] static OctreeNode* new_octree_node();

    [[nodiscard]] static int get_num_ranks();

    [[nodiscard]] static int get_my_rank();

    [[nodiscard]] static size_t get_num_avail_objects();

    [[nodiscard]] static OctreeNode* get_buffer_octree_nodes();

    [[nodiscard]] static size_t get_num_buffer_octree_nodes();

    [[nodiscard]] static std::string get_my_rank_str();

    static void delete_octree_node(OctreeNode* ptr);

    // NOLINTNEXTLINE
    static void wait_request(AsyncToken& request);

    [[nodiscard]] static AsyncToken get_non_null_request();

    [[nodiscard]] static AsyncToken get_null_request();

    // NOLINTNEXTLINE
    static void all_gather_v(size_t total_num_neurons, std::vector<double>& xyz_pos, std::vector<int>& recvcounts, std::vector<int>& displs);

    // NOLINTNEXTLINE
    static void wait_all_tokens(std::vector<AsyncToken>& tokens);

    static void lock_window(int rank, MPI_Locktype lock_type);

    static void unlock_window(int rank);

    static void finalize() /*noexcept*/;
};
