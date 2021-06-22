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

#include "../util/RelearnException.h"

#if !MPI_FOUND
#include "MPINoWrapper.h"

using MPIWrapper = MPINoWrapper;
#else // #if MPI_FOUND

#include "MPITypes.h"

#include <array>
#include <cstdint>
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

/** 
 * This class provides a static interface to every kind of MPI functionality that should be called from other classes.
 * It wraps functionality in a C++ type safe manner.
 * The first call must be MPIWrapper::init(...) and the last one MPIWrapper::finalize(), not calling any of those inbetween.
 */
class MPIWrapper {
    struct RMABufferOctreeNodes {
        OctreeNode* ptr;
        size_t num_nodes;
    };

public:
    /**
     * This enum serves as a marker for the scope of the MPI functions. 
     * It only serves the purpose of providing an interface for future work.
     * Scope::none is not supported and always triggers a RelearnException.
     */
    enum class Scope : char {
        global = 0,
        none = 1
    };

    /**
     * This enum serves as a marker for the function that should be used in reductions.
     * ReduceFunction::none is not supported and always triggers a RelearnException.
     */
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

    static void init_globals();

    static inline MPI_Op minsummax{};

    static MPI_Op translate_reduce_function(ReduceFunction rf);

    static MPI_Comm translate_scope(Scope scope);

    static void register_custom_function();

    static void free_custom_function();

    static inline RMABufferOctreeNodes rma_buffer_branch_nodes{};

    static inline int num_ranks{ -1 }; // Number of ranks in MPI_COMM_WORLD
    static inline int my_rank{ -1 }; // My rank in MPI_COMM_WORLD

    static inline int thread_level_provided{ -1 }; // Thread level provided by MPI

    // NOLINTNEXTLINE
    static inline std::string my_rank_str{ "-1" };

    static void get(void* ptr, int size, int target_rank, int64_t target_display);

    static void all_gather(const void* own_data, void* buffer, int size, Scope scope);

    static void all_gather_inl(void* ptr, int count, Scope scope);

    static void reduce(const void* src, void* dst, int size, ReduceFunction function, int root_rank, Scope scope);

    // NOLINTNEXTLINE
    static void async_s(const void* buffer, int count, int rank, Scope scope, AsyncToken& token);

    // NOLINTNEXTLINE
    static void async_recv(void* buffer, int count, int rank, Scope scope, AsyncToken& token);

public:
    /**
     * @brief Initializes the local MPI implementation via MPI_Init_Thread;
     *      initializes the global variables and the custom functions. Must be called before every other call to a member function.
     * @parameter argc Is passed to MPI_Init_Thread
     * @parameter argv Is passed to MPI_Init_Thread
     */
    static void init(int argc, char** argv);

    /**
     * @brief Initializes the shared RMA memory. Must be called before any call involving OctreeNode*.
     * @parameter num_partitions The number of partitions across all MPI ranks (of the form 8^k)
     */
    static void init_buffer_octree(size_t num_partitions);

    /**
     * @brief The calling MPI rank halts until all MPI ranks within the scope reach the method.
     * @parameter scope The scope in which the MPI ranks are synchronized
     * @exception Throws a RelearnException if an MPI error occurs or scope is Scope::none
     */
    static void barrier(Scope scope);

    /**
     * @brief
     * @parameter
     * @parameter
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs or if root_rank is < 0
     * @return
     */
    [[nodiscard]] static double reduce(double value, ReduceFunction function, int root_rank, Scope scope);

    /**
     * @brief
     * @parameter
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs
     * @return
     */
    [[nodiscard]] static double all_reduce(double value, ReduceFunction function, Scope scope);

    /**
     * @brief
     * @parameter
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs or if src.size() != dst.size()
     */
    // NOLINTNEXTLINE
    static void all_to_all(const std::vector<size_t>& src, std::vector<size_t>& dst, Scope scope);

    /**
     * @brief
     * @parameter
     * @parameter
     * @parameter
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    template <typename T>
    // NOLINTNEXTLINE
    static void async_send(const T* buffer, size_t size_in_bytes, int rank, Scope scope, AsyncToken& token) {
        async_s(buffer, static_cast<int>(size_in_bytes), rank, scope, token);
    }

    /**
     * @brief
     * @parameter
     * @parameter
     * @parameter
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    template <typename T>
    // NOLINTNEXTLINE
    static void async_receive(T* buffer, size_t size_in_bytes, int rank, Scope scope, AsyncToken& token) {
        async_recv(buffer, static_cast<int>(size_in_bytes), rank, scope, token);
    }

    /**
     * @brief
     * @parameter
     * @parameter
     * @parameter
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs, if root_rank is < 0 or if src.size() != dst.size()
     */
    template <typename T, size_t size>
    static void reduce(const std::array<T, size>& src, std::array<T, size>& dst, ReduceFunction function, int root_rank, Scope scope) {
        RelearnException::check(root_rank >= 0, "In MPIWrapper::reduce, root_rank was negative");
        RelearnException::check(src.size() == dst.size(), "Sizes of vectors don't match");

        const auto count = static_cast<int>(src.size() * sizeof(T));
        reduce(src.data(), dst.data(), count, function, root_rank, scope);
    }

    /**
     * @brief
     * @parameter
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs
     */
    template <typename T>
    static void all_gather(T own_data, std::vector<T>& results, Scope scope) {
        all_gather(&own_data, results.data(), sizeof(T), scope);
    }

    /**
     * @brief
     * @parameter
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs or if target_rank is < 0
     */
    template <typename T>
    static void get(T* ptr, int target_rank, int64_t target_display) {
        get(ptr, sizeof(T), target_rank, target_display);
    }

    /**
     * @brief
     * @parameter
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs or if count <= 0
     */
    template <typename T>
    static void all_gather_inline(T* ptr, int count, Scope scope) {
        all_gather_inl(ptr, count * sizeof(T), scope);
    }

    /**
     * @brief
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if target_rank is < 0 or larger than the number of base pointers
     * @return
     */
    [[nodiscard]] static int64_t get_ptr_displacement(int target_rank, const OctreeNode* ptr);

    /**
     * @brief
     * @exception Throws a RelearnException if no shared memory is available
     * @return
     */
    [[nodiscard]] static OctreeNode* new_octree_node();

    /**
     * @brief
     * @exception Throws a RelearnException if the MPIWrapper is not initialized
     * @return
     */
    [[nodiscard]] static int get_num_ranks();

    /**
     * @brief
     * @exception Throws a RelearnException if the MPIWrapper is not initialized
     * @return
     */
    [[nodiscard]] static int get_my_rank();

    /**
     * @brief
     * @return
     */
    [[nodiscard]] static size_t get_num_avail_objects();

    /**
     * @brief
     * @return
     */
    [[nodiscard]] static OctreeNode* get_buffer_octree_nodes();

    /**
     * @brief
     * @return
     */
    [[nodiscard]] static size_t get_num_buffer_octree_nodes();

    /**
     * @brief
     * @return
     */
    [[nodiscard]] static std::string get_my_rank_str();

    /**
     * @brief
     * @parameter
     */
    static void delete_octree_node(OctreeNode* ptr);

    /**
     * @brief
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs
     */
    // NOLINTNEXTLINE
    static void wait_request(AsyncToken& request);

    /**
     * @brief
     * @return
     */
    [[nodiscard]] static AsyncToken get_non_null_request();

    /**
     * @brief
     * @return
     */
    [[nodiscard]] static AsyncToken get_null_request();

    /**
     * @brief
     * @parameter
     * @parameter
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs
     */
    // NOLINTNEXTLINE
    static void all_gather_v(size_t total_num_neurons, std::vector<double>& xyz_pos, std::vector<int>& recvcounts, std::vector<int>& displs);

    /**
     * @brief
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs
     */
    // NOLINTNEXTLINE
    static void wait_all_tokens(std::vector<AsyncToken>& tokens);

    /**
     * @brief
     * @parameter
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs or if rank <= 0
     */
    static void lock_window(int rank, MPI_Locktype lock_type);

    /**
     * @brief
     * @parameter
     * @exception Throws a RelearnException if an MPI error occurs or if rank <= 0
     */
    static void unlock_window(int rank);

    /**
     * @brief
     * @exception Throws a RelearnException if an MPI error occurs
     */
    static void finalize() /*noexcept*/;
};

#endif
