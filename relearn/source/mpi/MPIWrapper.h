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

    static void get(void* ptr, int size, int target_rank, int64_t target_displacement);

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
     *      initializes the global variables and the custom functions. Must be called before any other call to a member function.
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
     * @brief Reduces a value for every MPI rank in the given scope with a reduction function such that the root_rank has the final result
     * @parameter value The local value that should be reduced
     * @parameter function The reduction function, should be associative and commutative
     * @parameter root_rank The MPI rank that shall hold the final result
     * @parameter scope The scope in which the reduction has to take place
     * @exception Throws a RelearnException if an MPI error occurs or if root_rank is < 0
     * @return On the MPI rank root_rank: The result of the reduction; A dummy value on every other MPI rank
     */
    [[nodiscard]] static double reduce(double value, ReduceFunction function, int root_rank, Scope scope);

    /**
     * @brief Reduces a value for every MPI rank in the given scope with a reduction function such that every rank has the final result
     * @parameter value The local value that should be reduced
     * @parameter function The reduction function, should be associative and commutative
     * @parameter scope The scope in which the reduction has to take place
     * @exception Throws a RelearnException if an MPI error occurs
     * @return The final result of the reduction
     */
    [[nodiscard]] static double all_reduce(double value, ReduceFunction function, Scope scope);

    /**
     * @brief Reduces multiple values for every MPI rank in the given scope with a reduction function such that the root_rank has the final result. The reduction is performed componentwise
     * @parameter src The local array of values that shall be reduced
     * @parameter function The reduction function, should be associative and commutative
     * @parameter root_rank The MPI rank that shall hold the final result
     * @parameter scope The scope in which the reduction has to take place
     * @exception Throws a RelearnException if an MPI error occurs or if root_rank is < 0
     * @return On the MPI rank root_rank: The results of the componentwise reduction; A dummy value on every other MPI rank
     */
    template <typename T, size_t size>
    static std::array<T, size> reduce(const std::array<T, size>& src, ReduceFunction function, int root_rank, Scope scope) {
        RelearnException::check(root_rank >= 0, "In MPIWrapper::reduce, root_rank was negative");

        std::array<T, size> dst{};

        const auto count = static_cast<int>(src.size() * sizeof(T));
        reduce(src.data(), dst.data(), count, function, root_rank, scope);

        return dst;
    }

    /**
     * @brief Exchanges one size_t between every pair for MPI ranks in the given scope
     * @parameter src The values that shall be sent to the other MPI ranks. MPI rank i receives src[i]
     * @parameter dst The values that were transmitted by the other MPI ranks. MPI rank i sent dst[i]
     * @parameter scope The scope in which the all to all communication has to take place
     * @exception Throws a RelearnException if an MPI error occurs or if src.size() != dst.size()
     */
    // NOLINTNEXTLINE
    static void all_to_all(const std::vector<size_t>& src, std::vector<size_t>& dst, Scope scope);

    /**
     * @brief Gathers one value for each MPI rank into a vector on all MPI ranks
     * @parameter own_data The local value that shall be sent to all MPI ranks
     * @parameter results The data from all MPI ranks. The value of MPI rank i is in results[i]
     * @parameter scope The scope in which the all to all communication has to take place
     * @exception Throws a RelearnException if an MPI error occurs
     */
    template <typename T>
    static void all_gather(T own_data, std::vector<T>& results, Scope scope) {
        all_gather(&own_data, results.data(), sizeof(T), scope);
    }

    /**
     * @brief Gathers multiple values for each MPI rank into the provided buffer on all MPI ranks
     * @parameter ptr The buffer to which the data will be written. The values of MPI rank i are in ptr[count * i + {0, 1, ..., count - 1}]
     * @parameter count The number of local values that shall be gathered
     * @parameter scope The scope in which the all to all communication has to take place
     * @exception Throws a RelearnException if an MPI error occurs or if count <= 0
     */
    template <typename T>
    static void all_gather_inline(T* ptr, int count, Scope scope) {
        all_gather_inl(ptr, count * sizeof(T), scope);
    }

    /**
     * @brief Sends data to another MPI rank asynchronously
     * @parameter buffer The data that shall be sent to the other MPI rank
     * @parameter size_in_bytes The number of bytes that shall be sent
     * @parameter rank The other MPI rank that shall receive the data
     * @parameter scope The scope in which the communication has to take place
     * @parameter token A token that can be used to query if the asynchronous communication completed
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    template <typename T>
    // NOLINTNEXTLINE
    static void async_send(const T* buffer, size_t size_in_bytes, int rank, Scope scope, AsyncToken& token) {
        async_s(buffer, static_cast<int>(size_in_bytes), rank, scope, token);
    }

    /**
     * @brief Receives data from another MPI rank asynchronously
     * @parameter buffer The address where the data shall be written to
     * @parameter size_in_bytes The number of bytes that shall be received
     * @parameter rank The other MPI rank that shall send the data
     * @parameter scope The scope in which the communication has to take place
     * @parameter token A token that can be used to query if the asynchronous communication completed
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    template <typename T>
    // NOLINTNEXTLINE
    static void async_receive(T* buffer, size_t size_in_bytes, int rank, Scope scope, AsyncToken& token) {
        async_recv(buffer, static_cast<int>(size_in_bytes), rank, scope, token);
    }

    /**
     * @brief Waits for the token if it is not a dummy token
     * @parameter request The token to be waited on
     * @exception Throws a RelearnException if an MPI error occurs
     */
    // NOLINTNEXTLINE
    static void wait_request(AsyncToken& request);

    /**
     * @brief Waits for all supplied tokens
     * @parameter The tokens to be waited on
     * @exception Throws a RelearnException if an MPI error occurs
     */
    // NOLINTNEXTLINE
    static void wait_all_tokens(std::vector<AsyncToken>& tokens);

    /** 
     * @brief Downloads an OctreeNode on another MPI rank
     * @parameter dst The local node which shall be the copy of the remote node
     * @parameter target_rank The other MPI rank
     * @parameter src The pointer to the remote node, must be inside the remote's memory window
     * @exception Throws a RelearnException if an MPI error occurs or if target_rank < 0
     */
    static void download_octree_node(OctreeNode* dst, int target_rank, const OctreeNode* src);

    /**
     * @brief Creates a new OctreeNode in the local memory window
     * @exception Throws a RelearnException if no shared memory is available
     * @return A valid pointer to an OctreeNode
     */
    [[nodiscard]] static OctreeNode* new_octree_node();

    /**
     * @brief Deletes an OctreeNode in the memory window that was previously created via new_octree_node()
     * @parameter ptr A pointer to the object that shall be deleted
     */
    static void delete_octree_node(OctreeNode* ptr);

    /**
     * @brief Returns the number of MPI ranks
     * @exception Throws a RelearnException if the MPIWrapper is not initialized
     * @return The number of MPI ranks
     */
    [[nodiscard]] static int get_num_ranks();

    /**
     * @brief Returns the current MPI rank's id
     * @exception Throws a RelearnException if the MPIWrapper is not initialized
     * @return The current MPI rank's id
     */
    [[nodiscard]] static int get_my_rank();

    /**
     * @brief Returns the current MPI rank's id as string
     * @exception Throws a RelearnException if the MPIWrapper is not initialized
     * @return The current MPI rank's id as string
     */
    [[nodiscard]] static std::string get_my_rank_str();

    /**
     * @brief Returns the number of still available OctreeNodes
     * @return The number of still available OctreeNodes
     */
    [[nodiscard]] static size_t get_num_avail_objects();

    /**
     * @brief Returns the OctreeNodes that are used to synchronize the local trees.
     *      MPI rank i owns the objects return[i * num_local_trees + {0, 1, ..., num_local_trees - 1}]
     *      The number of objects can be requested via get_num_buffer_octree_nodes()
     * @return A pointer to the local trees on each rank
     */
    [[nodiscard]] static OctreeNode* get_buffer_octree_nodes();

    /**
     * @brief Returns the number of local trees across all MPI ranks
     * @return The number of local trees across all MPI ranks
     */
    [[nodiscard]] static size_t get_num_buffer_octree_nodes();

    /**
     * @brief Locks the memory window on another MPI rank with the desired read/write protections
     * @parameter rank The other MPI rank
     * @parameter lock_type The type of locking
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    static void lock_window(int rank, MPI_Locktype lock_type);

    /**
     * @brief Unlocks the memory window on another MPI rank
     * @parameter The other MPI rank
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    static void unlock_window(int rank);

    /**
     * @brief Finalizes the local MPI implementation.
     * @exception Throws a RelearnException if an MPI error occurs
     */
    static void finalize() /*noexcept*/;
};

#endif
