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

#include "../Config.h"

#if !RELEARN_MPI_FOUND
#include "MPINoWrapper.h"

using MPIWrapper = MPINoWrapper;
#else // #if MPI_FOUND

#include "../io/LogFiles.h"
#include "../util/MemoryHolder.h"
#include "../util/RelearnException.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

template <typename T>
class OctreeNode;
class RelearnTest;

/**
 * This enum allows a type safe choice of locking types for memory windows
 */
enum class MPI_Locktype {
    exclusive,
    shared,
};

namespace MPIUserDefinedOperation {
/**
 * @brief Provides a custom reduction function for MPI that simultaneously computes the min, sum, and max of multiple values.
 * @param invec A double* (cast to int* because of MPI) with a tuple of data to reduce. 
 *      Size must be at least *len / sizeof(double) / 3
 * @param inoutvec A double* (cast to int* because of MPI) with a tuple of data to reduce. 
 *      Size must be at least *len / sizeof(double) / 3. 
 *      Is also used as return value.
 * @param len The length of a tuple of data. Is only accessed hat *len.
 * @param dtype Unused
 */
void min_sum_max(const void* invec, void* inoutvec, const int* len, void* dtype);
} // namespace MPIUserDefinedOperation

/** 
 * This class provides a static interface to every kind of MPI functionality that should be called from other classes.
 * It wraps functionality in a C++ type safe manner.
 * The first call must be MPIWrapper::init(...) and the last one MPIWrapper::finalize(), not calling any of those inbetween.
 */
class MPIWrapper {
    friend class RelearnTest;

public:
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

    using AsyncToken = size_t;

private:
    MPIWrapper() = default;

    [[nodiscard]] static size_t init_window(size_t size_requested, size_t octree_node_size);

    static void init_globals();

    static inline void* minsummax{};

    static void* translate_reduce_function(ReduceFunction rf);

    static void register_custom_function();

    static void free_custom_function();

    static inline int num_ranks{ -1 }; // Number of ranks in MPI_COMM_WORLD
    static inline int my_rank{ -1 }; // My rank in MPI_COMM_WORLD

    static inline int thread_level_provided{ -1 }; // Thread level provided by MPI

    //NOLINTNEXTLINE
    static inline void* mpi_window{ nullptr }; // RMA window object

    static inline void* base_ptr{ nullptr }; // Start address of MPI-allocated memory
    static inline std::vector<int64_t> base_pointers{}; // RMA window base pointers of all procs

    // NOLINTNEXTLINE
    static inline std::string my_rank_str{ "-1" };

    static void all_gather(const void* own_data, void* buffer, int size);

    static void all_gather_inl(void* ptr, int count);

    static void reduce(const void* src, void* dst, int size, ReduceFunction function, int root_rank);

    // NOLINTNEXTLINE
    static void async_s(const void* buffer, int count, int rank, AsyncToken& token);

    // NOLINTNEXTLINE
    static void async_recv(void* buffer, int count, int rank, AsyncToken& token);

    static int translate_lock_type(MPI_Locktype lock_type);

    static void get(void* origin, size_t size, int target_rank, int64_t displacement);

    static void reduce_int64(const int64_t* src, int64_t* dst, size_t size, ReduceFunction function, int root_rank);

    static void reduce_double(const double* src, double* dst, size_t size, ReduceFunction function, int root_rank);

    /**
     * @brief Returns the base addresses of the memory windows of all memory windows.
     * @return The base addresses of the memory windows. The base address for MPI rank i
     *      is found at <return>[i]
     */
    [[nodiscard]] static const std::vector<int64_t>& get_base_pointers() noexcept {
        return base_pointers;
    }

public:
    /**
     * @brief Initializes the local MPI implementation via MPI_Init_Thread;
     *      initializes the global variables and the custom functions. Must be called before any other call to a member function.
     * @param argc Is passed to MPI_Init_Thread
     * @param argv Is passed to MPI_Init_Thread
     */
    static void init(int argc, char** argv);

    /**
     * @brief Initializes the shared RMA memory. Must be called before any call involving OctreeNode*.
     */
    template <typename AdditionalCellAttributes>
    static void init_buffer_octree() {
        const auto octree_node_size = sizeof(OctreeNode<AdditionalCellAttributes>);
        size_t max_num_objects = init_window(Constants::mpi_alloc_mem, octree_node_size);

        // NOLINTNEXTLINE
        auto* cast = reinterpret_cast<OctreeNode<AdditionalCellAttributes>*>(base_ptr);

        MemoryHolder<AdditionalCellAttributes>::init(cast, max_num_objects);

        LogFiles::print_message_rank(0, "MPI RMA MemAllocator: max_num_objects: {}  sizeof(OctreeNode): {}", max_num_objects, sizeof(OctreeNode<AdditionalCellAttributes>));
    }

    /**
     * @brief The calling MPI rank halts until all MPI ranks reach the method.
     * @exception Throws a RelearnException if an MPI error occurs
     */
    static void barrier();

    /**
     * @brief Reduces a value for every MPI rank with a reduction function such that the root_rank has the final result
     * @param value The local value that should be reduced
     * @param function The reduction function, should be associative and commutative
     * @param root_rank The MPI rank that shall hold the final result
     * @exception Throws a RelearnException if an MPI error occurs or if root_rank is < 0
     * @return On the MPI rank root_rank: The result of the reduction; A dummy value on every other MPI rank
     */
    [[nodiscard]] static double reduce(double value, ReduceFunction function, int root_rank);

    /**
     * @brief Reduces a value for every MPI rank with a reduction function such that every rank has the final result
     * @param value The local value that should be reduced
     * @param function The reduction function, should be associative and commutative
     * @exception Throws a RelearnException if an MPI error occurs
     * @return The final result of the reduction
     */
    [[nodiscard]] static double all_reduce_double(double value, ReduceFunction function);

    /**
     * @brief Reduces a value for every MPI rank with a reduction function such that every rank has the final result
     * @param value The local value that should be reduced
     * @param function The reduction function, should be associative and commutative
     * @exception Throws a RelearnException if an MPI error occurs
     * @return The final result of the reduction
     */
    [[nodiscard]] static uint64_t all_reduce_uint64(uint64_t value, ReduceFunction function);

    /**
     * @brief Reduces multiple values for every MPI rank with a reduction function such that the root_rank has the final result. The reduction is performed componentwise
     * @param src The local array of values that shall be reduced
     * @param function The reduction function, should be associative and commutative
     * @param root_rank The MPI rank that shall hold the final result
     * @exception Throws a RelearnException if an MPI error occurs or if root_rank is < 0
     * @return On the MPI rank root_rank: The results of the componentwise reduction; A dummy value on every other MPI rank
     */
    template <size_t size>
    [[nodiscard]] static std::array<double, size> reduce(const std::array<double, size>& src, const ReduceFunction function, const int root_rank) {
        RelearnException::check(root_rank >= 0, "MPIWrapper::reduce: root_rank was negative");

        std::array<double, size> dst{ 0.0 };
        reduce_double(src.data(), dst.data(), size, function, root_rank);

        return dst;
    }

    /**
     * @brief Reduces multiple values for every MPI rank with a reduction function such that the root_rank has the final result. The reduction is performed componentwise
     * @param src The local array of values that shall be reduced
     * @param function The reduction function, should be associative and commutative
     * @param root_rank The MPI rank that shall hold the final result
     * @exception Throws a RelearnException if an MPI error occurs or if root_rank is < 0
     * @return On the MPI rank root_rank: The results of the componentwise reduction; A dummy value on every other MPI rank
     */
    template <size_t size>
    [[nodiscard]] static std::array<int64_t, size> reduce(const std::array<int64_t, size>& src, const ReduceFunction function, const int root_rank) {
        RelearnException::check(root_rank >= 0, "MPIWrapper::reduce: root_rank was negative");

        std::array<int64_t, size> dst{ 0 };
        reduce_int64(src.data(), dst.data(), size, function, root_rank);

        return dst;
    }

    /**
     * @brief Exchanges one size_t between every pair for MPI ranks
     * @param src The values that shall be sent to the other MPI ranks. MPI rank i receives src[i]
     * @param dst The values that were transmitted by the other MPI ranks. MPI rank i sent dst[i]
     * @exception Throws a RelearnException if an MPI error occurs or if src.size() != dst.size()
     */
    // NOLINTNEXTLINE
    static void all_to_all(const std::vector<size_t>& src, std::vector<size_t>& dst);

    /**
     * @brief Gathers one value for each MPI rank into a vector on all MPI ranks
     * @param own_data The local value that shall be sent to all MPI ranks
     * @param results The data from all MPI ranks. The value of MPI rank i is in results[i]
     * @exception Throws a RelearnException if an MPI error occurs
     */
    template <typename T>
    static void all_gather(T own_data, std::vector<T>& results) {
        all_gather(&own_data, results.data(), sizeof(T));
    }

    /**
     * @brief Gathers multiple values for each MPI rank into the provided buffer on all MPI ranks
     * @param ptr The buffer to which the data will be written. The values of MPI rank i are in ptr[count * i + {0, 1, ..., count - 1}]
     * @param count The number of local values that shall be gathered
     * @exception Throws a RelearnException if an MPI error occurs or if count <= 0
     */
    template <typename T>
    static void all_gather_inline(T* ptr, const int count) {
        all_gather_inl(ptr, count * sizeof(T));
    }

    /**
     * @brief Sends data to another MPI rank asynchronously
     * @param buffer The data that shall be sent to the other MPI rank
     * @param size_in_bytes The number of bytes that shall be sent
     * @param rank The other MPI rank that shall receive the data
     * @param token A token that can be used to query if the asynchronous communication completed
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    template <typename T>
    // NOLINTNEXTLINE
    static void async_send(const T* buffer, const size_t size_in_bytes, const int rank, AsyncToken& token) {
        async_s(buffer, static_cast<int>(size_in_bytes), rank, token);
    }

    /**
     * @brief Receives data from another MPI rank asynchronously
     * @param buffer The address where the data shall be written to
     * @param size_in_bytes The number of bytes that shall be received
     * @param rank The other MPI rank that shall send the data
     * @param token A token that can be used to query if the asynchronous communication completed
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    template <typename T>
    // NOLINTNEXTLINE
    static void async_receive(T* buffer, const size_t size_in_bytes, const int rank, AsyncToken& token) {
        async_recv(buffer, static_cast<int>(size_in_bytes), rank, token);
    }

    /**
     * @brief Waits for the token if it is not a dummy token
     * @param request The token to be waited on
     * @exception Throws a RelearnException if an MPI error occurs
     */
    // NOLINTNEXTLINE
    static void wait_request(AsyncToken& request);

    /**
     * @brief Waits for all supplied tokens
     * @param The tokens to be waited on
     * @exception Throws a RelearnException if an MPI error occurs
     */
    // NOLINTNEXTLINE
    static void wait_all_tokens(std::vector<AsyncToken>& tokens);

    /** 
     * @brief Downloads an OctreeNode on another MPI rank
     * @param dst The local node which shall be the copy of the remote node
     * @param target_rank The other MPI rank
     * @param src The pointer to the remote node, must be inside the remote's memory window
     * @exception Throws a RelearnException if an MPI error occurs or if target_rank < 0
     */
    template <typename AdditionalCellAttributes>
    static void download_octree_node(OctreeNode<AdditionalCellAttributes>* dst, const int target_rank, const OctreeNode<AdditionalCellAttributes>* src) {
        RelearnException::check(target_rank >= 0, "MPIWrapper::download_octree_node: target_rank is negative");
        const auto& base_ptrs = get_base_pointers();
        RelearnException::check(target_rank < base_ptrs.size(), "MPIWrapper::download_octree_node: target_rank is larger than the pointers");
        const auto displacement = int64_t(src) - base_ptrs[target_rank];

        get(dst, sizeof(OctreeNode<AdditionalCellAttributes>), target_rank, displacement);
    }

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
     * @brief Locks the memory window on another MPI rank with the desired read/write protections
     * @param rank The other MPI rank
     * @param lock_type The type of locking
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    static void lock_window(int rank, MPI_Locktype lock_type);

    /**
     * @brief Unlocks the memory window on another MPI rank
     * @param The other MPI rank
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    static void unlock_window(int rank);

    /**
     * @brief Finalizes the local MPI implementation.
     * @exception Throws a RelearnException if an MPI error occurs
     */
    static void finalize();
};

#endif
