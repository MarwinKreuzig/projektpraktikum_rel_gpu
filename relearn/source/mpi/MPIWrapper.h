#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#if !RELEARN_MPI_FOUND
#include "MPINoWrapper.h"

using MPIWrapper = MPINoWrapper;
#else // #if MPI_FOUND

#include "CommunicationMap.h"
#include "io/LogFiles.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <vector>

template <typename T>
class OctreeNode;
class RelearnTest;

/**
 * This enum allows a type safe choice of locking types for memory windows
 */
enum class MPI_Locktype {
    Exclusive,
    Shared,
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
     * ReduceFunction::None is not supported and always triggers a RelearnException.
     */
    enum class ReduceFunction : char {
        Min = 0,
        Max = 1,
        Sum = 2,
        None = 3,
        MinSumMax = 100
    };

    using AsyncToken = size_t;

private:
    MPIWrapper() = default;

    [[nodiscard]] static size_t init_window(size_t size_requested, size_t octree_node_size);

    static void init_globals();

    static void register_custom_function();

    static void free_custom_function();

    static inline int num_ranks{ -1 }; // Number of ranks in MPI_COMM_WORLD
    static inline int my_rank{ -1 }; // My rank in MPI_COMM_WORLD

    static inline int thread_level_provided{ -1 }; // Thread level provided by MPI

    static inline void* base_ptr{ nullptr }; // Start address of MPI-allocated memory
    static inline std::vector<int64_t> base_pointers{}; // RMA window base pointers of all procs

    // NOLINTNEXTLINE
    static inline std::string my_rank_str{ "-1" };

    static inline std::atomic<uint64_t> bytes_sent{ 0 };
    static inline std::atomic<uint64_t> bytes_received{ 0 };
    static inline std::atomic<uint64_t> bytes_remote{ 0 };

    static void all_gather(const void* own_data, void* buffer, int size);

    static void all_gather_inl(void* ptr, int count);

    static void reduce(const void* src, void* dst, int size, ReduceFunction function, int root_rank);

    // NOLINTNEXTLINE
    [[nodiscard]] static AsyncToken async_s(const void* buffer, int count, int rank);

    // NOLINTNEXTLINE
    [[nodiscard]] static AsyncToken async_recv(void* buffer, int count, int rank);

    [[nodiscard]] static int translate_lock_type(MPI_Locktype lock_type);

    static void get(void* origin, size_t size, int target_rank, int64_t displacement, int number_elements);

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

    /**
     * @brief Sends data to another MPI rank asynchronously
     * @param buffer The buffer that shall be sent to the other MPI rank
     * @param rank The other MPI rank that shall receive the data
     * @param token A token that can be used to query if the asynchronous communication completed
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    template <typename T>
    [[nodiscard]] static AsyncToken async_send(std::span<T> buffer, const int rank) {
        return async_s(buffer.data(), static_cast<int>(buffer.size_bytes()), rank);
    }

    /**
     * @brief Receives data from another MPI rank asynchronously
     * @param buffer The buffer where the data shall be written to
     * @param rank The other MPI rank that shall send the data
     * @param token A token that can be used to query if the asynchronous communication completed
     * @exception Throws a RelearnException if an MPI error occurs or if rank < 0
     */
    template <typename T>
    [[nodiscard]] static AsyncToken async_receive(std::span<T> buffer, const int rank) {
        return async_recv(buffer.data(), static_cast<int>(buffer.size_bytes()), rank);
    }

    /**
     * @brief Waits for all supplied tokens
     * @param The tokens to be waited on
     * @exception Throws a RelearnException if an MPI error occurs
     */
    static void wait_all_tokens(const std::vector<AsyncToken>& tokens);

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
     * @return The values that were transmitted by the other MPI ranks. MPI rank i sent <return>[i]
     */
    static std::vector<size_t> all_to_all(const std::vector<size_t>& src);

    /**
     * @brief Gathers one value for each MPI rank into a vector on all MPI ranks
     * @param own_data The local value that shall be sent to all MPI ranks
     * @return The data from all MPI ranks. The value of MPI rank i is in results[i]
     * @exception Throws a RelearnException if an MPI error occurs
     */
    template <typename T>
    [[nodiscard]] static std::vector<T> all_gather(T own_data) {
        std::vector<T> results(get_num_ranks());
        all_gather(&own_data, results.data(), sizeof(T));
        return results;
    }

    /**
     * @brief Gathers multiple values for each MPI rank into the provided buffer on all MPI ranks
     * @param buffer The buffer to which the data will be written. The values of MPI rank i are in ptr[count * i + {0, 1, ..., count - 1}]
     * @param count The number of local values that shall be gathered
     * @exception Throws a RelearnException if an MPI error occurs or if count <= 0
     */
    template <typename T>
    static void all_gather_inline(std::span<T> buffer) {
        all_gather_inl(buffer.data(), static_cast<int>(buffer.size_bytes()));
    }

    /**
     * @brief Exchanges vectors of data with all MPI ranks
     * @tparam T The type that should be exchanged
     * @param values The values that should be exchanged. values[i] should be send to MPI rank i
     * @exception Throws a RelearnException if values.size() does not match the number of MPI ranks
     * @return The values that were received from the MPI ranks. <return>[i] on rank j was values[j] on rank i
     */
    template <typename T>
    [[nodiscard]] static std::vector<std::vector<T>> exchange_values(const std::vector<std::vector<T>>& values) {
        RelearnException::check(values.size() == num_ranks,
            "MPIWrapper::exchange_values: There are too many values: {} for the number of ranks {}!", values.size(), num_ranks);

        std::vector<size_t> request_sizes(num_ranks, 0);
        for (auto target_rank = 0; target_rank < num_ranks; target_rank++) {
            request_sizes[target_rank] = values[target_rank].size();
        }

        std::vector<size_t> response_sizes = all_to_all(request_sizes);

        std::vector<std::vector<T>> retrieved_data(num_ranks);
        for (auto rank = 0; rank < num_ranks; rank++) {
            retrieved_data[rank].resize(response_sizes[rank]);
        }

        std::vector<AsyncToken> async_tokens{};
        for (auto rank = 0; rank < num_ranks; rank++) {
            if (rank == my_rank) {
                continue;
            }

            const auto token = async_receive(std::span{ retrieved_data[rank] }, rank);
            async_tokens.emplace_back(token);
        }

        for (auto rank = 0; rank < num_ranks; rank++) {
            if (rank == my_rank) {
                continue;
            }

            const auto token = async_send(std::span{ values[rank] }, rank);
            async_tokens.emplace_back(token);
        }

        wait_all_tokens(async_tokens);
        return retrieved_data;
    }

    /**
     * @brief Exchanges data with all MPI ranks
     * @tparam RequestType The type that should be exchanged
     * @param outgoing_requests The values that should be exchanged. values[i] should be send to MPI rank i (if present)
     * @return The values that were received from the MPI ranks. <return>[i] on rank j was values[j] on rank i
     */
    template <typename RequestType>
    [[nodiscard]] static CommunicationMap<RequestType> exchange_requests(const CommunicationMap<RequestType>& outgoing_requests) {
        const auto number_ranks = get_num_ranks();
        const auto my_rank = get_my_rank();

        std::vector<size_t> number_requests_outgoing = outgoing_requests.get_request_sizes();
        std::vector<size_t> number_requests_incoming = all_to_all(number_requests_outgoing);

        const auto size_hint = outgoing_requests.size();
        CommunicationMap<RequestType> incoming_requests(number_ranks, size_hint);
        incoming_requests.resize(number_requests_incoming);

        std::vector<AsyncToken> async_tokens{};

        for (auto rank_id = 0; rank_id < number_ranks; rank_id++) {
            if (!incoming_requests.contains(rank_id)) {
                continue;
            }

            auto* buffer = incoming_requests.get_data(rank_id);
            const auto size = incoming_requests.size(rank_id);

            const auto token = async_receive(incoming_requests.get_span(rank_id), rank_id);
            async_tokens.emplace_back(token);
        }

        for (auto rank_id = 0; rank_id < number_ranks; rank_id++) {
            if (!outgoing_requests.contains(rank_id)) {
                continue;
            }

            const auto* buffer = outgoing_requests.get_data(rank_id);
            const auto size = outgoing_requests.size(rank_id);

            const auto token = async_send(outgoing_requests.get_span(rank_id), rank_id);
            async_tokens.emplace_back(token);
        }

        // Wait for all sends and receives to complete
        wait_all_tokens(async_tokens);

        return incoming_requests;
    }

    /**
     * @brief Downloads an OctreeNode on another MPI rank
     * @param dst The local node which shall be the copy of the remote node
     * @param target_rank The other MPI rank
     * @param src The pointer to the remote node, must be inside the remote's memory window
     * @param number_elements The number of elements to download
     * @exception Throws a RelearnException if an MPI error occurs, if number_elements <= 0, or if target_rank < 0
     */
    template <typename AdditionalCellAttributes>
    static void download_octree_node(OctreeNode<AdditionalCellAttributes>* dst, const int target_rank, const OctreeNode<AdditionalCellAttributes>* src, const int number_elements) {
        RelearnException::check(number_elements > 0, "MPIWrapper::download_octree_node: number_elements is not positive");
        RelearnException::check(target_rank >= 0, "MPIWrapper::download_octree_node: target_rank is negative");

        const auto& base_ptrs = get_base_pointers();
        RelearnException::check(target_rank < base_ptrs.size(), "MPIWrapper::download_octree_node: target_rank is larger than the pointers");
        const auto displacement = int64_t(src) - base_ptrs[target_rank];

        RelearnException::check(displacement >= 0, "MPIWrapper::download_octree_node: displacement is too small: {:X} - {:X}", int64_t(src), base_ptrs[target_rank]);

        get(dst, sizeof(OctreeNode<AdditionalCellAttributes>), target_rank, displacement, number_elements);
    }

    /**
     * @brief Downloads an OctreeNode on another MPI rank
     * @param dst The local node which shall be the copy of the remote node
     * @param target_rank The other MPI rank
     * @param offset The offset in the remote's memory window
     * @param number_elements The number of elements to download
     * @exception Throws a RelearnException if an MPI error occurs, if number_elements <= 0, if offset < 0, or if target_rank < 0
     */
    template <typename AdditionalCellAttributes>
    static void download_octree_node(OctreeNode<AdditionalCellAttributes>* dst, const int target_rank, const int offset, const int number_elements) {
        RelearnException::check(number_elements > 0, "MPIWrapper::download_octree_node: number_elements is not positive");
        RelearnException::check(target_rank >= 0, "MPIWrapper::download_octree_node: target_rank is negative");
        RelearnException::check(offset >= 0, "MPIWrapper::download_octree_node: offset is negative: {}", offset);

        const auto& base_ptrs = get_base_pointers();
        RelearnException::check(target_rank < base_ptrs.size(), "MPIWrapper::download_octree_node: target_rank is larger than the pointers");

        get(dst, sizeof(OctreeNode<AdditionalCellAttributes>), target_rank, offset, number_elements);
    }

    /**
     * @brief Returns the number of MPI ranks
     * @exception Throws a RelearnException if the MPIWrapper is not initialized
     * @return The number of MPI ranks
     */
    [[nodiscard]] static int get_num_ranks();

    /**
     * @brief Get a range of all ranks [0, num_ranks)
     *
     * @return auto the range of all ranks
     */
    [[nodiscard]] static const std::vector<int>& get_ranks() {
        static std::vector<int> ranks = []() {
            std::vector<int> r(get_num_ranks());
            std::iota(r.begin(), r.end(), 0);
            return r;
        }();

        return ranks;
    }

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
     * @brief Returns an approximation of how many bytes were sent.
     *      E.g., it only counts reduce once, so this is an underapproximation.
     * @return The number of bytes sent
     */
    static uint64_t get_number_bytes_sent() noexcept {
        return bytes_sent.load(std::memory_order::relaxed);
    }

    /**
     * @brief Returns an approximation of how many bytes were received.
     *      E.g., it only counts reduce on the root rank, so this is an underapproximation.
     * @return The number of bytes received
     */
    static uint64_t get_number_bytes_received() noexcept {
        return bytes_received.load(std::memory_order::relaxed);
    }

    /**
     * @brief Returns the number of bytes accessed remotely in windows
     * @return The number of bytes remotely accessed
     */
    static uint64_t get_number_bytes_remote_accessed() noexcept {
        return bytes_remote.load(std::memory_order::relaxed);
    }

    /**
     * @brief Finalizes the local MPI implementation.
     * @exception Throws a RelearnException if an MPI error occurs
     */
    static void finalize();
};

#endif
