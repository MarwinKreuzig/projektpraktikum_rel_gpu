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
#include "MPINo_RMA_MemAllocator.h"

template <typename T>
using MPI_RMA_MemAllocator = MPINo_RMA_MemAllocator<T>;
#else // #if MPI_FOUND
#pragma message("Using MPI_RMA_MemAllocator")

#include "../util/RelearnException.h"

#include <cstdint>
#include <queue>
#include <vector>

template <typename T>
class OctreeNode;

/**
 * This class provides a static interface for allocating and deallocating objects of type OctreeNode
 * that are placed within an MPI memory window, i.e., they can be accessed by other MPI processes.
 * The first call must be to MPI_RMA_MemAllocator::init(...) and the last call must be to MPI_RMA_MemAllocator::finalize().
 */
template <typename AdditionalCellAttributes>
class MPI_RMA_MemAllocator {
    MPI_RMA_MemAllocator() = default;
public:
    /**
     * @brief Initializes the memory window to the requested size and exchanges the pointers across all MPi processes
     * @param size_requested The size of the memory window in bytes
     * @exception Throws a RelearnException if an MPI operation fails
     */
    static void init(size_t size_requested);

    /**
     * @brief Frees the memory window and deallocates all shared memory.
     * @exception Throws a RelearnException if an MPI operation fails
     */
    static void finalize();

    /**
     * @brief Returns the base addresses of the memory windows of all memory windows.
     * @return The base addresses of the memory windows. The base address for MPI rank i
     *      is found at <return>[i]
     */
    [[nodiscard]] static const std::vector<int64_t>& get_base_pointers() noexcept {
        return base_pointers;
    }

    //NOLINTNEXTLINE
    static inline void* mpi_window{ nullptr }; // RMA window object

private:
    static inline size_t size_requested{ Constants::uninitialized }; // Bytes requested for the allocator
    static inline size_t max_size{ Constants::uninitialized }; // Size in Bytes of MPI-allocated memory
    static inline size_t max_num_objects{ Constants::uninitialized }; // Max number objects that are available

    static inline OctreeNode<AdditionalCellAttributes>* base_ptr{ nullptr }; // Start address of MPI-allocated memory

    static inline size_t num_ranks{ Constants::uninitialized }; // Number of ranks in MPI_COMM_WORLD
    static inline int displ_unit{ -1 }; // RMA window displacement unit
    static inline std::vector<int64_t> base_pointers{}; // RMA window base pointers of all procs
};

#endif
