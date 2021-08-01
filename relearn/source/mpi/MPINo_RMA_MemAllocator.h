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

#pragma message("Using MPINo_RMA_MemAllocator")

#include <cstdint>
#include <queue>
#include <vector>

template <typename T>
class OctreeNode;

template <typename AdditionalCellAttributes>
class MPINo_RMA_MemAllocator {
    MPINo_RMA_MemAllocator() = default;
    
public:
    static void init(size_t size_requested);

    static void finalize();

    [[nodiscard]] static int64_t get_base_pointers() noexcept {
        return base_pointers;
    }

    //NOLINTNEXTLINE
    // static inline MPI_Win mpi_window{ 0 }; // RMA window object

private:
    static inline size_t size_requested{ Constants::uninitialized }; // Bytes requested for the allocator
    static inline size_t max_size{ Constants::uninitialized }; // Size in Bytes of MPI-allocated memory
    static inline size_t max_num_objects{ Constants::uninitialized }; // Max number objects that are available

    static inline OctreeNode<AdditionalCellAttributes>* base_ptr{ nullptr }; // Start address of MPI-allocated memory

    static inline size_t num_ranks{ 1 }; // Number of ranks in MPI_COMM_WORLD
    static inline int displ_unit{ -1 }; // RMA window displacement unit
    static inline int64_t base_pointers{}; // RMA window base pointers of all procs
};

#endif

