/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MPI_RMA_MemAllocator.h"

#if RELEARN_MPI_FOUND

#include "../algorithm/BarnesHutCell.h"
#include "../io/LogFiles.h"
#include "../structure/OctreeNode.h"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <sstream>

template class MPI_RMA_MemAllocator<BarnesHutCell>;

template <typename AdditionalCellAttributes>
void MPI_RMA_MemAllocator<AdditionalCellAttributes>::init(size_t size_requested) {
    MPI_RMA_MemAllocator::size_requested = size_requested;

    // Number of objects "size_requested" Bytes correspond to
    max_num_objects = size_requested / sizeof(OctreeNode<AdditionalCellAttributes>);
    max_size = max_num_objects * sizeof(OctreeNode<AdditionalCellAttributes>);

    // Store size of MPI_COMM_WORLD
    int my_num_ranks = -1;
    // NOLINTNEXTLINE
    const int error_code_1 = MPI_Comm_size(MPI_COMM_WORLD, &my_num_ranks);
    RelearnException::check(error_code_1 == 0, "Error in MPI_RMA_MemAllocator::init()");

    num_ranks = static_cast<size_t>(my_num_ranks);

    // Allocate block of memory which is managed later on
    // NOLINTNEXTLINE
    if (MPI_SUCCESS != MPI_Alloc_mem(max_size, MPI_INFO_NULL, &base_ptr)) {
        RelearnException::fail("MPI_Alloc_mem failed");
    }

    // Set window's displacement unit
    displ_unit = 1;
    mpi_window = new MPI_Win;
    // NOLINTNEXTLINE
    const int error_code_2 = MPI_Win_create(base_ptr, max_size, displ_unit, MPI_INFO_NULL, MPI_COMM_WORLD, (MPI_Win*)mpi_window);
    RelearnException::check(error_code_2 == 0, "Error in MPI_RMA_MemAllocator::init()");

    // Vector must have space for one pointer from each rank
    base_pointers.resize(num_ranks);

    // NOLINTNEXTLINE
    const int error_code_3 = MPI_Allgather(&base_ptr, 1, MPI_AINT, base_pointers.data(), 1, MPI_AINT, MPI_COMM_WORLD);
    RelearnException::check(error_code_3 == 0, "Error in MPI_RMA_MemAllocator::init()");

    holder_base_ptr = HolderOctreeNode(base_ptr, max_num_objects);

    LogFiles::print_message_rank(0, "MPI RMA MemAllocator: max_num_objects: {}  sizeof(OctreeNode): {}", max_num_objects, sizeof(OctreeNode<AdditionalCellAttributes>));
}

template <typename AdditionalCellAttributes>
void MPI_RMA_MemAllocator<AdditionalCellAttributes>::finalize() {
    const int error_code_1 = MPI_Win_free((MPI_Win*)mpi_window);
    RelearnException::check(error_code_1 == 0, "Error in MPI_RMA_MemAllocator::finalize()");
    delete mpi_window;
    const int error_code_2 = MPI_Free_mem(base_ptr);
    RelearnException::check(error_code_2 == 0, "Error in MPI_RMA_MemAllocator::finalize()");
}

#endif
