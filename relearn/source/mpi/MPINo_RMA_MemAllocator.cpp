/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MPINo_RMA_MemAllocator.h"

#if !MPI_FOUND

#include "../algorithm/BarnesHutCell.h"
#include "../io/LogFiles.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>


template class MPINo_RMA_MemAllocator<BarnesHutCell>;

MPINo_RMA_MemAllocator::HolderOctreeNode MPINo_RMA_MemAllocator::holder_base_ptr{};

void MPINo_RMA_MemAllocator::init(size_t size_requested, size_t num_local_trees) {
    MPINo_RMA_MemAllocator::size_requested = size_requested;
    max_num_objects = size_requested / sizeof(OctreeNode<BarnesHutCell>);
    max_size = size_requested;

    data.resize(max_num_objects);
    base_ptr = data.data();

    // create_rma_window();
    base_pointers = reinterpret_cast<int64_t>(base_ptr);

    holder_base_ptr = HolderOctreeNode(base_ptr, max_num_objects);

    root_nodes_for_local_trees.resize(num_local_trees);

    LogFiles::print_message_rank(0, "MPI RMA MemAllocator: max_num_objects: {}  sizeof(OctreeNode<BarnesHutCell>): {}", max_num_objects, sizeof(OctreeNode<BarnesHutCell>));
}

#endif
