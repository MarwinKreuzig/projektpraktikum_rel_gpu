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

#if MPI_FOUND

#include "../io/LogFiles.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <sstream>

void MPI_RMA_MemAllocator::init(size_t size_requested, size_t num_branch_nodes) {
    MPI_RMA_MemAllocator::size_requested = size_requested;

    // Number of objects "size_requested" Bytes correspond to
    max_num_objects = size_requested / sizeof(OctreeNode);
    max_size = max_num_objects * sizeof(OctreeNode);

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
    // NOLINTNEXTLINE
    const int error_code_2 = MPI_Win_create(base_ptr, max_size, displ_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &mpi_window);
    RelearnException::check(error_code_2 == 0, "Error in MPI_RMA_MemAllocator::init()");

    // Vector must have space for one pointer from each rank
    base_pointers.resize(num_ranks);

    // NOLINTNEXTLINE
    const int error_code_3 = MPI_Allgather(&base_ptr, 1, MPI_AINT, base_pointers.data(), 1, MPI_AINT, MPI_COMM_WORLD);
    RelearnException::check(error_code_3 == 0, "Error in MPI_RMA_MemAllocator::init()");

    holder_base_ptr = HolderOctreeNode(base_ptr, max_num_objects);

    const auto requested_size = num_branch_nodes * sizeof(OctreeNode);

    // NOLINTNEXTLINE
    if (MPI_SUCCESS != MPI_Alloc_mem(static_cast<int>(requested_size), MPI_INFO_NULL, &root_nodes_for_local_trees)) {
        RelearnException::fail("MPI_Alloc_mem failed for local trees");
    }

    LogFiles::print_message_rank(0, "MPI RMA MemAllocator: max_num_objects: {}  sizeof(OctreeNode): {}", max_num_objects, sizeof(OctreeNode));
}

void MPI_RMA_MemAllocator::finalize() {
    const int error_code_1 = MPI_Win_free(&mpi_window);
    RelearnException::check(error_code_1 == 0, "Error in MPI_RMA_MemAllocator::finalize()");
    const int error_code_2 = MPI_Free_mem(base_ptr);
    RelearnException::check(error_code_2 == 0, "Error in MPI_RMA_MemAllocator::finalize()");
    const int error_code_3 = MPI_Free_mem(root_nodes_for_local_trees);
    RelearnException::check(error_code_3 == 0, "Error in MPI_RMA_MemAllocator::finalize()");
}

[[nodiscard]] OctreeNode* MPI_RMA_MemAllocator::new_octree_node() {
    return holder_base_ptr.get_available();
}

void MPI_RMA_MemAllocator::delete_octree_node(OctreeNode* ptr) {
    holder_base_ptr.make_available(ptr);
}

[[nodiscard]] const std::vector<int64_t>& MPI_RMA_MemAllocator::get_base_pointers() noexcept {
    return base_pointers;
}

[[nodiscard]] OctreeNode* MPI_RMA_MemAllocator::get_branch_nodes() {
    return root_nodes_for_local_trees;
}

[[nodiscard]] size_t MPI_RMA_MemAllocator::get_num_avail_objects() noexcept {
    return holder_base_ptr.get_num_available();
}

MPI_RMA_MemAllocator::HolderOctreeNode::HolderOctreeNode(OctreeNode* ptr, size_t length)
    : non_available(length, nullptr)
    , base_ptr(ptr)
    , free(length)
    , total(length) {
    for (size_t counter = 0; counter < length; counter++) {
        // NOLINTNEXTLINE
        available.push(ptr + counter);
    }
}

[[nodiscard]] OctreeNode* MPI_RMA_MemAllocator::HolderOctreeNode::get_available() {
    RelearnException::check(!available.empty(), "In MPI_RMA_MemAllocator::HolderOctreeNode::get_available, there are no free nodes.");

    // Get last available element and save it
    OctreeNode* ptr = available.front();
    available.pop();
    const size_t dist = std::distance(base_ptr, ptr);

    non_available[dist] = ptr;

    return ptr;
}

void MPI_RMA_MemAllocator::HolderOctreeNode::make_available(OctreeNode* ptr) {
    const size_t dist = std::distance(base_ptr, ptr);

    available.push(ptr);
    non_available[dist] = nullptr;

    ptr->reset();
}

[[nodiscard]] size_t MPI_RMA_MemAllocator::HolderOctreeNode::get_size() const noexcept {
    return total;
}

[[nodiscard]] size_t MPI_RMA_MemAllocator::HolderOctreeNode::get_num_available() const noexcept {
    return available.size();
}

#endif
