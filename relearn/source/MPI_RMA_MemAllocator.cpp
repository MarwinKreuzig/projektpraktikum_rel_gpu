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

#include "LogMessages.h"
#include "OctreeNode.h"
#include "RelearnException.h"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <sstream>

MPI_Win MPI_RMA_MemAllocator::mpi_window{ 0 };

size_t MPI_RMA_MemAllocator::size_requested{ Constants::uninitialized }; // Bytes requested for the allocator
size_t MPI_RMA_MemAllocator::max_size{ Constants::uninitialized }; // Size in Bytes of MPI-allocated memory
size_t MPI_RMA_MemAllocator::max_num_objects{ Constants::uninitialized }; // Max number objects that are available

OctreeNode* MPI_RMA_MemAllocator::root_nodes_for_local_trees{ nullptr };

OctreeNode* MPI_RMA_MemAllocator::base_ptr{ nullptr }; // Start address of MPI-allocated memory
MPI_RMA_MemAllocator::HolderOctreeNode MPI_RMA_MemAllocator::holder_base_ptr;

size_t MPI_RMA_MemAllocator::num_ranks{ Constants::uninitialized }; // Number of ranks in MPI_COMM_WORLD
int MPI_RMA_MemAllocator::displ_unit{ -1 }; // RMA window displacement unit
std::vector<int64_t> MPI_RMA_MemAllocator::base_pointers; // RMA window base pointers of all procs

void MPI_RMA_MemAllocator::init(size_t size_requested) {
    MPI_RMA_MemAllocator::size_requested = size_requested;

    // Number of objects "size_requested" Bytes correspond to
    max_num_objects = size_requested / sizeof(OctreeNode);
    max_size = max_num_objects * sizeof(OctreeNode);

    // Store size of MPI_COMM_WORLD
    int my_num_ranks = -1;
    // NOLINTNEXTLINE
    MPI_Comm_size(MPI_COMM_WORLD, &my_num_ranks);
    num_ranks = static_cast<size_t>(my_num_ranks);

    // Allocate block of memory which is managed later on
    // NOLINTNEXTLINE
    if (MPI_SUCCESS != MPI_Alloc_mem(max_size, MPI_INFO_NULL, &base_ptr)) {
        RelearnException::fail("MPI_Alloc_mem failed");
    }

    create_rma_window();
    gather_rma_window_base_pointers();

    holder_base_ptr = HolderOctreeNode(base_ptr, max_num_objects);

    std::stringstream sstring;
    sstring << "MPI RMA MemAllocator: max_num_objects: " << max_num_objects << "  sizeof(OctreeNode): " << sizeof(OctreeNode);
    LogMessages::print_message_rank(sstring.str().c_str(), 0);
    sstring.str("");
}

void MPI_RMA_MemAllocator::deallocate_rma_mem() {
    MPI_Free_mem(base_ptr);
    MPI_Free_mem(root_nodes_for_local_trees);
}

void MPI_RMA_MemAllocator::free_rma_window() {
    MPI_Win_free(&mpi_window);
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

[[nodiscard]] OctreeNode* MPI_RMA_MemAllocator::get_root_nodes_for_local_trees(size_t num_local_trees) {
    const size_t requested_size = num_local_trees * sizeof(OctreeNode);
    const auto requested_size_conv = static_cast<int>(requested_size);

    // NOLINTNEXTLINE
    if (MPI_SUCCESS != MPI_Alloc_mem(requested_size_conv, MPI_INFO_NULL, &root_nodes_for_local_trees)) {
        RelearnException::fail("MPI_Alloc_mem failed for local trees");
    }

    return root_nodes_for_local_trees;
}

[[nodiscard]] size_t MPI_RMA_MemAllocator::get_min_num_avail_objects() noexcept {
    return holder_base_ptr.get_num_available();
}

void MPI_RMA_MemAllocator::gather_rma_window_base_pointers() {
    // Vector must have space for one pointer from each rank
    base_pointers.resize(num_ranks);

    // NOLINTNEXTLINE
    MPI_Allgather(&base_ptr, 1, MPI_AINT, base_pointers.data(), 1, MPI_AINT, MPI_COMM_WORLD);
}

void MPI_RMA_MemAllocator::create_rma_window() noexcept {
    // Set window's displacement unit
    displ_unit = 1;
    MPI_Win_create(base_ptr, max_size, displ_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &mpi_window);
}

[[nodiscard]] size_t MPI_RMA_MemAllocator::HolderOctreeNode::calculate_distance(OctreeNode* ptr) const noexcept {
    const auto dist = size_t(ptr - base_ptr);
    return dist;
}

MPI_RMA_MemAllocator::HolderOctreeNode::HolderOctreeNode(OctreeNode* ptr, size_t length)
    : available(length)
    , non_available(length, nullptr)
    , base_ptr(ptr)
    , free(length)
    , total(length) {

    for (size_t counter = 0; counter < length; counter++) {
        available[counter] = ptr + counter;
    }
}

[[nodiscard]] OctreeNode* MPI_RMA_MemAllocator::HolderOctreeNode::get_available() {
    // Get last available element and save it
    auto last = available.end() - 1;
    OctreeNode* ptr = *last;
    const size_t dist = calculate_distance(ptr);

    available.erase(last);
    non_available[dist] = ptr;

    return ptr;
}

void MPI_RMA_MemAllocator::HolderOctreeNode::make_available(OctreeNode* ptr) {
    const size_t dist = calculate_distance(ptr);

    available.push_back(ptr);
    non_available[dist] = nullptr;

    ptr->reset();
}

[[nodiscard]] size_t MPI_RMA_MemAllocator::HolderOctreeNode::get_size() const noexcept {
    return total;
}

[[nodiscard]] size_t MPI_RMA_MemAllocator::HolderOctreeNode::get_num_available() const noexcept {
    return free;
}
