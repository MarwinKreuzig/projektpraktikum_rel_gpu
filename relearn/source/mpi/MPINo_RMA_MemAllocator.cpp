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

#include "../io/LogFiles.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>

MPINo_RMA_MemAllocator::HolderOctreeNode MPINo_RMA_MemAllocator::holder_base_ptr{};

void MPINo_RMA_MemAllocator::init(size_t size_requested) {
    MPINo_RMA_MemAllocator::size_requested = size_requested;
    max_num_objects = size_requested / sizeof(OctreeNode);
    max_size = size_requested;

    data.resize(max_num_objects);
    base_ptr = data.data();

    // create_rma_window();
    gather_rma_window_base_pointers();

    holder_base_ptr = HolderOctreeNode(base_ptr, max_num_objects);

    std::stringstream sstring{};
    sstring << "MPI RMA MemAllocator: max_num_objects: " << max_num_objects << "  sizeof(OctreeNode): " << sizeof(OctreeNode);
    LogFiles::print_message_rank(sstring.str().c_str(), 0);
}

void MPINo_RMA_MemAllocator::deallocate_rma_mem() {
}

void MPINo_RMA_MemAllocator::free_rma_window() {
}

OctreeNode* MPINo_RMA_MemAllocator::new_octree_node() {
    return holder_base_ptr.get_available();
}

void MPINo_RMA_MemAllocator::delete_octree_node(OctreeNode* ptr) {
    holder_base_ptr.make_available(ptr);
}

int64_t MPINo_RMA_MemAllocator::get_base_pointers() noexcept {
    return base_pointers;
}

OctreeNode* MPINo_RMA_MemAllocator::get_root_nodes_for_local_trees(size_t num_local_trees) {
    root_nodes_for_local_trees.resize(num_local_trees);
    return root_nodes_for_local_trees.data();
}

size_t MPINo_RMA_MemAllocator::get_min_num_avail_objects() noexcept {
    return holder_base_ptr.get_num_available();
}

void MPINo_RMA_MemAllocator::gather_rma_window_base_pointers() {
    base_pointers = reinterpret_cast<int64_t>(base_ptr);
}

MPINo_RMA_MemAllocator::HolderOctreeNode::HolderOctreeNode(OctreeNode* ptr, size_t length)
    : non_available(length, nullptr)
    , base_ptr(ptr)
    , total(length) {
    for (size_t counter = 0; counter < length; counter++) {
        available.push(ptr + counter);
    }
}

OctreeNode* MPINo_RMA_MemAllocator::HolderOctreeNode::get_available() {
    // Get last available element and save it
    OctreeNode* ptr = available.front();
    available.pop();

    const size_t dist = std::distance(base_ptr, ptr);
    non_available[dist] = ptr;

    return ptr;
}

void MPINo_RMA_MemAllocator::HolderOctreeNode::make_available(OctreeNode* ptr) {
    const size_t dist = std::distance(base_ptr, ptr);

    available.push(ptr);
    non_available[dist] = nullptr;

    ptr->reset();
}

size_t MPINo_RMA_MemAllocator::HolderOctreeNode::get_size() const noexcept {
    return total;
}

size_t MPINo_RMA_MemAllocator::HolderOctreeNode::get_num_available() const noexcept {
    return available.size();
}

#endif
