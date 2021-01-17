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

#include "Commons.h"

#include <mpi.h>

#include <vector>

class OctreeNode;

class MPI_RMA_MemAllocator {

    class HolderOctreeNode {
        std::vector<OctreeNode*> available;
        std::vector<OctreeNode*> non_available;
        OctreeNode* base_ptr{ nullptr };

        size_t free{ Constants::uninitialized };
        size_t total{ Constants::uninitialized };

        [[nodiscard]] size_t calculate_distance(OctreeNode* ptr) const noexcept;

    public:
        HolderOctreeNode() = default;

        HolderOctreeNode(OctreeNode* ptr, size_t length);

        [[nodiscard]] OctreeNode* get_available();

        void make_available(OctreeNode* ptr);

        [[nodiscard]] size_t get_size() const noexcept;

        [[nodiscard]] size_t get_num_available() const noexcept;
    };

public:
    MPI_RMA_MemAllocator() = default;

    MPI_RMA_MemAllocator(const MPI_RMA_MemAllocator& other) = delete;
    MPI_RMA_MemAllocator(MPI_RMA_MemAllocator&& other) = delete;

    MPI_RMA_MemAllocator& operator=(const MPI_RMA_MemAllocator& other) = delete;
    MPI_RMA_MemAllocator& operator=(MPI_RMA_MemAllocator&& other) = delete;

    ~MPI_RMA_MemAllocator() = default;

    void init(size_t size_requested);

    void deallocate_rma_mem();

    // Free the MPI RMA window
    // This call is collective over MPI_COMM_WORLD
    void free_rma_window();

    [[nodiscard]] OctreeNode* new_octree_node();

    void delete_octree_node(OctreeNode* ptr);

    [[nodiscard]] const std::vector<MPI_Aint>& get_base_pointers() const noexcept;

    [[nodiscard]] OctreeNode* get_root_nodes_for_local_trees(size_t num_local_trees);

    [[nodiscard]] size_t get_min_num_avail_objects() const noexcept;

    //NOLINTNEXTLINE
    MPI_Win mpi_window{ 0 }; // RMA window object

private:
    // Store RMA window base pointers of all ranks
    void gather_rma_window_base_pointers();

    // Create MPI RMA window with all the memory of the allocator
    // This call is collective over MPI_COMM_WORLD
    void create_rma_window() noexcept;

    size_t size_requested{ Constants::uninitialized }; // Bytes requested for the allocator
    size_t max_size{ Constants::uninitialized }; // Size in Bytes of MPI-allocated memory
    size_t max_num_objects{ Constants::uninitialized }; // Max number objects that are available

    OctreeNode* root_nodes_for_local_trees{ nullptr };

    OctreeNode* base_ptr{ nullptr }; // Start address of MPI-allocated memory
    HolderOctreeNode holder_base_ptr;

    size_t num_ranks{ Constants::uninitialized }; // Number of ranks in MPI_COMM_WORLD
    int displ_unit{ -1 }; // RMA window displacement unit
    std::vector<MPI_Aint> base_pointers; // RMA window base pointers of all procs
};
