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

#include "Config.h"
#include "MPITypes.h"

#include <cstdint>
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
        HolderOctreeNode() { }

        HolderOctreeNode(OctreeNode* ptr, size_t length);

        [[nodiscard]] OctreeNode* get_available();

        void make_available(OctreeNode* ptr);

        [[nodiscard]] size_t get_size() const noexcept;

        [[nodiscard]] size_t get_num_available() const noexcept;
    };

    MPI_RMA_MemAllocator() = default;

public:
    static void init(size_t size_requested);

    static void deallocate_rma_mem();

    // Free the MPI RMA window
    // This call is collective over MPI_COMM_WORLD
    static void free_rma_window();

    [[nodiscard]] static OctreeNode* new_octree_node();

    static void delete_octree_node(OctreeNode* ptr);

    [[nodiscard]] static const std::vector<int64_t>& get_base_pointers() noexcept;

    [[nodiscard]] static OctreeNode* get_root_nodes_for_local_trees(size_t num_local_trees);

    [[nodiscard]] static size_t get_min_num_avail_objects() noexcept;

    //NOLINTNEXTLINE
    static inline MPI_Win mpi_window{ 0 }; // RMA window object

private:
    // Store RMA window base pointers of all ranks
    static void gather_rma_window_base_pointers();

    // Create MPI RMA window with all the memory of the allocator
    // This call is collective over MPI_COMM_WORLD
    static void create_rma_window() noexcept;

    static inline size_t size_requested{ Constants::uninitialized }; // Bytes requested for the allocator
    static inline size_t max_size{ Constants::uninitialized }; // Size in Bytes of MPI-allocated memory
    static inline size_t max_num_objects{ Constants::uninitialized }; // Max number objects that are available

    static inline OctreeNode* root_nodes_for_local_trees{ nullptr };

    static inline OctreeNode* base_ptr{ nullptr }; // Start address of MPI-allocated memory
    static inline HolderOctreeNode holder_base_ptr{};

    static inline size_t num_ranks{ Constants::uninitialized }; // Number of ranks in MPI_COMM_WORLD
    static inline int displ_unit{ -1 }; // RMA window displacement unit
    static inline std::vector<int64_t> base_pointers{}; // RMA window base pointers of all procs
};
