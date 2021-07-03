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

#if !MPI_FOUND

#include "../Config.h"
#include "MPITypes.h"
#include "../structure/Octree.h"

#include <vector>
#include <queue>
#include <cstdint>

class MPINo_RMA_MemAllocator {

    class HolderOctreeNode {
        std::queue<OctreeNode*> available{};
        std::vector<OctreeNode*> non_available{};
        OctreeNode* base_ptr{ nullptr };

        size_t total{ Constants::uninitialized };

    public:
        HolderOctreeNode() = default;

        HolderOctreeNode(OctreeNode* ptr, size_t length);

        [[nodiscard]] OctreeNode* get_available();

        void make_available(OctreeNode* ptr);

        [[nodiscard]] size_t get_size() const noexcept;

        [[nodiscard]] size_t get_num_available() const noexcept;
    };

    MPINo_RMA_MemAllocator() = default;

public:
    static void init(size_t size_requested, size_t num_local_trees);

    static void finalize();

    [[nodiscard]] static OctreeNode* new_octree_node();

    static void delete_octree_node(OctreeNode* ptr);

    [[nodiscard]] static int64_t get_base_pointers() noexcept;

    [[nodiscard]] static OctreeNode* get_branch_nodes();

    [[nodiscard]] static size_t get_num_avail_objects() noexcept;

    //NOLINTNEXTLINE
    // static inline MPI_Win mpi_window{ 0 }; // RMA window object

private:
    static inline size_t size_requested{ Constants::uninitialized }; // Bytes requested for the allocator
    static inline size_t max_size{ Constants::uninitialized }; // Size in Bytes of MPI-allocated memory
    static inline size_t max_num_objects{ Constants::uninitialized }; // Max number objects that are available

    static inline std::vector<OctreeNode> root_nodes_for_local_trees{};

    static inline OctreeNode* base_ptr{ nullptr }; // Start address of MPI-allocated memory
    static inline std::vector<OctreeNode> data{};
    static HolderOctreeNode holder_base_ptr;

    static inline size_t num_ranks{ 1 }; // Number of ranks in MPI_COMM_WORLD
    static inline int displ_unit{ -1 }; // RMA window displacement unit
    static inline int64_t base_pointers{}; // RMA window base pointers of all procs
};

#endif
