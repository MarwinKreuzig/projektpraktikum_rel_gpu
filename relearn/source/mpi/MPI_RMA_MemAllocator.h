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

#if MPI_FOUND

#include "../Config.h"
#include "MPITypes.h"

#include <cstdint>
#include <queue>
#include <vector>

class OctreeNode;

/**
 * This class provides a static interface for allocating and deallocating objects of type OctreeNode
 * that are placed within an MPI memory window, i.e., they can be accessed by other MPI processes.
 * The first call must be to MPI_RMA_MemAllocator::init(...) and the last call must be to MPI_RMA_MemAllocator::finalize().
 */
class MPI_RMA_MemAllocator {

    class HolderOctreeNode {
        std::queue<OctreeNode*> available{};
        std::vector<OctreeNode*> non_available{};
        OctreeNode* base_ptr{ nullptr };

        size_t free{ Constants::uninitialized };
        size_t total{ Constants::uninitialized };

    public:
        // NOLINTNEXTLINE
        HolderOctreeNode() { /* This is not defaulted because of compiler errors */
        }

        HolderOctreeNode(OctreeNode* ptr, size_t length);

        [[nodiscard]] OctreeNode* get_available();

        void make_available(OctreeNode* ptr);

        [[nodiscard]] size_t get_size() const noexcept;

        [[nodiscard]] size_t get_num_available() const noexcept;
    };

    MPI_RMA_MemAllocator() = default;

public:
    /**
     * @brief Initializes the memory window to the requested size and exchanges the pointers across all MPi processes
     * @param size_requested The size of the memory window in bytes
     * @param num_local_trees The number of branch nodes across all MPI processes
     * @exception Throws a RelearnException if an MPI operation fails
     */
    static void init(size_t size_requested, size_t num_branch_nodes);

    /**
     * @brief Frees the memory window and deallocates all shared memory.
     * @exception Throws a RelearnException if an MPI operation fails
     */
    static void finalize();

    /**
     * @brief Returns a pointer to a fresh OctreeNode in the memory window.
     * @expection Throws a RelearnException if not enough memory is available.
     * @return A valid pointer to an OctreeNode
     */
    [[nodiscard]] static OctreeNode* new_octree_node();

    /**
     * @brief Deletes the object pointed to. Internally calls OctreeNode::reset().
     *      The pointer is invalidated.
     * @param ptr The pointer to object that shall be deleted
     */
    static void delete_octree_node(OctreeNode* ptr);

    /**
     * @brief Returns the base addresses of the memory windows of all memory windows.
     * @return The base addresses of the memory windows. The base address for MPI rank i
     *      is found at <return>[i]
     */
    [[nodiscard]] static const std::vector<int64_t>& get_base_pointers() noexcept;

    /**
     * @brief Returns the roots for all branch nodes
     * @return The roots for all branch nodes
     */
    [[nodiscard]] static OctreeNode* get_branch_nodes();
    
    /**
     * @brief Returns the number of available objects in the memory window.
     * @return The number of available objects in the memory window
     */
    [[nodiscard]] static size_t get_num_avail_objects() noexcept;

    //NOLINTNEXTLINE
    static inline MPI_Win mpi_window{ 0 }; // RMA window object

private:
    static inline size_t size_requested{ Constants::uninitialized }; // Bytes requested for the allocator
    static inline size_t max_size{ Constants::uninitialized }; // Size in Bytes of MPI-allocated memory
    static inline size_t max_num_objects{ Constants::uninitialized }; // Max number objects that are available

    static inline OctreeNode* root_nodes_for_local_trees{ nullptr };

    static inline OctreeNode* base_ptr{ nullptr }; // Start address of MPI-allocated memory
    //NOLINTNEXTLINE
    static inline HolderOctreeNode holder_base_ptr{};

    static inline size_t num_ranks{ Constants::uninitialized }; // Number of ranks in MPI_COMM_WORLD
    static inline int displ_unit{ -1 }; // RMA window displacement unit
    static inline std::vector<int64_t> base_pointers{}; // RMA window base pointers of all procs
};

#endif
