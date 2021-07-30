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
class HolderOctreeNode {
    std::queue<OctreeNode<AdditionalCellAttributes>*> available{};
    std::vector<OctreeNode<AdditionalCellAttributes>*> non_available{};
    OctreeNode<AdditionalCellAttributes>* base_ptr{ nullptr };

    size_t total{ Constants::uninitialized };

public:
    HolderOctreeNode() = default;

    HolderOctreeNode(OctreeNode<AdditionalCellAttributes>* ptr, size_t length)
        : non_available(length, nullptr)
        , base_ptr(ptr)
        , total(length) {
        for (size_t counter = 0; counter < length; counter++) {
            available.push(ptr + counter);
        }
    }

    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_available() {
        // Get last available element and save it
        OctreeNode<AdditionalCellAttributes>* ptr = available.front();
        available.pop();

        const size_t dist = std::distance(base_ptr, ptr);
        non_available[dist] = ptr;

        return ptr;
    }

    void make_available(OctreeNode<AdditionalCellAttributes>* ptr);

    void make_all_available() noexcept;

    [[nodiscard]] size_t get_size() const noexcept {
        return total;
    }

    [[nodiscard]] size_t get_num_available() const noexcept {
        return available.size();
    }
};

template <typename AdditionalCellAttributes>
class MPINo_RMA_MemAllocator {
    friend class OctreeNode<AdditionalCellAttributes>;

    MPINo_RMA_MemAllocator() = default;

    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* new_octree_node() {
        return holder_base_ptr.get_available();
    }   

    static void delete_octree_node(OctreeNode<AdditionalCellAttributes>* ptr) {
        holder_base_ptr.make_available(ptr);
    }
    
public:
    static void init(size_t size_requested);

    static void finalize();

    [[nodiscard]] static int64_t get_base_pointers() noexcept {
        return base_pointers;
    }

    [[nodiscard]] static size_t get_num_avail_objects() noexcept {
        return holder_base_ptr.get_num_available();
    }

    static void make_all_available() noexcept {
        return holder_base_ptr.make_all_available();
    }

    //NOLINTNEXTLINE
    // static inline MPI_Win mpi_window{ 0 }; // RMA window object

private:
    static inline size_t size_requested{ Constants::uninitialized }; // Bytes requested for the allocator
    static inline size_t max_size{ Constants::uninitialized }; // Size in Bytes of MPI-allocated memory
    static inline size_t max_num_objects{ Constants::uninitialized }; // Max number objects that are available

    static inline OctreeNode<AdditionalCellAttributes>* base_ptr{ nullptr }; // Start address of MPI-allocated memory
    static inline HolderOctreeNode<AdditionalCellAttributes> holder_base_ptr{};

    static inline size_t num_ranks{ 1 }; // Number of ranks in MPI_COMM_WORLD
    static inline int displ_unit{ -1 }; // RMA window displacement unit
    static inline int64_t base_pointers{}; // RMA window base pointers of all procs
};

#endif

