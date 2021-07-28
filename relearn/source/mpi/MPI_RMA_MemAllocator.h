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
#include "../util/RelearnException.h"

#include <cstdint>
#include <queue>
#include <vector>

template <typename T>
class OctreeNode;

/**
 * This class provides a static interface for allocating and deallocating objects of type OctreeNode
 * that are placed within an MPI memory window, i.e., they can be accessed by other MPI processes.
 * The first call must be to MPI_RMA_MemAllocator::init(...) and the last call must be to MPI_RMA_MemAllocator::finalize().
 */
template <typename AdditionalCellAttributes>
class MPI_RMA_MemAllocator {
    friend class OctreeNode<AdditionalCellAttributes>;

    class HolderOctreeNode {
        std::queue<OctreeNode<AdditionalCellAttributes>*> available{};
        std::vector<OctreeNode<AdditionalCellAttributes>*> non_available{};
        OctreeNode<AdditionalCellAttributes>* base_ptr{ nullptr };

        size_t free{ Constants::uninitialized };
        size_t total{ Constants::uninitialized };

    public:
        // NOLINTNEXTLINE
        HolderOctreeNode() { /* This is not defaulted because of compiler errors */
        }

        HolderOctreeNode(OctreeNode<AdditionalCellAttributes>* ptr, size_t length)
            : non_available(length, nullptr)
            , base_ptr(ptr)
            , free(length)
            , total(length) {
            for (size_t counter = 0; counter < length; counter++) {
                // NOLINTNEXTLINE
                available.push(ptr + counter);
            }
        }

        [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_available() {
            RelearnException::check(!available.empty(), "In MPI_RMA_MemAllocator::HolderOctreeNode::get_available, there are no free nodes.");

            // Get last available element and save it
            OctreeNode<AdditionalCellAttributes>* ptr = available.front();
            available.pop();
            const size_t dist = std::distance(base_ptr, ptr);

            non_available[dist] = ptr;

            return ptr;
        }

        void make_available(OctreeNode<AdditionalCellAttributes>* ptr) {
            const size_t dist = std::distance(base_ptr, ptr);

            available.push(ptr);
            non_available[dist] = nullptr;

            ptr->reset();
        }

        void make_all_available() noexcept {
            for (auto& ptr : non_available) {
                if (ptr == nullptr) {
                    continue;
                }

                available.push(ptr);

                ptr->reset();
                ptr = nullptr;
            }
        }

        [[nodiscard]] size_t get_size() const noexcept {
            return total;
        }

        [[nodiscard]] size_t get_num_available() const noexcept {
            return available.size();
        }
    };

    MPI_RMA_MemAllocator() = default;

    /**
     * @brief Returns a pointer to a fresh OctreeNode in the memory window.
     * @expection Throws a RelearnException if not enough memory is available.
     * @return A valid pointer to an OctreeNode
     */
    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* new_octree_node() {
        return holder_base_ptr.get_available();
    }

    /**
     * @brief Deletes the object pointed to. Internally calls OctreeNode::reset().
     *      The pointer is invalidated.
     * @param ptr The pointer to object that shall be deleted
     */
    static void delete_octree_node(OctreeNode<AdditionalCellAttributes>* ptr) {
        holder_base_ptr.make_available(ptr);
    }

public:
    /**
     * @brief Initializes the memory window to the requested size and exchanges the pointers across all MPi processes
     * @param size_requested The size of the memory window in bytes
     * @exception Throws a RelearnException if an MPI operation fails
     */
    static void init(size_t size_requested);

    /**
     * @brief Frees the memory window and deallocates all shared memory.
     * @exception Throws a RelearnException if an MPI operation fails
     */
    static void finalize();

    /**
     * @brief Returns the base addresses of the memory windows of all memory windows.
     * @return The base addresses of the memory windows. The base address for MPI rank i
     *      is found at <return>[i]
     */
    [[nodiscard]] static const std::vector<int64_t>& get_base_pointers() noexcept {
        return base_pointers;
    }

    /**
     * @brief Returns the number of available objects in the memory window.
     * @return The number of available objects in the memory window
     */
    [[nodiscard]] static size_t get_num_avail_objects() noexcept {
        return holder_base_ptr.get_num_available();
    }

    /**
     * @brief Deletes all created nodes. Internally calls OctreeNode::reset().
     *      Invalidates all pointers
     */
    static void make_all_available() noexcept {
        holder_base_ptr.make_all_available();
    }

    //NOLINTNEXTLINE
    static inline void* mpi_window{ nullptr }; // RMA window object

private:
    static inline size_t size_requested{ Constants::uninitialized }; // Bytes requested for the allocator
    static inline size_t max_size{ Constants::uninitialized }; // Size in Bytes of MPI-allocated memory
    static inline size_t max_num_objects{ Constants::uninitialized }; // Max number objects that are available

    static inline OctreeNode<AdditionalCellAttributes>* base_ptr{ nullptr }; // Start address of MPI-allocated memory
    //NOLINTNEXTLINE
    static inline HolderOctreeNode holder_base_ptr{};

    static inline size_t num_ranks{ Constants::uninitialized }; // Number of ranks in MPI_COMM_WORLD
    static inline int displ_unit{ -1 }; // RMA window displacement unit
    static inline std::vector<int64_t> base_pointers{}; // RMA window base pointers of all procs
};

#endif
