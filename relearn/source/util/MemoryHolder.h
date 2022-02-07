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

#include "RelearnException.h"

#include <queue>
#include <vector>

template <typename T>
class OctreeNode;

/**
 * This class manages a portion of memory of a specified size and can hand out new objects
 * of the specified type as long as there is space available in the memory portion.
 * Hands out pointers via get_available(), which have to be reclaimed with make_available() or make_all_available() later on.
 *
 * In effect calls OctreeNode<AdditionalCellAttributes>::reset()
 *
 * @tparam AdditionalCellAttributes The template parameter of the objects
 */
template <typename AdditionalCellAttributes>
class MemoryHolder {
    // NOLINTNEXTLINE
    static inline std::queue<OctreeNode<AdditionalCellAttributes>*> available{};
    static inline std::vector<OctreeNode<AdditionalCellAttributes>*> non_available{};
    static inline OctreeNode<AdditionalCellAttributes>* base_ptr{ nullptr };

    static inline size_t total{ Constants::uninitialized };

public:
    /**
     * @brief Initializes the class to hold the specified pointer.
     *      Does not transfer ownership
     * @param ptr The pointer to the memory location in which objects should be created and deleted
     * @param length The number of objects that fit into the memory
     */
    static void init(OctreeNode<AdditionalCellAttributes>* ptr, const size_t length) {
        non_available.resize(length, nullptr);
        base_ptr = ptr;
        total = length;

        for (size_t counter = 0; counter < length; counter++) {
            // NOLINTNEXTLINE
            auto* current_value = ptr + counter;
            current_value->reset();
            available.push(current_value);
        }
    }

    /**
     * @brief Returns a non-null pointer to a new object, which has to be reclaimed later.
     *      Does not transfer ownership
     * @exception Throws a RelearnException if all objects are in use
     * @return A non-null pointer to a new object
     */
    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* get_available() {
        RelearnException::check(!available.empty(), "MemoryHolder::get_available: There are no free nodes.");

        // Get last available element and save it
        OctreeNode<AdditionalCellAttributes>* ptr = available.front();
        available.pop();
        const size_t dist = std::distance(base_ptr, ptr);

        RelearnException::check(dist < non_available.size(), "MemoryHolder::get_available: The distance was too large: {} vs {}.", dist, non_available.size());
        non_available[dist] = ptr;

        return ptr;
    }

    /**
     * @brief Returns the pointer to the available ones and destroys the object pointed to.
     * @param ptr The pointer that should be freed, not null, must have been retrieved via get_available()
     * @exception Throws a RelearnException if ptr is nullptr or ptr did not come from get_available()
     */
    static void make_available(OctreeNode<AdditionalCellAttributes>* ptr) {
        RelearnException::check(ptr != nullptr, "MemoryHolder::make_available: ptr was nullptr");

        const size_t dist = std::distance(base_ptr, ptr);

        RelearnException::check(dist < non_available.size(), "MemoryHolder::get_available: The distance was too large: {} vs {}.", dist, non_available.size());
        available.push(ptr);
        non_available[dist] = nullptr;

        ptr->reset();
    }

    /**
     * @brief Destroys all objects that were handed out via get_available. All pointers are invalidated.
     */
    static void make_all_available() noexcept {
        for (OctreeNode<AdditionalCellAttributes>*& ptr : non_available) {
            if (ptr == nullptr) {
                continue;
            }

            available.push(ptr);

            ptr->reset();
            ptr = nullptr;
        }
    }

    /**
     * @brief Returns the number of objects that fit into the memory portion
     * @return The number of objects that fit into the memory portion
     */
    [[nodiscard]] static size_t get_size() noexcept {
        return total;
    }

    /**
     * @brief Returns the number of currently available objects
     * @return The number of currently available objects
     */
    [[nodiscard]] static size_t get_number_available_objects() noexcept {
        return available.size();
    }
};
