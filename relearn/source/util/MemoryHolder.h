#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"
#include "RelearnException.h"

#include <unordered_map>
#include <queue>
#include <vector>

template <typename T>
class OctreeNode;

/**
 * This class manages a portion of memory of a specified size and can hand out OctreeNodes as long as there is space left.
 * Hands out pointers via get_available(), which have to be reclaimed with make_all_available() later on.
 * get_available() makes sure that memory is really handled in portions, and all children are next to each other.
 *
 * In effect calls OctreeNode<AdditionalCellAttributes>::reset()
 *
 * @tparam AdditionalCellAttributes The template parameter of the objects
 */
template <typename AdditionalCellAttributes>
class MemoryHolder {
    // NOLINTNEXTLINE
    static inline OctreeNode<AdditionalCellAttributes>* base_ptr{ nullptr };

    static inline size_t current_filling{ 0 };
    static inline size_t total{ Constants::uninitialized };

    static inline std::unordered_map<OctreeNode<AdditionalCellAttributes>*, size_t> parent_to_offset{};

public:
    /**
     * @brief Initializes the class to hold the specified pointer.
     *      Does not transfer ownership
     * @param ptr The pointer to the memory location in which objects should be created and deleted
     * @param length The number of objects that fit into the memory
     */
    static void init(OctreeNode<AdditionalCellAttributes>* ptr, const size_t length) {
        base_ptr = ptr;
        total = length;
    }

    /**
     * @brief Returns the pointer for the octant-th child of parent.
     *      Is deterministic if called repeatedly without calls to make_all_available inbetween.
     * @param parent The OctreeNode whose child the newly created node shall be
     * @param octant The octant of the newly created child
     * @exception Throws a RelearnException if parent == nullptr, octant >= Constants::number_oct, or if there is no more space left
     * @return Returns a pointer to the newly created child
     */
    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* get_available(OctreeNode<AdditionalCellAttributes>* parent, unsigned int octant) {
        RelearnException::check(parent != nullptr, "MemoryHolder::get_available: parent is nullptr");
        RelearnException::check(octant < Constants::number_oct, "MemoryHolder::get_available: octant is too large: {} vs {}", octant, Constants::number_oct);

        if (parent_to_offset.find(parent) == parent_to_offset.end()) {
            parent_to_offset[parent] = current_filling;
            current_filling += Constants::number_oct;
        }

        const auto offset = parent_to_offset[parent];
        RelearnException::check(offset + Constants::number_oct <= total, 
            "MemoryHolder::get_available: The offset is too large: {} + {} vs {}", offset, Constants::number_oct, total);

        return base_ptr + (offset + octant);
    }

    /**
     * @brief Destroys all objects that were handed out via get_available. All pointers are invalidated.
     */
    static void make_all_available() noexcept {
        for (size_t i = 0; i < total; i++) {
            base_ptr[i].reset();
        }

        current_filling = 0;
        parent_to_offset.clear();
    }

    /**
     * @brief Returns the number of objects that fit into the memory portion
     * @return The number of objects that fit into the memory portion
     */
    [[nodiscard]] static size_t get_size() noexcept {
        return total;
    }
};
