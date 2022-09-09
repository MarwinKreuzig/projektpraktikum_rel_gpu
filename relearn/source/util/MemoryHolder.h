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

#include <span>
#include <unordered_map>

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
    static inline std::span<OctreeNode<AdditionalCellAttributes>> memory_holder{ };
    static inline size_t current_filling{ 0 };

    static inline std::unordered_map<OctreeNode<AdditionalCellAttributes>*, size_t> parent_to_offset{};

public:
    /**
     * @brief Initializes the class to hold the specified pointer.
     *      Does not transfer ownership
     * @param ptr The pointer to the memory location in which objects should be created and deleted
     * @param length The number of objects that fit into the memory
     */
    static void init(std::span<OctreeNode<AdditionalCellAttributes>> memory) noexcept {
        memory_holder = memory;
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
        RelearnException::check(offset + Constants::number_oct <= memory_holder.size(),
            "MemoryHolder::get_available: The offset is too large: {} + {} vs {}", offset, Constants::number_oct, memory_holder.size());

        return &memory_holder[offset + octant];
    }

    /**
     * @brief Destroys all objects that were handed out via get_available. All pointers are invalidated.
     */
    static void make_all_available() noexcept {
        for (size_t i = 0; i < memory_holder.size(); i++) {
            memory_holder[i].reset();
        }

        current_filling = 0;
        parent_to_offset.clear();
    }

    /**
     * @brief Returns the number of objects that fit into the memory portion
     * @return The number of objects that fit into the memory portion
     */
    [[nodiscard]] static size_t get_size() noexcept {
        return memory_holder.size();
    }

    /**
     * @brief Returns the offset of the specified node's children with respect to the base pointer
     * @param parent_node The node for whose children we want to have the offset
     * @exception Throws a RelearnException if parent_node does not have an associated children array
     * @return The offset of node wrt. the base pointer
     */
    [[nodiscard]] static std::uint64_t get_offset(OctreeNode<AdditionalCellAttributes>* parent_node) {
        const auto iterator = parent_to_offset.find(parent_node);

        RelearnException::check(iterator != parent_to_offset.end(), "MemoryHolder::get_offset: parent_node didn't have an offset.");

        const auto offset = iterator->second;
        return offset * sizeof(OctreeNode<AdditionalCellAttributes>);
    }

    /**
     * @brief Returns the OctreeNode at the specified offset
     * @param offset The offset at which the OctreeNode shall be returned
     * @exception Throws a RelaernException if offset is larger or equal to the total number of objects or to the current filling
     * @return The OctreeNode with the specified offset
     */
    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* get_node_from_offset(std::uint64_t offset) {
        RelearnException::check(offset < memory_holder.size(), "MemoryHolder::get_node_from_offset(): offset ({}) is too large: ({}).", offset, memory_holder.size());
        RelearnException::check(offset < current_filling, "MemoryHolder::get_node_from_offset(): offset ({}) is too large: ({}).", offset, current_filling);
        return &memory_holder[offset];
    }
};
