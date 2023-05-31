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
#include "util/RelearnException.h"

#include <memory>
#include <span>
#include <unordered_map>

#include <range/v3/algorithm/for_each.hpp>

template <typename T>
class OctreeNode;

/**
 * This class manages a portion of memory and can hand out OctreeNodes as long as there is space left.
 * Hands out pointers via get_available(), which have to be reclaimed with make_all_available() later on.
 * get_available() makes sure that memory is really handled in portions, and all children are next to each other.
 *
 * In effect calls OctreeNode<AdditionalCellAttributes>::reset()
 *
 * @tparam AdditionalCellAttributes The template parameter of the objects
 */
template <typename AdditionalCellAttributes>
class MemoryHolder {
public:
    /**
     * @brief Initializes the class to hold the specified span of memory.
     * @param memory The span of memory, the elements are constructed to ensure memory correctness
     */
    static void init(const std::span<OctreeNode<AdditionalCellAttributes>> memory) noexcept {
        memory_holder = memory;
        current_filling = 0;
        parent_to_offset.clear();
        parent_to_offset.reserve(memory.size());
        offset_to_parent.clear();
        offset_to_parent.reserve(memory.size());
        std::ranges::uninitialized_default_construct(memory_holder);
    }

    /**
     * @brief Returns the currently held memory
     * @return The currently held memory
     */
    [[nodiscard]] static std::span<OctreeNode<AdditionalCellAttributes>> get_current_memory() noexcept {
        return memory_holder;
    }

    /**
     * @brief Returns the number of objects that fit into the memory portion
     * @return The number of objects that fit into the memory portion
     */
    [[nodiscard]] static typename std::span<OctreeNode<AdditionalCellAttributes>>::size_type get_size() noexcept {
        return memory_holder.size();
    }

    /**
     * @brief Returns the number of objects that are currently held
     * @return The number of objects that are currently held
     */
    [[nodiscard]] static std::uint64_t get_current_filling() noexcept {
        return current_filling;
    }

    /**
     * @brief Destroys all objects that were handed out via get_available. All pointers are invalidated.
     */
    static void make_all_available() noexcept {
        ranges::for_each(memory_holder, &OctreeNode<AdditionalCellAttributes>::reset);

        current_filling = 0;
        parent_to_offset.clear();
        offset_to_parent.clear();
    }

    /**
     * @brief Returns the pointer for the octant-th child of parent.
     *      Is deterministic if called repeatedly without calls to make_all_available inbetween.
     * @param parent The OctreeNode whose child the newly created node shall be
     * @param octant The octant of the newly created child
     * @exception Throws a RelearnException if parent == nullptr, octant >= Constants::number_oct, or if there is no more space left
     * @return Returns a pointer to the newly created child
     */
    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* get_available(OctreeNode<AdditionalCellAttributes>* const parent, const unsigned int octant) {
        RelearnException::check(parent != nullptr, "MemoryHolder::get_available: parent is nullptr");
        RelearnException::check(octant < Constants::number_oct, "MemoryHolder::get_available: octant is too large: {} vs {}", octant, Constants::number_oct);

        if (!parent_to_offset.contains(parent)) {
            parent_to_offset[parent] = current_filling;
            offset_to_parent[current_filling] = parent;
            current_filling += Constants::number_oct;
        }

        const auto offset = parent_to_offset[parent];
        RelearnException::check(offset + Constants::number_oct <= memory_holder.size(),
            "MemoryHolder::get_available: The offset is too large: {} + {} vs {}", offset, Constants::number_oct, memory_holder.size());

        return &memory_holder[offset + octant];
    }

    /**
     * @brief Returns the offset of the specified node's children with respect to the base pointer in bytes
     * @param parent_node The node for whose children we want to have the offset
     * @exception Throws a RelearnException if parent_node does not have an associated children array
     * @return The offset of node wrt. the base pointer
     */
    [[nodiscard]] static std::uint64_t get_offset_from_parent(OctreeNode<AdditionalCellAttributes>* const parent_node) {
        const auto iterator = parent_to_offset.find(parent_node);

        RelearnException::check(iterator != parent_to_offset.end(), "MemoryHolder::get_offset_from_parent: parent_node {} does not have an offset.", (void*)parent_node);

        const auto offset = iterator->second;
        return offset * sizeof(OctreeNode<AdditionalCellAttributes>);
    }

    /**
     * @brief Returns the parent node for a node specified by the offset
     * @param offset The offset, must be a multple of Constants::number_oct
     * @exception Throws a RelearnException if the offset is not saved for a parent or if offset % Constants::number_oct != 0
     * @return A pointer to the parent of the node stored at the offset
     */
    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* get_parent_from_offset(const std::uint64_t offset) {
        RelearnException::check(offset % Constants::number_oct == 0, "MemoryHolder::get_parent_from_offset: offset {} is not a multiple of {}.", offset, Constants::number_oct);
        const auto iterator = offset_to_parent.find(offset);

        RelearnException::check(iterator != offset_to_parent.end(), "MemoryHolder::get_parent_from_offset: offset {} does not have a parent node.", offset);

        const auto parent = iterator->second;
        return parent;
    }

    /**
     * @brief Returns the OctreeNode at the specified offset
     * @param offset The offset at which the OctreeNode shall be returned
     * @exception Throws a RelearnException if offset is larger or equal to the total number of objects or to the current filling
     * @return The OctreeNode with the specified offset
     */
    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* get_node_from_offset(const std::uint64_t offset) {
        RelearnException::check(offset < memory_holder.size(), "MemoryHolder::get_node_from_offset(): offset ({}) is too large. The total size is: {}.", offset, memory_holder.size());
        RelearnException::check(offset < current_filling, "MemoryHolder::get_node_from_offset(): offset ({}) is too large. I only contain: {} elements.", offset, current_filling);
        return &memory_holder[offset];
    }

    /**
     * Dump content of the memory holder to a file
     * @param file_path The file path where the content will be dumped
     */
    static void dump_to_file(const std::filesystem::path& file_path) {
        std::ofstream out_stream{ file_path };
        RelearnException::check(out_stream.good() && !out_stream.bad(), "Octree::print_to_file: Unable to open stream for {}", file_path.string());
        std::stringstream ss;
        for (auto offset = 0; offset < memory_holder.size(); offset++) {
            if (!memory_holder[offset].get_mpi_rank().is_initialized()) {
                continue;
            }
            const void* address = &memory_holder[offset];
            out_stream << address << " " << offset << " " << offset * sizeof(OctreeNode<AdditionalCellAttributes>) << " " << memory_holder[offset].to_string() << "\n";
        }
        out_stream << ss.rdbuf();
        out_stream.flush();
        out_stream.close();
    }

private:
    // NOLINTNEXTLINE
    static inline std::span<OctreeNode<AdditionalCellAttributes>> memory_holder{};
    static inline std::uint64_t current_filling{ 0 };

    static inline std::unordered_map<OctreeNode<AdditionalCellAttributes>*, std::uint64_t> parent_to_offset{};
    static inline std::unordered_map<std::uint64_t, OctreeNode<AdditionalCellAttributes>*> offset_to_parent{};
};
