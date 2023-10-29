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

#include <cstdint>
#include <string>
#include <unordered_map>

/**
 * This class captures the memory footprint of the simulation and all existing objects
 * (if used correctly).
 */
class MemoryFootprint {
public:
    /**
     * @brief Constructs the object
     * @param reserved_space The number of objects that will write their size into this
     */
    MemoryFootprint(const std::size_t reserved_space) {
        memory_description.reserve(reserved_space);
    }

    /**
     * @brief Records the size of an objects
     * @tparam T The key type, must be convertible to std::string
     * @param key The description
     * @param memory_size The size in bytes
     */
    template <typename T>
    void emplace(T&& key, const std::uint64_t memory_size) {
        memory_description.emplace(std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple(memory_size));
    }

    /**
     * @brief Returns the stored memory descriptions, i.e., a mapping from description to size
     * @return A constant reference to the description
     */
    const std::unordered_map<std::string, std::uint64_t>& get_descriptions() const noexcept {
        return memory_description;
    }

private:
    std::unordered_map<std::string, std::uint64_t> memory_description{};
};
