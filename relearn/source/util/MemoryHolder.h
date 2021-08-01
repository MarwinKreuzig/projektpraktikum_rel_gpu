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

#include <functional>
#include <queue>
#include <vector>

template <template <typename> typename OctreeNode, typename AdditionalCellAttributes>
class MemoryHolder {
    static inline std::queue<OctreeNode<AdditionalCellAttributes>*> available{};
    static inline std::vector<OctreeNode<AdditionalCellAttributes>*> non_available{};
    static inline OctreeNode<AdditionalCellAttributes>* base_ptr{ nullptr };

    static inline size_t total{ Constants::uninitialized };

public:
    static void init(OctreeNode<AdditionalCellAttributes>* ptr, size_t length) {
        non_available.resize(length, nullptr);
        base_ptr = ptr;
        total = length;

        for (size_t counter = 0; counter < length; counter++) {
            // NOLINTNEXTLINE
            available.push(ptr + counter);
        }
    }

    static [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_available() {
        RelearnException::check(!available.empty(), "In MemoryHolder::get_available, there are no free nodes.");

        // Get last available element and save it
        OctreeNode<AdditionalCellAttributes>* ptr = available.front();
        available.pop();
        const size_t dist = std::distance(base_ptr, ptr);

        non_available[dist] = ptr;

        return ptr;
    }

    static void make_available(OctreeNode<AdditionalCellAttributes>* ptr) {
        const size_t dist = std::distance(base_ptr, ptr);

        available.push(ptr);
        non_available[dist] = nullptr;

        ptr->reset();
    }

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

    [[nodiscard]] static size_t get_size() noexcept {
        return total;
    }

    [[nodiscard]] static size_t get_num_available() noexcept {
        return available.size();
    }
};
