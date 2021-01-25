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

#include "Cell.h"
#include "Config.h"

#include <array>
#include <cstddef>

class OctreeNode {
    friend class Octree;
    
    bool parent{ false };

    int rank{ -1 }; // MPI rank who owns this octree node
    size_t level{ 0 }; // Level in the tree [0 (= root) ... depth of tree]

public:
    Cell cell{};
    std::array<OctreeNode*, Constants::number_oct> children{ nullptr };

    [[nodiscard]] int get_rank() const noexcept {
        return rank;
    }

    [[nodiscard]] size_t get_level() const noexcept {
        return level;
    }

    [[nodiscard]] bool is_parent() const noexcept {
        return parent;
    }

    void print() const;

    void reset() {
        cell = Cell{};
        children = std::array<OctreeNode*, Constants::number_oct>{ nullptr };
        parent = false;
        rank = -1;
        level = 0;
    }
};
