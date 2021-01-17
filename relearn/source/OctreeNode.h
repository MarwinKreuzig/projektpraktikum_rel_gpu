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
#include "Commons.h"

#include <array>
#include <cstddef>

struct OctreeNode {
    void print() const;

    Cell cell;
    std::array<OctreeNode*, Constants::number_oct> children { nullptr };
    bool is_parent { false };

    size_t rank { 0 }; // MPI rank who owns this octree node
    size_t level { 0 }; // Level in the tree [0 (= root) ... depth of tree]
};
