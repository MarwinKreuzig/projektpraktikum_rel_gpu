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

#include <array>
#include <cstddef>

class OctreeNode {
public:
	OctreeNode() /*noexcept*/;
	~OctreeNode() = default;

	OctreeNode(const OctreeNode& other) = default;
	OctreeNode(OctreeNode&& other) = default;

	OctreeNode& operator=(const OctreeNode& other) = default;
	OctreeNode& operator=(OctreeNode&& other) = default;

	void print() const;

	Cell cell;
	std::array<OctreeNode*, 8> children;
	bool is_parent{ false };

	size_t rank{ 1111222233334444 };             // MPI rank who owns this octree node
	size_t level{ 1111222233334444 };         // Level in the tree [0 (= root) ... depth of tree]
};
