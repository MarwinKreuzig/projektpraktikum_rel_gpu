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

#include <cstddef>

class OctreeNode {
public:
	OctreeNode() /*noexcept*/;
	~OctreeNode() /*noexcept*/;

	OctreeNode(const OctreeNode& other) = default;
	OctreeNode(OctreeNode&& other) = default;

	OctreeNode& operator=(const OctreeNode& other) = default;
	OctreeNode& operator=(OctreeNode&& other) = default;

	void print() const;

	Cell cell;
	OctreeNode* children[8];
	bool is_parent;

	int rank;             // MPI rank who owns this octree node
	size_t level;         // Level in the tree [0 (= root) ... depth of tree]
};
