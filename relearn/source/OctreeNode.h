/*
 * File:   OctreeNode.h
 * Author: rinke
 *
 * Created on Oct 28, 2014
 */

#ifndef OCTREENODE_H
#define OCTREENODE_H

#include <cstddef>
#include "Cell.h"

class OctreeNode {
public:
	OctreeNode() noexcept;
	~OctreeNode() noexcept;

	OctreeNode(const OctreeNode& other) = default;
	OctreeNode(OctreeNode&& other) = default;

	OctreeNode& operator = (const OctreeNode & other) = default;
	OctreeNode& operator = (OctreeNode && other) = default;

	void print() const;

	Cell cell;
	OctreeNode* children[8];
	bool is_parent;

	int rank;             // MPI rank who owns this octree node
	size_t level;         // Level in the tree [0 (= root) ... depth of tree]
};


#endif /* OCTREENODE_H */
