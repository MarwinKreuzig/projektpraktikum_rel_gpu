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
	OctreeNode();
	~OctreeNode();

	void print() const;

	Cell cell;
	OctreeNode* children[8];
	bool is_parent;

	int rank;             // MPI rank who owns this octree node
	size_t level;         // Level in the tree [0 (= root) ... depth of tree]
};


#endif /* OCTREENODE_H */
