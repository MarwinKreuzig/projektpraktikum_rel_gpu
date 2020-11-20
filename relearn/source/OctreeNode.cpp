/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "OctreeNode.h"

OctreeNode::OctreeNode() /*noexcept*/ :
	is_parent(false),
	level(0),
	rank(0), children(std::array<OctreeNode*, 8>{nullptr}) {
}

void OctreeNode::print() const {
	std::cout << "== OctreeNode (" << this << ") ==\n";

	std::cout << "  children[8]: ";
	for (const auto child : children) {
		std::cout << child << " ";
	}
	std::cout << "\n";

	std::cout << "  is_parent  : " << is_parent << "\n\n";
	std::cout << "  rank       : " << rank << "\n";
	std::cout << "  level      : " << level << "\n\n";

	cell.print();

	std::cout << std::endl;
}
