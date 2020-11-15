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
	rank(0) {
	int i;

	for (i = 0; i < 8; i++) {
		children[i] = nullptr;
	}
}

void OctreeNode::print() const {
	using namespace std;

	cout << "== OctreeNode (" << this << ") ==\n";

	cout << "  children[8]: ";
	for (int i = 0; i < 8; i++) {
		cout << children[i] << " ";
	}
	cout << "\n";

	cout << "  is_parent  : " << is_parent << "\n\n";
	cout << "  rank       : " << rank << "\n";
	cout << "  level      : " << level << "\n\n";

	cell.print();

	cout << endl;
}
