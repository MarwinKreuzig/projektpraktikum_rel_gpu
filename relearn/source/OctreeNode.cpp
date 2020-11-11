/*
 * File:   OctreeNode.cpp
 * Author: rinke
 *
 * Created on Oct 28, 2014
 */

#include "OctreeNode.h"

OctreeNode::OctreeNode() /*noexcept*/ :
	is_parent(0),
	level(0),
	rank(0) {
	int i;

	for (i = 0; i < 8; i++) {
		children[i] = nullptr;
	}
}

OctreeNode::~OctreeNode() /*noexcept*/ {}

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
