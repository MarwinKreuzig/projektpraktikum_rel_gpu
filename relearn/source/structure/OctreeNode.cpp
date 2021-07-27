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

#include "../io/LogFiles.h"
#include "../mpi/MPIWrapper.h"

#include <sstream>

bool OctreeNode::is_local() const {
    return rank == MPIWrapper::get_my_rank();
}

OctreeNode* OctreeNode::insert(const Vec3d& position, const size_t neuron_id, int rank) {
    Vec3d cell_xyz_min;
    Vec3d cell_xyz_max;

    std::tie(cell_xyz_min, cell_xyz_max) = cell.get_size();

    RelearnException::check(cell_xyz_min.get_x() <= position.get_x() && position.get_x() <= cell_xyz_max.get_x(), "In OctreeNode::insert, x was not in range");
    RelearnException::check(cell_xyz_min.get_y() <= position.get_y() && position.get_y() <= cell_xyz_max.get_y(), "In OctreeNode::insert, y was not in range");
    RelearnException::check(cell_xyz_min.get_z() <= position.get_z() && position.get_z() <= cell_xyz_max.get_z(), "In OctreeNode::insert, z was not in range");

    RelearnException::check(rank >= 0, "In OctreeNode::insert, rank was smaller than 0");
    RelearnException::check(neuron_id <= Constants::uninitialized, "In OctreeNode::insert, neuron_id was too large");

    unsigned char new_position_octant = 0;

    OctreeNode* parent_node = nullptr;

    // Correct position for new node not found yet
    for (OctreeNode* current_node = this; nullptr != current_node;) {
        /**
		 * My parent already exists.
		 * Calc which child to follow, i.e., determine octant
		 */
        new_position_octant = current_node->get_cell().get_octant_for_position(position);

        parent_node = current_node;
        current_node = current_node->get_child(new_position_octant);
    }

    RelearnException::check(parent_node != nullptr, "parent_node is nullptr");

    /**
	 * Found my octant, but
	 * I'm the very first child of that node.
	 * I.e., the node is a leaf.
	 */
    if (!parent_node->is_parent()) {
        /**
         * The found parent node is virtual and can just be substituted,
         * i.e., it was constructed while constructing the upper part to the branch nodes.
         */
        if (parent_node->get_cell_neuron_id() == Constants::uninitialized && neuron_id != parent_node->get_cell_neuron_id()) {
            parent_node->set_cell_neuron_id(neuron_id);
            parent_node->set_cell_neuron_position({ position });
            parent_node->set_rank(rank);
            return parent_node;
        }

        for (unsigned char idx = new_position_octant; idx == new_position_octant;) {
            /**
		     * Make node containing my octant a parent by
		     * adding neuron in this node as child node
		     */

            // Determine octant for neuron
            const auto& cell_own_position = parent_node->get_cell().get_dendrite_position();
            RelearnException::check(cell_own_position.has_value(), "While building the octree, the cell doesn't have a position");

            idx = parent_node->get_cell().get_octant_for_position(cell_own_position.value());
            OctreeNode* new_node = MPIWrapper::new_octree_node();
            parent_node->set_child(new_node, idx);

            /**
			 * Init this new node properly
			 */
            Vec3d xyz_min;
            Vec3d xyz_max;
            std::tie(xyz_min, xyz_max) = parent_node->get_cell().get_size_for_octant(idx);

            new_node->set_cell_size(xyz_min, xyz_max);
            new_node->set_cell_neuron_position(cell_own_position);

            // Neuron ID
            const auto prev_neuron_id = parent_node->get_cell_neuron_id();
            new_node->set_cell_neuron_id(prev_neuron_id);

            /**
			 * Set neuron ID of parent (inner node) to uninitialized.
			 * It is not used for inner nodes.
			 */
            parent_node->set_cell_neuron_id(Constants::uninitialized);
            parent_node->set_parent(); // Mark node as parent

            // MPI rank who owns this node
            new_node->set_rank(parent_node->get_rank());

            // New node is one level below
            new_node->set_level(parent_node->get_level() + 1);

            // Determine my octant
            new_position_octant = parent_node->get_cell().get_octant_for_position(position);

            if (new_position_octant == idx) {
                parent_node = new_node;
            }
        }
    }

    OctreeNode* new_node_to_insert = MPIWrapper::new_octree_node();
    RelearnException::check(new_node_to_insert != nullptr, "new_node_to_insert is nullptr");

    /**
	 * Found my position in children array,
	 * add myself to the array now
	 */
    parent_node->set_child(new_node_to_insert, new_position_octant);
    new_node_to_insert->set_level(parent_node->get_level() + 1); // Now we know level of me

    Vec3d xyz_min;
    Vec3d xyz_max;
    std::tie(xyz_min, xyz_max) = parent_node->get_cell().get_size_for_octant(new_position_octant);

    new_node_to_insert->set_cell_size(xyz_min, xyz_max);
    new_node_to_insert->set_cell_neuron_position({ position });
    new_node_to_insert->set_cell_neuron_id(neuron_id);
    new_node_to_insert->set_rank(rank);

    bool has_children = false;

    for (auto* child : children) {
        if (child != nullptr) {
            has_children = true;
        }
    }

    RelearnException::check(has_children, "");

    return new_node_to_insert;
}

void OctreeNode::print() const {
    std::stringstream ss;

    ss << "== OctreeNode (" << this << ") ==\n";

    ss << "  children[8]: ";
    for (const auto* const child : children) {
        ss << child << " ";
    }
    ss << "\n";

    ss << "  is_parent  : " << parent << "\n\n";
    ss << "  rank       : " << rank << "\n";
    ss << "  level      : " << level << "\n\n";

    cell.print();

    ss << "\n";

    LogFiles::write_to_file(LogFiles::EventType::Cout, true, ss.str());
}
