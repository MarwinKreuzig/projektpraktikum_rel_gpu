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

bool OctreeNode::is_local() const noexcept {
    return rank == MPIWrapper::get_my_rank();
}

OctreeNode* OctreeNode::insert(const Vec3d& position, const size_t neuron_id, int rank) {
    Vec3d cell_xyz_min;
    Vec3d cell_xyz_max;

    std::tie(cell_xyz_min, cell_xyz_max) = cell.get_size();

    RelearnException::check(cell_xyz_min.get_x() <= position.get_x() && position.get_x() <= cell_xyz_max.get_x(), "In OctreeNode::insert, x was not in range");
    RelearnException::check(cell_xyz_min.get_y() <= position.get_y() && position.get_y() <= cell_xyz_max.get_y(), "In OctreeNode::insert, x was not in range");
    RelearnException::check(cell_xyz_min.get_z() <= position.get_z() && position.get_z() <= cell_xyz_max.get_z(), "In OctreeNode::insert, x was not in range");

    RelearnException::check(rank >= 0, "In OctreeNode::insert, rank was smaller than 0");
    RelearnException::check(neuron_id <= Constants::uninitialized, "In OctreeNode::insert, neuron_id was too large");

    OctreeNode* new_node_to_insert = MPIWrapper::new_octree_node();
    RelearnException::check(new_node_to_insert != nullptr, "new_node_to_insert is nullptr");

    unsigned char my_idx = 0;
    unsigned char idx = 0;

    OctreeNode* prev = nullptr;
    OctreeNode* curr = this;
    // Correct position for new node not found yet
    while (nullptr != curr) {
        /**
		    * My parent already exists.
		    * Calc which child to follow, i.e., determine octant
		    */
        my_idx = curr->get_cell().get_octant_for_position(position);

        prev = curr;
        curr = curr->get_child(my_idx);
    }

    RelearnException::check(prev != nullptr, "prev is nullptr");

    /**
	    * Found my octant, but
	    * I'm the very first child of that node.
	    * I.e., the node is a leaf.
	    */
    if (!prev->is_parent()) {
        do {
            /**
			    * Make node containing my octant a parent by
			    * adding neuron in this node as child node
			    */

            // Determine octant for neuron
            const auto& cell_own_position = prev->get_cell().get_neuron_position();
            RelearnException::check(cell_own_position.has_value(), "While building the octree, the cell doesn't have a position");
            idx = prev->get_cell().get_octant_for_position(cell_own_position.value());
            OctreeNode* new_node = MPIWrapper::new_octree_node(); // new OctreeNode();
            prev->set_child(new_node, idx);

            /**
			    * Init this new node properly
			    */
            // Cell size
            Vec3d xyz_min;
            Vec3d xyz_max;
            std::tie(xyz_min, xyz_max) = prev->get_cell().get_size_for_octant(idx);

            new_node->set_cell_size(xyz_min, xyz_max);

            std::optional<Vec3d> opt_vec = prev->get_cell().get_neuron_position();
            RelearnException::check(opt_vec.has_value(), "In Octree::insert, the previous cell does not have a position");
            new_node->set_cell_neuron_position(opt_vec);

            // Neuron ID
            const auto prev_neuron_id = prev->get_cell().get_neuron_id();
            new_node->set_cell_neuron_id(prev_neuron_id);
            /**
			    * Set neuron ID of parent (inner node) to uninitialized.
			    * It is not used for inner nodes.
			    */
            prev->set_cell_neuron_id(Constants::uninitialized);
            prev->set_parent(); // Mark node as parent

            // MPI rank who owns this node
            new_node->set_rank(prev->get_rank());

            // New node is one level below
            new_node->set_level(prev->get_level() + 1);

            // Determine my octant
            my_idx = prev->get_cell().get_octant_for_position(position);

            if (my_idx == idx) {
                prev = new_node;
            }
        } while (my_idx == idx);
    }

    /**
	    * Found my position in children array,
	    * add myself to the array now
	    */
    prev->set_child(new_node_to_insert, my_idx);
    new_node_to_insert->set_level(prev->get_level() + 1); // Now we know level of me

    Vec3d xyz_min;
    Vec3d xyz_max;
    std::tie(xyz_min, xyz_max) = prev->get_cell().get_size_for_octant(my_idx);

    new_node_to_insert->set_cell_size(xyz_min, xyz_max);
    new_node_to_insert->set_cell_neuron_position({ position });
    new_node_to_insert->set_cell_neuron_id(neuron_id);
    new_node_to_insert->set_rank(rank);

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

    LogFiles::write_to_file(LogFiles::EventType::Cout, ss.str(), true);
}

const std::vector<Vec3d> OctreeNode::get_dendrite_pos_from_node_for(SignalType needed) const{
   int num_of_ports = 0;
    std::vector<Vec3d> result;
    std::stack<const OctreeNode*> stack;
    
    stack.push(this);

    while (!stack.empty()) {
        
        const OctreeNode* current_node = stack.top();
        
        stack.pop();
        if (!current_node->is_parent()) {
            num_of_ports = current_node->get_cell().get_neuron_num_dendrites_for(needed);
            if (num_of_ports > 0) {
                for(int i = 0; i<num_of_ports; i++){
                    RelearnException::check(current_node->get_cell().get_neuron_position().has_value(), "neuron had no dend_ex position");
                    result.push_back(current_node->get_cell().get_neuron_dendrite_position_for(needed).value());
                }

            }
        }

        else{
            for(int i = 0; i<8;i++){
                
                const  OctreeNode* children_node = current_node->get_child(i);
                if (children_node != nullptr && children_node->get_cell().get_neuron_num_dendrites_for(needed)>0)
                {
                    stack.push(children_node);
                }
            }
        }
    }
    return result;
}

const std::vector<Vec3d> OctreeNode::get_axon_pos_from_node_for(SignalType needed) const {
    int num_of_ports = 0;
    std::vector<Vec3d> result;
    std::stack<const OctreeNode*> stack;
    
    stack.push(this);

    while (!stack.empty()) {
        
        const OctreeNode* current_node = stack.top();
        
        stack.pop();
        if (!current_node->is_parent()) {
            num_of_ports = current_node->get_cell().get_neuron_num_axons_for(needed);
            if (num_of_ports > 0) {
                for(int i = 0; i<num_of_ports; i++){
                    RelearnException::check(current_node->get_cell().get_neuron_position().has_value(), "neuron had no dend_ex position");
                    result.push_back(current_node->get_cell().get_neuron_axon_position_for(needed).value());
                }

            }
        }

        else{
            for(int i = 0; i<8;i++){
                
                const  OctreeNode* children_node = current_node->get_child(i);
                if (children_node != nullptr && children_node->get_cell().get_neuron_num_axons_for(needed)>0)
                {
                    stack.push(children_node);
                }
            }
        }
    }
    return result;
}

