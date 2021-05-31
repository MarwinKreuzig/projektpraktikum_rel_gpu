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

#include "LogFiles.h"

#include <sstream>

#include <stack>

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
