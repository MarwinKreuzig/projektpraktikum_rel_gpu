/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

/*********************************************************************************
  * NOTE: We include Neurons.h here as the class Octree uses types from Neurons.h *
  * Neurons.h also includes Octree.h as it uses it too                            *
  *********************************************************************************/

#include "Octree.h"

#include "../io/LogFiles.h"
#include "../neurons/Neurons.h"
#include "../neurons/models/SynapticElements.h"
#include "../structure/SpaceFillingCurve.h"
#include "../util/Random.h"
#include "../util/RelearnException.h"
#include "../util/Timers.h"

#include <sstream>

Octree::Octree(const Vec3d& xyz_min, const Vec3d& xyz_max, size_t level_of_branch_nodes)
    : root_level(0)
    , level_of_branch_nodes(level_of_branch_nodes) {

    const auto num_local_trees = 1ULL << (3 * level_of_branch_nodes);
    local_trees.resize(num_local_trees, nullptr);

    set_size(xyz_min, xyz_max);
    construct_global_tree_part();
}

Octree::~Octree() /*noexcept(false)*/ {
    if (!no_free_in_destructor) {
        // Free all nodes
        free();
    }
}

void Octree::postorder_print() {
    std::stack<StackElement> stack;

    // Tree is empty
    if (root == nullptr) {
        return;
    }

    stack.emplace(root, 0);

    while (!stack.empty()) {

        std::stringstream ss;

        auto& elem = stack.top();
        const auto depth = static_cast<int>(elem.get_depth_in_tree());
        const auto* ptr = elem.get_octree_node();

        // Visit node now
        if (elem.get_visited()) {
            Vec3d xyz_min;
            Vec3d xyz_max;
            std::optional<Vec3d> xyz_pos;

            // Print node's address
            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }
            ss << "Address: " << ptr << "\n";

            // Print cell extent
            std::tie(xyz_min, xyz_max) = ptr->get_cell().get_size();
            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }

            ss << "Cell extent: (" << xyz_min.get_x() << " .. " << xyz_max.get_x() << ", "
               << xyz_min.get_y() << " .. " << xyz_max.get_y() << ", "
               << xyz_min.get_z() << " .. " << xyz_max.get_z() << ")\n";

            // Print neuron ID
            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }
            ss << "Neuron ID: " << ptr->get_cell().get_neuron_id() << "\n";

            // Print number of dendrites
            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }
            ss << "Number dendrites (exc, inh): (" << ptr->get_cell().get_neuron_num_dendrites_exc()
               << ", " << ptr->get_cell().get_neuron_num_dendrites_inh() << ")\n";

            // Print position DendriteType::EXCITATORY
            xyz_pos = ptr->get_cell().get_neuron_position_exc();
            // Note if position is invalid
            if (!xyz_pos.has_value()) {
                ss << "-- invalid!";
            }

            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }
            ss << "Position exc: (" << xyz_pos.value().get_x() << ", " << xyz_pos.value().get_y() << ", " << xyz_pos.value().get_z() << ") ";

            ss << "\n";
            // Print position DendriteType::INHIBITORY
            xyz_pos = ptr->get_cell().get_neuron_position_inh();
            // Note if position is invalid
            if (!xyz_pos.has_value()) {
                ss << "-- invalid!";
            }
            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }
            ss << "Position inh: (" << xyz_pos.value().get_x() << ", " << xyz_pos.value().get_y() << ", " << xyz_pos.value().get_z() << ") ";
            ss << "\n";
            ss << "\n";

            stack.pop();
        }
        // Visit children first
        else {
            elem.set_visited();

            for (auto j = 0; j < depth; j++) {
                ss << " ";
            }

            ss << "Child indices: ";
            int id = 0;
            for (const auto& child : ptr->get_children()) {
                if (child != nullptr) {
                    ss << id << " ";
                }
                id++;
            }
            /**
			* Push in reverse order so that visiting happens in
			* increasing order of child indices
			*/
            for (auto it = ptr->get_children().crbegin(); it != ptr->get_children().crend(); ++it) {
                if (*it != nullptr) {
                    stack.emplace(*it, static_cast<size_t>(depth) + 1);
                }
            }
            ss << "\n";
        }

        LogFiles::write_to_file(LogFiles::EventType::Cout, true, ss.str());
    }
}

void Octree::construct_global_tree_part() {
    RelearnException::check(root == nullptr, "root was not null in the construction of the global state!");

    SpaceFillingCurve<Morton> space_curve;
    space_curve.set_refinement_level(level_of_branch_nodes);

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto num_cells_per_dimension = 1 << level_of_branch_nodes; // (2^level_of_branch_nodes)

    const auto& cell_length = (xyz_max - xyz_min) / num_cells_per_dimension;

    const auto cell_length_x = cell_length.get_x();
    const auto cell_length_y = cell_length.get_y();
    const auto cell_length_z = cell_length.get_z();

    OctreeNode* local_root = MPIWrapper::new_octree_node();
    RelearnException::check(local_root != nullptr, "local_root is nullptr");

    local_root->set_cell_neuron_id(Constants::uninitialized);
    local_root->set_cell_size(xyz_min, xyz_max);
    local_root->set_level(0);
    local_root->set_rank(my_rank);
    local_root->set_cell_neuron_position(xyz_min + (cell_length / 2));

    const auto root_index1d = space_curve.map_3d_to_1d(Vec3s{ 0, 0, 0 });
    local_trees[root_index1d] = local_root;

    root = local_root;

    for (size_t id_x = 0; id_x < num_cells_per_dimension; id_x++) {
        for (size_t id_y = 0; id_y < num_cells_per_dimension; id_y++) {
            for (size_t id_z = 0; id_z < num_cells_per_dimension; id_z++) {
                if (id_x == 0 && id_y == 0 && id_z == 0) {
                    continue;
                }

                const Vec3d cell_offset{ id_x * cell_length_x, id_y * cell_length_y, id_z * cell_length_z };
                const auto& cell_min = xyz_min + cell_offset;
                const auto& cell_position = cell_min + (cell_length / 2);

                auto* current_node = root->insert(cell_position, Constants::uninitialized, my_rank);

                //const auto index1d = space_curve.map_3d_to_1d(Vec3s{ id_x, id_y, id_z });
                //local_trees[index1d] = current_node;
            }
        }
    }

    std::stack<std::pair<OctreeNode*, Vec3s>> stack{};
    stack.emplace(root, Vec3s{ 0, 0, 0 });

    while (!stack.empty()) {
        const auto [ptr, index3d] = stack.top();
        stack.pop();

        if (!ptr->is_parent()) {
            const auto index1d = space_curve.map_3d_to_1d(index3d);
            local_trees[index1d] = ptr;
            continue;
        }

        for (auto id = 0; id < 8; id++) {
            auto child_node = ptr->get_child(id);

            const size_t larger_x = ((id & 1) == 0) ? 0 : 1;
            const size_t larger_y = ((id & 2) == 0) ? 0 : 1;
            const size_t larger_z = ((id & 4) == 0) ? 0 : 1;

            const Vec3s offset{ larger_x, larger_y, larger_z };
            const Vec3s pos = (index3d * 2) + offset;
            stack.emplace(child_node, pos);
        }
    }
}

// Insert neuron into the tree
OctreeNode* Octree::insert(const Vec3d& position, size_t neuron_id, int rank) {
    RelearnException::check(xyz_min.get_x() <= position.get_x() && position.get_x() <= xyz_max.get_x(), "In Octree::insert, x was not in range");
    RelearnException::check(xyz_min.get_y() <= position.get_y() && position.get_y() <= xyz_max.get_y(), "In Octree::insert, x was not in range");
    RelearnException::check(xyz_min.get_z() <= position.get_z() && position.get_z() <= xyz_max.get_z(), "In Octree::insert, x was not in range");

    RelearnException::check(rank >= 0, "In Octree::insert, rank was smaller than 0");
    RelearnException::check(neuron_id < Constants::uninitialized, "In Octree::insert, neuron_id was too large");

    // Tree is empty
    if (nullptr == root) {
        // Create new tree node for the neuron
        OctreeNode* new_node_to_insert = MPIWrapper::new_octree_node();
        RelearnException::check(new_node_to_insert != nullptr, "new_node_to_insert is nullptr");

        // Init cell size with simulation box size
        new_node_to_insert->set_cell_size(this->xyz_min, this->xyz_max);
        new_node_to_insert->set_cell_neuron_position({ position });
        new_node_to_insert->set_cell_neuron_id(neuron_id);
        new_node_to_insert->set_rank(rank);

        // Init root with tree's root level
        new_node_to_insert->set_level(root_level);
        root = new_node_to_insert;

        return new_node_to_insert;
    }

    return root->insert(position, neuron_id, rank);
}

void Octree::print() {
    postorder_print();
}

void Octree::free() {
    // Provide allocator so that it can be used to free memory again
    const FunctorFreeNode free_node{};

    // The functor containing the visit function is of type FunctorFreeNode
    tree_walk_postorder<FunctorFreeNode>(root, free_node);
}

// The caller must ensure that only inner nodes are visited. "max_level" must be chosen correctly for this
void Octree::update_from_level(size_t max_level) {
    std::vector<double> dendrites_exc_cnts;
    std::vector<unsigned int> dendrites_exc_connected_cnts;
    std::vector<double> dendrites_inh_cnts;
    std::vector<unsigned int> dendrites_inh_connected_cnts;

    const FunctorUpdateNode update_functor(dendrites_exc_cnts, dendrites_exc_connected_cnts, dendrites_inh_cnts, dendrites_inh_connected_cnts, 0);

    /**
	* NOTE: It *must* be ensured that in tree_walk_postorder() only inner nodes
	* are visited as update_node cannot update leaf nodes here
	*/

    // The functor containing the visit function is of type FunctorUpdateNode
    tree_walk_postorder<FunctorUpdateNode>(root, update_functor, max_level);
}

void Octree::update_local_trees(const SynapticElements& dendrites_exc, const SynapticElements& dendrites_inh, size_t num_neurons) {
    const auto& de_ex_cnt = dendrites_exc.get_cnts();
    const auto& de_ex_conn_cnt = dendrites_exc.get_connected_cnts();
    const auto& de_in_cnt = dendrites_inh.get_cnts();
    const auto& de_in_conn_cnt = dendrites_inh.get_connected_cnts();

    const auto my_rank = MPIWrapper::get_my_rank();

    for (auto* local_tree : local_trees) {
        if (local_tree->get_rank() != my_rank) {
            continue;
        }

        const FunctorUpdateNode update_functor(de_ex_cnt, de_ex_conn_cnt, de_in_cnt, de_in_conn_cnt, num_neurons);
        // The functor containing the visit function is of type FunctorUpdateNode
        tree_walk_postorder<FunctorUpdateNode>(local_tree, update_functor);
    }
}

void Octree::empty_remote_nodes_cache() {
    for (auto& remode_node_in_cache : remote_nodes_cache) {
        MPIWrapper::delete_octree_node(remode_node_in_cache.second);
    }

    remote_nodes_cache.clear();
}

void Octree::synchronize_local_trees() {
    // Lock local RMA memory for local stores
    /**
    * Exchange branch nodes
    */
    GlobalTimers::timers.start(TimerRegion::EXCHANGE_BRANCH_NODES);
    OctreeNode* rma_buffer_branch_nodes = MPIWrapper::get_buffer_octree_nodes();
    // Copy local trees' root nodes to correct positions in receive buffer

    const size_t num_local_trees = local_trees.size() / MPIWrapper::get_num_ranks();
    for (size_t i = 0; i < local_trees.size(); i++) {
        rma_buffer_branch_nodes[i] = *local_trees[i];
    }

    // Allgather in-place branch nodes from every rank
    MPIWrapper::all_gather_inline(rma_buffer_branch_nodes, num_local_trees, MPIWrapper::Scope::global);

    GlobalTimers::timers.stop_and_add(TimerRegion::EXCHANGE_BRANCH_NODES);

    // Insert only received branch nodes into global tree
    // The local ones are already in the global tree
    GlobalTimers::timers.start(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);
    const size_t num_rma_buffer_branch_nodes = MPIWrapper::get_num_buffer_octree_nodes();
    for (size_t i = 0; i < num_rma_buffer_branch_nodes; i++) {
        *local_trees[i] = rma_buffer_branch_nodes[i];
    }
    GlobalTimers::timers.stop_and_add(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);

    // Update global tree
    GlobalTimers::timers.start(TimerRegion::UPDATE_GLOBAL_TREE);
    const auto level_branches = get_level_of_branch_nodes();

    // Only update whenever there are other branches to update
    if (level_branches > 0) {
        update_from_level(level_branches - 1);
    }
    GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_GLOBAL_TREE);
}
