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
#include "../util/DeriativesAndFunctions.h"

#include <sstream>

Octree::Octree(const Vec3d& xyz_min, const Vec3d& xyz_max, size_t level_of_branch_nodes)
    : root_level(0)
    , naive_method(acceptance_criterion == 0.0)
    , level_of_branch_nodes(level_of_branch_nodes) {

    const auto num_local_trees = 1ULL << (3 * level_of_branch_nodes);
    local_trees.resize(num_local_trees, nullptr);

    set_size(xyz_min, xyz_max);
    construct_global_tree_part();
}

Octree::Octree(const Vec3d& xyz_min, const Vec3d& xyz_max, size_t level_of_branch_nodes, double acceptance_criterion, double sigma)
    : root_level(0)
    , naive_method(acceptance_criterion == 0.0)
    , level_of_branch_nodes(level_of_branch_nodes) {

    const auto num_local_trees = 1ULL << (3 * level_of_branch_nodes);
    local_trees.resize(num_local_trees, nullptr);

    set_acceptance_criterion(acceptance_criterion);
    set_probability_parameter(sigma);
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
            xyz_pos = ptr->get_cell().get_neuron_position_dendrites_exc();
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
            xyz_pos = ptr->get_cell().get_neuron_position_dendrites_inh();
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

        LogFiles::write_to_file(LogFiles::EventType::Cout, ss.str(), true);
    }
}

std::tuple<bool, bool> Octree::acceptance_criterion_test(const Vec3d& axon_pos_xyz,
    const OctreeNode* const node_with_dendrite,
    SignalType dendrite_type_needed,
    bool naive_method) const /*noexcept*/ {

    const auto has_vacant_dendrites = node_with_dendrite->get_cell().get_neuron_num_dendrites_for(dendrite_type_needed) != 0;

    // Use naive method
    if (naive_method) {
        // Accept leaf only
        const auto is_child = !node_with_dendrite->is_parent();
        return std::make_tuple(is_child, has_vacant_dendrites);
    }

    if (!has_vacant_dendrites) {
        return std::make_tuple(false, false);
    }

    /**
	* Node is leaf node, i.e., not super neuron.
	* Thus the node is precise. Accept it.
	*/
    if (!node_with_dendrite->is_parent()) {
        return std::make_tuple(true, true);
    }

    // Check distance between neuron with axon and neuron with dendrite
    const auto& target_xyz = node_with_dendrite->get_cell().get_neuron_position_for(dendrite_type_needed);

    // NOTE: This assertion fails when considering inner nodes that don't have dendrites.
    RelearnException::check(target_xyz.has_value(), "target_xyz was bad");

    // Calc Euclidean distance between source and target neuron
    const auto distance_vector = target_xyz.value() - axon_pos_xyz;
    const auto distance = distance_vector.calculate_p_norm(2.0);

    const auto length = node_with_dendrite->get_cell().get_maximal_dimension_difference();

    // Original Barnes-Hut acceptance criterion
    const auto ret_val = (length / distance) < acceptance_criterion;
    return std::make_tuple(ret_val, has_vacant_dendrites);
}

ProbabilitySubintervalVector Octree::get_nodes_for_interval(
    const Vec3d& axon_pos_xyz,
    OctreeNode* root,
    SignalType dendrite_type_needed,
    bool naive_method) {

    /* Subtree is not empty AND (Dendrites are available OR We use naive method) */
    const auto flag = (root != nullptr) && (root->get_cell().get_neuron_num_dendrites_for(dendrite_type_needed) != 0 || naive_method);
    if (!flag) {
        return {};
    }

    std::stack<OctreeNode*> stack;
    std::array<OctreeNode*, Constants::number_oct> local_children = { nullptr };

    /**
	* The root node is parent (i.e., contains a super neuron) and thus cannot be the target neuron.
	* So, start considering its children.
	*/
    if (root->is_parent()) {
        // Node is owned by this rank
        if (root->is_local()) {
            // Push root's children onto stack
            const auto& children = root->get_children();
            for (auto it = children.crbegin(); it != children.crend(); ++it) {
                if (*it != nullptr) {
                    stack.push(*it);
                }
            }
        }
        // Node is owned by different rank
        else {
            const auto target_rank = root->get_rank();
            NodesCacheKey rank_addr_pair;
            rank_addr_pair.first = target_rank;

            // Start access epoch to remote rank
            MPIWrapper::lock_window(target_rank, MPI_Locktype::shared);

            // Fetch remote children if they exist
            // NOLINTNEXTLINE
            for (auto i = 7; i >= 0; i--) {
                if (nullptr == root->get_child(i)) {
                    // NOLINTNEXTLINE
                    local_children[i] = nullptr;
                    continue;
                }

                rank_addr_pair.second = root->get_child(i);

                std::pair<NodesCacheKey, NodesCacheValue> cache_key_val_pair;
                cache_key_val_pair.first = rank_addr_pair;
                cache_key_val_pair.second = nullptr;

                // Get cache entry for "cache_key_val_pair"
                // It is created if it does not exist yet
                std::pair<NodesCache::iterator, bool> ret = remote_nodes_cache.insert(cache_key_val_pair);

                // Cache entry just inserted as it was not in cache
                // So, we still need to init the entry by fetching
                // from the target rank
                if (ret.second) {
                    ret.first->second = MPIWrapper::new_octree_node();
                    auto* local_child_addr = ret.first->second;

                    // Calc displacement from absolute address
                    const auto target_child_displ = MPIWrapper::get_ptr_displacement(target_rank, root->get_child(i));

                    MPIWrapper::get(local_child_addr, target_rank, target_child_displ);
                }

                // Remember address of node
                // NOLINTNEXTLINE
                local_children[i] = ret.first->second;
            }

            // Complete access epoch
            MPIWrapper::unlock_window(target_rank);

            // Push root's children onto stack
            for (auto it = local_children.crbegin(); it != local_children.crend(); ++it) {
                if (*it != nullptr) {
                    stack.push(*it);
                }
            }
        } // Node owned by different rank
    } // Root of subtree is parent
    else {
        /**
		* The root node is a leaf and thus contains the target neuron.
		*
		* NOTE: Root is not intended to be a leaf but we handle this as well.
		* Without pushing root onto the stack, it would not make it into the "vector" of nodes.
		*/
        stack.push(root);
    }

    ProbabilitySubintervalVector vector;
    vector.reserve(stack.size());

    bool has_vacant_dendrites = false;
    while (!stack.empty()) {
        // Get top-of-stack node and remove it from stack
        auto* stack_elem = stack.top();
        stack.pop();

        /**
		* Should node be used for probability interval?
		*
		* Only take those that have dendrites available
		*/
        auto acc_vac = acceptance_criterion_test(axon_pos_xyz, stack_elem, dendrite_type_needed, naive_method);
        const auto accept = std::get<0>(acc_vac);
        const auto has_vac = std::get<1>(acc_vac);
        has_vacant_dendrites = has_vac;

        if (accept) {
            // Insert node into vector
            vector.emplace_back(std::make_shared<ProbabilitySubinterval>(stack_elem));
            continue;
        }

        if (!(has_vacant_dendrites || naive_method)) {
            continue;
        }

        // Node is owned by this rank
        if (stack_elem->is_local()) {
            // Push node's children onto stack
            const auto& children = stack_elem->get_children();
            for (auto it = children.crbegin(); it != children.crend(); ++it) {
                if (*it != nullptr) {
                    stack.push(*it);
                }
            }
            continue;
        }

        // Node is owned by different rank
        const auto target_rank = stack_elem->get_rank();
        NodesCacheKey rank_addr_pair;
        rank_addr_pair.first = target_rank;

        // Start access epoch to remote rank
        MPIWrapper::lock_window(target_rank, MPI_Locktype::shared);

        // Fetch remote children if they exist
        // NOLINTNEXTLINE
        for (auto i = 7; i >= 0; i--) {
            if (nullptr == stack_elem->get_child(i)) {
                // NOLINTNEXTLINE
                local_children[i] = nullptr;
                continue;
            }

            rank_addr_pair.second = stack_elem->get_child(i);

            std::pair<NodesCacheKey, NodesCacheValue> cache_key_val_pair;
            cache_key_val_pair.first = rank_addr_pair;
            cache_key_val_pair.second = nullptr;

            // Get cache entry for "rank_addr_pair"
            // It is created if it does not exist yet
            std::pair<NodesCache::iterator, bool> ret = remote_nodes_cache.insert(cache_key_val_pair);

            // Cache entry just inserted as it was not in cache
            // So, we still need to init the entry by fetching
            // from the target rank
            if (ret.second) {
                ret.first->second = MPIWrapper::new_octree_node();
                auto* local_child_addr = ret.first->second;
                const auto target_child_displ = MPIWrapper::get_ptr_displacement(target_rank, stack_elem->get_child(i));

                MPIWrapper::get(local_child_addr, target_rank, target_child_displ);
            }

            // Remember local address of node

            // NOLINTNEXTLINE
            local_children[i] = ret.first->second;
        }

        // Complete access epoch
        MPIWrapper::unlock_window(target_rank);

        // Push node's children onto stack
        for (auto it = local_children.crbegin(); it != local_children.crend(); ++it) {
            if (*it != nullptr) {
                stack.push(*it);
            }
        }
    } // while

    return vector;
}

std::vector<double> Octree::create_interval(size_t src_neuron_id, const Vec3d& axon_pos_xyz, SignalType dendrite_type_needed, const ProbabilitySubintervalVector& vector) const {
    // Does vector contain nodes?
    if (vector.empty()) {
        return {};
    }

    double sum = 0.0;

    std::vector<double> probabilities;
    std::for_each(vector.cbegin(), vector.cend(), [&](const std::shared_ptr<ProbabilitySubinterval>& ptr) {
        const auto prob = calc_attractiveness_to_connect(src_neuron_id, axon_pos_xyz, *(ptr->get_octree_node()), dendrite_type_needed);
        probabilities.push_back(prob);
        sum += prob;
    });

    /**
	* Make sure that we don't divide by 0 in case
	* all probabilities from above are 0.
	*/
    if (sum == 0.0) {
        return probabilities;
    }

    std::transform(probabilities.begin(), probabilities.end(), probabilities.begin(), [sum](double prob) { return prob / sum; });

    return probabilities;
}

double Octree::calc_attractiveness_to_connect(
    size_t src_neuron_id,
    const Vec3d& axon_pos_xyz,
    const OctreeNode& node_with_dendrite,
    SignalType dendrite_type_needed) const /*noexcept*/ {

    /**
	* If the axon's neuron itself is considered as target neuron, set attractiveness to 0 to avoid forming an autapse (connection to itself).
	* This can be done as the axon's neuron cells are always resolved until the normal (vs. super) axon's neuron is reached.
	* That is, the dendrites of the axon's neuron are not included in any super neuron considered.
	* However, this only works under the requirement that "acceptance_criterion" is <= 0.5.
	*/
    if ((!node_with_dendrite.is_parent()) && (src_neuron_id == node_with_dendrite.get_cell().get_neuron_id())) {
        return 0.0;
    }

    const auto& target_xyz = node_with_dendrite.get_cell().get_neuron_position_for(dendrite_type_needed);
    RelearnException::check(target_xyz.has_value(), "target_xyz is bad");

    const auto num_dendrites = node_with_dendrite.get_cell().get_neuron_num_dendrites_for(dendrite_type_needed);

    const auto position_diff = target_xyz.value() - axon_pos_xyz;
    const auto eucl_length = position_diff.calculate_p_norm(2.0);
    const auto numerator = pow(eucl_length, 2.0);

    // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
    const auto ret_val = (num_dendrites * exp(-numerator / (sigma * sigma)));
    return ret_val;
}

const std::vector<double> Octree::calc_attractiveness_to_connect_FMM(OctreeNode *source, const SignalType dendrite_type_needed) {
    
    // calculate vacant number of neurons in souce box
    const unsigned int source_num = source->get_cell().get_neuron_num_axons_for(dendrite_type_needed);
    //center of source box
    RelearnException::check(source->get_cell().get_neuron_axon_position_for(dendrite_type_needed).has_value(), "Source Box has no center!");
    Vec3d center_of_source_box = source->get_cell().get_neuron_axon_position_for(dendrite_type_needed).value();

    //find out the length of the interactionlist
    size_t target_list_length = source->get_interactionlist_length();

    //Initialize return value to 0
    std::vector<double> result(target_list_length, 0);

    //fill source list
    const std::vector<Vec3d> source_neurons_pos = source->get_axon_pos_from_node_for(dendrite_type_needed);
 
    bool hermite_set = false;
    std::vector<double> hermite_coefficients;
    hermite_coefficients.reserve(pow(Constants::p,3));
    //when there are not enough neurons in the source box ...
    if (source_num <= Constants::max_neurons_in_source) {
        
        for (size_t i = 0; i < target_list_length; i++) {
            // calculate vacant number of neurons in target box
            int target_num = (source->get_from_interactionlist(i))->get_cell().get_neuron_num_dendrites_for(dendrite_type_needed);
            //fill target list
            const std::vector<Vec3d> target_neurons_pos = source->get_from_interactionlist(i)->get_dendrite_pos_from_node_for(dendrite_type_needed);
            //... and there are not enough neurons in the target
            if (target_num <= Constants::max_neurons_in_target) {
                //calculate via direct Gauss
                result[i] = Functions::calc_direct_gauss(source_neurons_pos,target_neurons_pos);
            } else {
                //... and there are enough neurons in target
                //source to Taylor-Series about center of box C and direkt evaluation
                RelearnException::check(source->get_from_interactionlist(i)->get_cell().get_neuron_dendrite_position_for(dendrite_type_needed).has_value(), "Target Box has no center!");
                Vec3d center_of_target_box = source->get_from_interactionlist(i)->get_cell().get_neuron_position_for(dendrite_type_needed).value();
                result[i] = Functions::calc_taylor_expansion(source_neurons_pos, target_neurons_pos, center_of_target_box);
            }
        }

    } else //when there are enough neurons in the source box...
    { //Hermite Expansion about center of source box
        if (hermite_set == false) {
            //calculate Hermite coefficients
            Functions::calc_hermite_coefficients(center_of_source_box, source_neurons_pos, hermite_coefficients);
            hermite_set == true;
        }
        for (size_t i = 0; i < target_list_length; i++) {

            // get vacant number of neurons in target box
            int target_num = (source->get_from_interactionlist(i))->get_cell().get_neuron_num_dendrites_for(dendrite_type_needed);
            //fill target list
            const std::vector<Vec3d> target_neurons_pos = source->get_from_interactionlist(i)->get_dendrite_pos_from_node_for(dendrite_type_needed);
            
            //... and there are not enough neurons in the target
                //evaluate Hermite expansion at each target
                result[i] = Functions::calc_hermite(target_neurons_pos, hermite_coefficients, center_of_source_box);
        }
    }
    return result;
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

    std::vector<double> axons_exc_cnts;
    std::vector<unsigned int> axons_exc_connected_cnts;
    std::vector<double> axons_inh_cnts;
    std::vector<unsigned int> axons_inh_connected_cnts;

    const FunctorUpdateNode update_functor(
        dendrites_exc_cnts, 
        dendrites_exc_connected_cnts, 
        dendrites_inh_cnts, 
        dendrites_inh_connected_cnts,
        axons_exc_cnts,
        axons_exc_connected_cnts,
        axons_inh_cnts,
        axons_inh_connected_cnts,
        0);

    /**
	* NOTE: It *must* be ensured that in tree_walk_postorder() only inner nodes
	* are visited as update_node cannot update leaf nodes here
	*/

    // The functor containing the visit function is of type FunctorUpdateNode
    tree_walk_postorder<FunctorUpdateNode>(root, update_functor, max_level);
}

void Octree::update_local_trees(const SynapticElements& dendrites_exc, const SynapticElements& dendrites_inh, const SynapticElements& axons, size_t num_neurons) {
    const auto& de_ex_cnt = dendrites_exc.get_cnts();
    const auto& de_ex_conn_cnt = dendrites_exc.get_connected_cnts();
    const auto& de_in_cnt = dendrites_inh.get_cnts();
    const auto& de_in_conn_cnt = dendrites_inh.get_connected_cnts();

    const auto& axons_cnt = axons.get_cnts();
    const auto& axons_conn_cnt = axons.get_connected_cnts();

    std::vector<double> ax_ex_cnt;
    std::vector<std::seed_seq::result_type> ax_ex_conn_cnt;
    std::vector<double> ax_in_cnt;
    std::vector<std::seed_seq::result_type> ax_in_conn_cnt;

    for(int i = 0; i < axons_cnt.size(); i++){
        if (axons.get_signal_type(i) == SignalType::EXCITATORY)
        {
            ax_ex_cnt.push_back(axons_cnt.at(i));
            ax_ex_conn_cnt.push_back(axons_conn_cnt.at(i));

            ax_in_cnt.push_back(0);
            ax_in_conn_cnt.push_back(0);
        }
        if (axons.get_signal_type(i) == SignalType::INHIBITORY){
            ax_in_cnt.push_back(axons_cnt.at(i));
            ax_in_conn_cnt.push_back(axons_conn_cnt.at(i));
            
            ax_ex_cnt.push_back(0);
            ax_ex_conn_cnt.push_back(0);
        }
    }
    

    for (auto* local_tree : local_trees) {
        const FunctorUpdateNode update_functor(de_ex_cnt, de_ex_conn_cnt, de_in_cnt, de_in_conn_cnt, ax_ex_cnt,ax_ex_conn_cnt,ax_in_cnt,ax_in_conn_cnt, num_neurons);

        // The functor containing the visit function is of type FunctorUpdateNode
        tree_walk_postorder<FunctorUpdateNode>(local_tree, update_functor);
    }
}

std::optional<RankNeuronId> Octree::find_target_neuron(size_t src_neuron_id, const Vec3d& axon_pos_xyz, SignalType dendrite_type_needed) {
    OctreeNode* node_selected = nullptr;
    OctreeNode* root_of_subtree = root;

    while (true) {
        /**
		* Create vector with nodes that have at least one dendrite and are
		* precise enough given the position of an axon
		*/
        ProbabilitySubintervalVector vector = get_nodes_for_interval(axon_pos_xyz, root_of_subtree, dendrite_type_needed, naive_method);

        /**
		* Assign a probability to each node in the vector.
		* The probability for connecting to the same neuron (i.e., the axon's neuron) is set 0.
		* Nodes with 0 probability are removed.
		* The probabilities of all vector elements sum up to 1.
		*/
        const std::vector<double> prob = create_interval(src_neuron_id, axon_pos_xyz, dendrite_type_needed, vector);

        if (prob.empty()) {
            return {};
        }

        // Draw random number from [0,1]
        const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::Octree, 0.0, std::nextafter(1.0, Constants::eps));

        auto counter = 0;
        double sum_probabilities = 0.0;
        while (counter < prob.size()) {
            if (sum_probabilities >= random_number) {
                break;
            }

            sum_probabilities += prob[counter];
            counter++;
        }
        node_selected = vector[counter - 1]->get_octree_node();

        /**
		* Leave loop if no node was selected OR
		* the selected node is leaf node, i.e., contains normal neuron.
		*
		* No node is selected when all nodes in the interval, created in
		* get_nodes_for_interval(), have probability 0 to connect.
		*/
        const auto done = !node_selected->is_parent();

        // Update root of subtree
        root_of_subtree = node_selected;

        if (done) {
            break;
        }
    }

    RankNeuronId rank_neuron_id{ node_selected->get_rank(), node_selected->get_cell().get_neuron_id() };
    return rank_neuron_id;
}

const std::optional<OctreeNode*> Octree::do_random_experiment(OctreeNode *source, const std::vector<double> &atractiveness) {
    std::uniform_real_distribution<double> random_number_distribution(0.0, std::nextafter(1.0, 1.0 + Constants::eps));
    std::mt19937& random_number_generator = RandomHolder::RandomHolder::get_random_uniform_double(RandomHolderKey::Octree, 0.0, std::nextafter(1.0, Constants::eps));
    
    int vec_len = atractiveness.size();
    std::vector<double> intervals;
    intervals.reserve(vec_len+1);
    intervals[0]=0;
    double sum = 0;
    for (int i = 0; i < vec_len; i++)
    {
        sum = sum + atractiveness.at(i);
    }

   // RelearnException::check(temp,"The sum of all attractions was 0.");
    for (int i = 1; i < vec_len+1; i++)
    {
        intervals[i]= intervals[i-1]+ (atractiveness.at(i-1)/sum);
    }
   
    const auto random_number = random_number_distribution(random_number_generator);
    int i = 0;


    while (random_number > intervals[i+1] && i<=vec_len)
    {
        i++;
    }
    if (i>=vec_len+1)
    {
        return nullptr;
    }
    return source->get_from_interactionlist(i);
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
