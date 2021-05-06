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
#include "../util/Random.h"
#include "../util/RelearnException.h"

#include <sstream>

Octree::Octree(const Vec3d& xyz_min, const Vec3d& xyz_max, size_t level_of_branch_nodes)
    : root_level(0)
    , naive_method(acceptance_criterion == 0.0)
    , level_of_branch_nodes(level_of_branch_nodes) {

    set_size(xyz_min, xyz_max);
}

Octree::Octree(const Vec3d& xyz_min, const Vec3d& xyz_max, size_t level_of_branch_nodes, double acceptance_criterion, double sigma)
    : root_level(0)
    , naive_method(acceptance_criterion == 0.0)
    , level_of_branch_nodes(level_of_branch_nodes) {

    set_acceptance_criterion(acceptance_criterion);
    set_probability_parameter(sigma);
    set_size(xyz_min, xyz_max);
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

// Insert neuron into the tree
OctreeNode* Octree::insert(const Vec3d& position, size_t neuron_id, int rank) {
    // Create new tree node for the neuron
    OctreeNode* new_node_to_insert = MPIWrapper::new_octree_node();
    RelearnException::check(new_node_to_insert != nullptr, "new_node_to_insert is nullptr");

    // Tree is empty
    if (nullptr == root) {
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

    unsigned char my_idx = 0;
    unsigned char idx = 0;

    OctreeNode* prev = nullptr;
    OctreeNode* curr = root;
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

// Insert an octree node with its subtree into the tree
void Octree::insert(OctreeNode* node_to_insert) {
    RelearnException::check(node_to_insert != nullptr, "In Octree::insert, node_to_insert was nullptr");
    // NOLINTNEXTLINE
    const auto target_level = node_to_insert->get_level();

    // Tree is empty
    if (nullptr == root) {
        // Node should become root of the tree
        if (root_level == target_level) {
            root = node_to_insert;
            // NOTE: We assume that the tree's and the node's
            // box size are the same. That's why we don't set the tree's
            // box size explicitly here.

            return;
        }
        // Create tree's root

        // Create root node
        root = MPIWrapper::new_octree_node();

        // Init octree node
        root->set_rank(MPIWrapper::get_my_rank());
        root->set_level(root_level);
        root->set_parent(); // node will become parent

        // Init cell in octree node
        // cell size becomes tree's box size
        root->set_cell_size(this->xyz_min, this->xyz_max);
        root->set_cell_neuron_id(Constants::uninitialized);
    }

    auto* curr = root;
    auto next_level = curr->get_level() + 1; // next_level is the current level we consider for inserting the node
        // It's called next_level as it is the next level below the current node
        // "curr" in the tree

    unsigned char my_idx = 0;
    // Calc midpoint of node's cell
    Vec3d cell_xyz_min;
    Vec3d cell_xyz_max;
    std::tie(cell_xyz_min, cell_xyz_max) = node_to_insert->get_cell().get_size();
    const auto cell_xyz_mid = (cell_xyz_min + cell_xyz_max) / 2;

    while (true) {
        /**
		* My parent already exists.
		* Calc which child to follow, i.e., determine
		* my octant (index in the children array)
		* based on the midpoint of my cell
		*/
        my_idx = curr->get_cell().get_octant_for_position(cell_xyz_mid);

        // Target level reached, so insert me
        if (next_level == target_level) {
            // Make sure that no other node is already
            // on my index in the children array

            curr->set_child(node_to_insert, my_idx);
            break;
        }
        // Target level not yet reached

        // A node exists on my index in the
        // children array, so follow this node.
        if (curr->get_child(my_idx) != nullptr) {
            curr = curr->get_child(my_idx);
        }
        // New node must be created which
        // I can then follow
        else {
            Vec3d new_node_xyz_min;
            Vec3d new_node_xyz_max;

            // Create node
            auto* new_node = MPIWrapper::new_octree_node();

            // Init octree node
            new_node->set_rank(MPIWrapper::get_my_rank());
            new_node->set_level(next_level);
            new_node->set_parent(); // node will become parent

            // Init cell in octree node
            // cell size becomes size of new node's octant
            std::tie(new_node_xyz_min, new_node_xyz_max) = curr->get_cell().get_size_for_octant(my_idx);
            new_node->set_cell_size(new_node_xyz_min, new_node_xyz_max);
            new_node->set_cell_neuron_id(Constants::uninitialized);

            curr->set_child(new_node, my_idx);
            curr = new_node;
        }
        next_level++;

    } // while
}

void Octree::insert_local_tree(Octree* node_to_insert) {
    OctreeNode* local_root = node_to_insert->get_root();
    RelearnException::check(local_root != nullptr, "Local tree is empty, probably because the corresponding subdomain contains no neuron.");

    insert(local_root);
    local_trees.emplace_back(node_to_insert);
}

void Octree::print() {
    postorder_print();
}

void Octree::free() {
    // Provide allocator so that it can be used to free memory again
    const FunctorFreeNode free_node{};

    // The functor containing the visit function is of type FunctorFreeNode
    tree_walk_postorder<FunctorFreeNode>(this, free_node);
}

// The caller must ensure that only inner nodes are visited. "max_level" must be chosen correctly for this
void Octree::update_from_level(size_t max_level) {
    std::vector<double> dendrites_exc_cnts;
    std::vector<unsigned int> dendrites_exc_connected_cnts;
    std::vector<double> dendrites_inh_cnts;
    std::vector<unsigned int> dendrites_inh_connected_cnts;

    const FunctorUpdateNode update_node(dendrites_exc_cnts, dendrites_exc_connected_cnts, dendrites_inh_cnts, dendrites_inh_connected_cnts, 0);

    /**
	* NOTE: It *must* be ensured that in tree_walk_postorder() only inner nodes
	* are visited as update_node cannot update leaf nodes here
	*/

    // The functor containing the visit function is of type FunctorUpdateNode
    tree_walk_postorder<FunctorUpdateNode>(this, update_node, max_level);
}

void Octree::update_local_trees(const SynapticElements& dendrites_exc, const SynapticElements& dendrites_inh, size_t num_neurons) {
    const auto& de_ex_cnt = dendrites_exc.get_cnts();
    const auto& de_ex_conn_cnt = dendrites_exc.get_connected_cnts();
    const auto& de_in_cnt = dendrites_inh.get_cnts();
    const auto& de_in_conn_cnt = dendrites_inh.get_connected_cnts();

    for (auto* local_tree : local_trees) {
        const FunctorUpdateNode update_node(de_ex_cnt, de_ex_conn_cnt, de_in_cnt, de_in_conn_cnt, num_neurons);

        // The functor containing the visit function is of type FunctorUpdateNode
        tree_walk_postorder<FunctorUpdateNode>(local_tree, update_node);
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

void Octree::empty_remote_nodes_cache() {
    for (auto& remode_node_in_cache : remote_nodes_cache) {
        MPIWrapper::delete_octree_node(remode_node_in_cache.second);
    }

    remote_nodes_cache.clear();
}
