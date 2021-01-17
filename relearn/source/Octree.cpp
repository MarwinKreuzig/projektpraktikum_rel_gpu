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

#include "Neurons.h"
#include "Partition.h"
#include "Random.h"
#include "RelearnException.h"

#include <mpi.h>

Octree::Octree()
    : random_number_generator(RandomHolder<Octree>::get_random_generator())
    , mpi_rma_node_allocator(MPIWrapper::mpi_rma_mem_allocator)
    , random_number_distribution(0.0, std::nextafter(1.0, 2.0)) {

    random_number_generator.seed(randomNumberSeeds::octree);
}

Octree::Octree(std::shared_ptr<Partition> part, const Parameters& params)
    : root_level(0)
    , acceptance_criterion(params.accept_criterion)
    , sigma(params.sigma)
    , naive_method(params.naive_method)
    , level_of_branch_nodes(part->get_level_of_subdomain_trees())
    , max_num_pending_vacant_axons(params.max_num_pending_vacant_axons)
    , mpi_rma_node_allocator(MPIWrapper::mpi_rma_mem_allocator)
    , random_number_generator(RandomHolder<Octree>::get_random_generator())
    , random_number_distribution(0.0, std::nextafter(1.0, 2.0)) {

    random_number_generator.seed(randomNumberSeeds::octree);

    Vec3d xyz_min;
    Vec3d xyz_max;
    std::tie(xyz_min, xyz_max) = part->get_simulation_box_size();

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

    stack.emplace(root, false, 0);

    while (!stack.empty()) {

        auto& elem = stack.top();
        const auto depth = static_cast<int>(elem.depth);

        // Visit node now
        if (elem.flag) {
            Vec3d xyz_min;
            Vec3d xyz_max;
            Vec3d xyz_pos;

            // Print node's address
            for (auto j = 0; j < depth; j++) {
                std::cout << " ";
            }
            std::cout << "Address: " << elem.ptr << "\n";

            // Print cell extent
            std::tie(xyz_min, xyz_max) = elem.ptr->cell.get_size();
            for (auto j = 0; j < depth; j++) {
                std::cout << " ";
            }

            std::cout << "Cell extent: (" << xyz_min.x << " .. " << xyz_max.x << ", "
                      << xyz_min.y << " .. " << xyz_max.y << ", "
                      << xyz_min.z << " .. " << xyz_max.z << ")\n";

            // Print neuron ID
            for (auto j = 0; j < depth; j++) {
                std::cout << " ";
            }
            std::cout << "Neuron ID: " << elem.ptr->cell.get_neuron_id() << "\n";

            // Print number of dendrites
            for (auto j = 0; j < depth; j++) {
                std::cout << " ";
            }
            std::cout << "Number dendrites (exc, inh): (" << elem.ptr->cell.get_neuron_num_dendrites_exc()
                      << ", " << elem.ptr->cell.get_neuron_num_dendrites_inh() << ")\n";

            // Print position DendriteType::EXCITATORY
            bool pos_valid = false;
            std::tie(xyz_pos, pos_valid) = elem.ptr->cell.get_neuron_position_exc();
            for (auto j = 0; j < depth; j++) {
                std::cout << " ";
            }
            std::cout << "Position exc: (" << xyz_pos.x << ", " << xyz_pos.y << ", " << xyz_pos.z << ") ";
            // Note if position is invalid
            if (!pos_valid) {
                std::cout << "-- invalid!";
            }
            std::cout << "\n";
            // Print position DendriteType::INHIBITORY
            std::tie(xyz_pos, pos_valid) = elem.ptr->cell.get_neuron_position_inh();
            for (auto j = 0; j < depth; j++) {
                std::cout << " ";
            }
            std::cout << "Position inh: (" << xyz_pos.x << ", " << xyz_pos.y << ", " << xyz_pos.z << ") ";
            // Note if position is invalid
            if (!pos_valid) {
                std::cout << "-- invalid!";
            }
            std::cout << "\n";
            std::cout << "\n";

            stack.pop();
        }
        // Visit children first
        else {
            elem.flag = true;

            for (auto j = 0; j < depth; j++) {
                std::cout << " ";
            }

            std::cout << "Child indices: ";
            int id = 0;
            for (auto& child : elem.ptr->children) {
                if (child != nullptr) {
                    std::cout << id << " ";
                }
                id++;
            }
            /**
			* Push in reverse order so that visiting happens in
			* increasing order of child indices
			*/
            for (auto it = elem.ptr->children.crbegin(); it != elem.ptr->children.crend(); ++it) {
                if (*it != nullptr) {
                    stack.emplace(*it, false, static_cast<size_t>(depth) + 1);
                }
            }
            std::cout << std::endl;
        }
    }
}

bool Octree::acceptance_criterion_test(const Vec3d& axon_pos_xyz,
    const OctreeNode* const node_with_dendrite,
    Cell::DendriteType dendrite_type_needed,
    bool naive_method,
    bool& has_vacant_dendrites) const /*noexcept*/ {

    // Use naive method
    if (naive_method) {
        // Accept leaf only
        const auto is_child = !node_with_dendrite->is_parent;
        return is_child;
    }

    has_vacant_dendrites = node_with_dendrite->cell.get_neuron_num_dendrites_for(dendrite_type_needed) != 0;

    // There are vacant dendrites available
    if (has_vacant_dendrites) {

        /**
		* Node is leaf node, i.e., not super neuron.
		* Thus the node is precise. Accept it.
		*/
        if (!node_with_dendrite->is_parent) {
            return true;
        }

        // Check distance between neuron with axon and neuron with dendrite
        Vec3d target_xyz;
        bool pos_valid = false;
        std::tie(target_xyz, pos_valid) = node_with_dendrite->cell.get_neuron_position_for(dendrite_type_needed);

        // NOTE: This assertion fails when considering inner nodes that don't have dendrites.
        RelearnException::check(pos_valid);

        // Calc Euclidean distance between source and target neuron
        const auto distance_vector = target_xyz - axon_pos_xyz;
        const auto distance = distance_vector.calculate_p_norm(2.0);

        const auto length = node_with_dendrite->cell.get_length();

        // Original Barnes-Hut acceptance criterion
        const auto ret_val = (length / distance) < acceptance_criterion;
        return ret_val;
    }

    return false;
}

void Octree::get_nodes_for_interval(
    const Vec3d& axon_pos_xyz,
    OctreeNode* root,
    Cell::DendriteType dendrite_type_needed,
    ProbabilitySubintervalList& list,
    bool naive_method) {

    /* Subtree is not empty AND (Dendrites are available OR We use naive method) */
    const auto flag = (root != nullptr) && (root->cell.get_neuron_num_dendrites_for(dendrite_type_needed) != 0 || naive_method);
    if (!flag) {
        return;
    }

    std::stack<OctreeNode*> stack;
    std::array<OctreeNode*, Constants::number_oct> local_children = { nullptr };

    const MPI_Aint* base_pointers = mpi_rma_node_allocator.get_base_pointers();

    /**
	* The root node is parent (i.e., contains a super neuron) and thus cannot be the target neuron.
	* So, start considering its children.
	*/
    if (root->is_parent) {
        // Node is owned by this rank
        if (node_is_local(*root)) {
            // Push root's children onto stack
            for (auto it = root->children.crbegin(); it != root->children.crend(); ++it) {
                if (*it != nullptr) {
                    stack.push(*it);
                }
            }
        }
        // Node is owned by different rank
        else {
            const auto target_rank = root->rank;
            NodesCacheKey rank_addr_pair;
            rank_addr_pair.first = target_rank;

            // Start access epoch to remote rank
            MPIWrapper::lock_window(target_rank, MPI_Locktype::shared);

            // Fetch remote children if they exist
            for (auto i = 7; i >= 0; i--) {
                if (nullptr == root->children[i]) {
                    local_children[i] = nullptr;
                    continue;
                }

                rank_addr_pair.second = root->children[i];

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
                    ret.first->second = mpi_rma_node_allocator.newObject();
                    auto* local_child_addr = ret.first->second;

                    // Calc displacement from absolute address
                    const auto target_child_displ = MPI_Aint(root->children[i]) - base_pointers[target_rank];

                    MPIWrapper::get(local_child_addr, target_rank, target_child_displ, mpi_rma_node_allocator.mpi_window);
                }

                // Remember address of node
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
		* Without pushing root onto the stack, it would not make it into the "list" of nodes.
		*/
        stack.push(root);
    }

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
        const auto accept = acceptance_criterion_test(axon_pos_xyz, stack_elem, dendrite_type_needed, naive_method,
            has_vacant_dendrites);
        if (accept) {
            //std::cout << "accepted: " << stack_elem->cell.get_neuron_id() << std::endl;
            // Insert node into list
            auto list_elem = std::make_shared<ProbabilitySubinterval>();
            list_elem->ptr = stack_elem;
            list.push_back(list_elem);
        } else if (has_vacant_dendrites || naive_method) {
            // Node is owned by this rank
            if (node_is_local(*stack_elem)) {
                // Push node's children onto stack
                for (auto it = stack_elem->children.crbegin(); it != stack_elem->children.crend(); ++it) {
                    if (*it != nullptr) {
                        stack.push(*it);
                    }
                }
            }
            // Node is owned by different rank
            else {
                const auto target_rank = stack_elem->rank;
                NodesCacheKey rank_addr_pair;
                rank_addr_pair.first = target_rank;

                // Start access epoch to remote rank
                MPIWrapper::lock_window(target_rank, MPI_Locktype::shared);

                // Fetch remote children if they exist
                for (auto i = 7; i >= 0; i--) {
                    if (nullptr == stack_elem->children[i]) {
                        local_children[i] = nullptr;
                        continue;
                    }

                    rank_addr_pair.second = stack_elem->children[i];

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
                        ret.first->second = mpi_rma_node_allocator.newObject();
                        auto* local_child_addr = ret.first->second;

                        // Calc displacement from absolute address
                        const auto target_child_displ = MPI_Aint(stack_elem->children[i]) - base_pointers[target_rank];

                        MPIWrapper::get(local_child_addr, target_rank, target_child_displ, mpi_rma_node_allocator.mpi_window);
                    }

                    // Remember local address of node
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
            } // Node owned by different rank
        } // Acceptance criterion rejected
    } // while
}

void Octree::create_interval(size_t src_neuron_id, const Vec3d& axon_pos_xyz, Cell::DendriteType dendrite_type_needed, ProbabilitySubintervalList& list) const {
    // Does list contain nodes?
    if (list.empty()) {
        return;
    }

    double sum = 0.0;
    for (auto& prob_subinterval : list) {
        const auto prob = calc_attractiveness_to_connect(src_neuron_id, axon_pos_xyz, *(prob_subinterval->ptr), dendrite_type_needed);
        prob_subinterval->probability = prob;
        sum += prob;
    }

    /**
	* Make sure that we don't divide by 0 in case
	* all probabilities from above are 0.
	*/
    sum = (sum == 0.0) ? 1.0 : sum;

    // Norm the values to [0,1] and thus get probabilities
    for (auto it = list.begin(); it != list.end();) {
        (*it)->probability /= sum;

        // Remove node that has probability 0
        if ((*it)->probability == 0.0) { // We want exact comparison of double to constant 0 here
            it = list.erase(it); // "it" points now to successor element
                //std::cout << __func__ << ": probability == 0, node removed" << std::endl;
        } else {
            it++;
        }
    }
}

double Octree::calc_attractiveness_to_connect(
    size_t src_neuron_id,
    const Vec3d& axon_pos_xyz,
    const OctreeNode& node_with_dendrite,
    Cell::DendriteType dendrite_type_needed) const /*noexcept*/ {

    /**
	* If the axon's neuron itself is considered as target neuron, set attractiveness to 0 to avoid forming an autapse (connection to itself).
	* This can be done as the axon's neuron cells are always resolved until the normal (vs. super) axon's neuron is reached.
	* That is, the dendrites of the axon's neuron are not included in any super neuron considered.
	* However, this only works under the requirement that "acceptance_criterion" is <= 0.5.
	*/
    if ((!node_with_dendrite.is_parent) && (src_neuron_id == node_with_dendrite.cell.get_neuron_id())) {
        return 0.0;
    }

    Vec3d target_xyz;
    bool pos_valid = false;
    std::tie(target_xyz, pos_valid) = node_with_dendrite.cell.get_neuron_position_for(dendrite_type_needed);
    RelearnException::check(pos_valid);

    const auto num_dendrites = node_with_dendrite.cell.get_neuron_num_dendrites_for(dendrite_type_needed);

    const auto position_diff = target_xyz - axon_pos_xyz;
    const auto eucl_length = position_diff.calculate_p_norm(2.0);
    const auto numerator = pow(eucl_length, 2.0);

    // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
    const auto ret_val = (num_dendrites * exp(-numerator / (sigma * sigma)));
    return ret_val;
}

OctreeNode* Octree::select_subinterval(const ProbabilitySubintervalList& list) {
    // Does list contain nodes?
    if (list.empty()) {
        return nullptr;
    }

    // Draw random number from [0,1]
    const double random_number = random_number_distribution(random_number_generator);

    /**
	* Also check for it != list.end() to account for that, due to numeric inaccuracies in summation,
	* it might happen that random_number > sum_probabilities in the end
	*/
    auto it = list.cbegin();
    double sum_probabilities = (*it)->probability;
    it++; // Point to second element
    while (random_number > sum_probabilities && it != list.cend()) {
        sum_probabilities += (*it)->probability;
        it++;
    }
    it--; // Undo it++ before or in loop to get correct subinterval
    return (*it)->ptr;
}

bool Octree::node_is_local(const OctreeNode& node) /*noexcept*/ {
    return node.rank == MPIWrapper::my_rank;
}

void Octree::append_node(OctreeNode* node, ProbabilitySubintervalList& list) {
    list.emplace_back(std::make_shared<ProbabilitySubinterval>(node));
}

void Octree::append_children(OctreeNode* node, ProbabilitySubintervalList& list, AccessEpochsStarted& epochs_started) {
    // Node is local
    if (node_is_local(*node)) {
        // Append all children != nullptr
        for (auto& child : node->children) {
            if (child != nullptr) {
                list.emplace_back(std::make_shared<ProbabilitySubinterval>(child));
            }
        }

        return;
    }
    // Node is remote

    const int target_rank = node->rank;
    NodesCacheKey rank_addr_pair;
    rank_addr_pair.first = target_rank;

    // Start access epoch if necessary
    if (!epochs_started[target_rank]) {
        // Start access epoch to remote rank
        MPIWrapper::lock_window(target_rank, MPI_Locktype::shared);
        epochs_started[target_rank] = true;
    }

    for (auto& child : node->children) {
        if (child != nullptr) {
            rank_addr_pair.second = child;

            std::pair<NodesCacheKey, NodesCacheValue> cache_key_val_pair;
            cache_key_val_pair.first = rank_addr_pair;
            cache_key_val_pair.second = nullptr;

            // Get cache entry for "cache_key_val_pair"
            std::pair<NodesCache::iterator, bool> ret = remote_nodes_cache.insert(cache_key_val_pair);

            // TODO(fabian): Kill pointer here
            auto prob_sub = std::make_shared<ProbabilitySubinterval>();
            // Cache entry just inserted as it was not in cache
            // So, we still need to init the entry by fetching
            // from the target rank
            if (ret.second) {
                // Create new object which contains the remote node's information
                ret.first->second = prob_sub->ptr = mpi_rma_node_allocator.newObject();

                const MPI_Aint* base_pointers = mpi_rma_node_allocator.get_base_pointers();
                // Calc displacement from absolute address
                const auto target_child_displ = MPI_Aint(child - base_pointers[target_rank]);

                MPIWrapper::get(prob_sub->ptr, target_rank, target_child_displ, mpi_rma_node_allocator.mpi_window);

                prob_sub->mpi_request = MPIWrapper::get_non_null_request();
                prob_sub->request_rank = target_rank;
            } else {
                prob_sub->ptr = ret.first->second;
            }
            list.emplace_back(prob_sub);
        }
    } // for all children
}

void Octree::find_target_neurons(MapSynapseCreationRequests& map_synapse_creation_requests_outgoing,
    const Neurons& neurons) {

    VacantAxonList vacant_axons;
    bool axon_added = false;

    AccessEpochsStarted access_epochs_started(MPIWrapper::num_ranks, false);

    do {
        axon_added = false;

        Cell::DendriteType dendrite_type_needed;

        size_t source_neuron_id { Constants::uninitialized };
        Vec3d xyz_pos;

        bool ret = false;

        std::tie(ret, source_neuron_id, xyz_pos, dendrite_type_needed) = neurons.get_vacant_axon();
        // Append one vacant axon to list of pending axons if too few are pending
        if ((vacant_axons.size() < max_num_pending_vacant_axons) && ret) {
            auto axon = std::make_shared<VacantAxon>(source_neuron_id, xyz_pos, dendrite_type_needed);

            if (root->is_parent) {
                append_children(root, axon->nodes_to_visit, access_epochs_started);
            } else {
                append_node(root, axon->nodes_to_visit);
            }

            vacant_axons.emplace_back(std::move(axon));

            axon_added = true;
        }

        // Vacant axons exist
        if (!vacant_axons.empty()) {
            std::shared_ptr<VacantAxon> axon = vacant_axons.front();
            bool delete_axon = false;

            // Go through all nodes to visit of this axon
            for (size_t i = 0; i < axon->nodes_to_visit.size(); i++) {
                auto& node_to_visit = axon->nodes_to_visit.front();

                // Node is from different rank and MPI request still open
                // So complete getting the contents of the remote node
                MPIWrapper::wait_request(node_to_visit->mpi_request);

                bool has_vacant_dendrites = false;
                const auto accept = acceptance_criterion_test(axon->xyz_pos, node_to_visit->ptr, axon->dendrite_type_needed, false, has_vacant_dendrites);
                // Check if the node is accepted and if yes, append it to nodes_accepted
                if (accept) {
                    axon->nodes_accepted.emplace_back(node_to_visit);
                } else {
                    // Node was rejected only because it's too close
                    if (has_vacant_dendrites) {
                        append_children(node_to_visit->ptr, axon->nodes_to_visit, access_epochs_started);
                    }
                }

                // Node is visited now, so remove it
                axon->nodes_to_visit.pop_front();
            }

            // No nodes to visit anymore
            if (axon->nodes_to_visit.empty()) {
                /**
				 * Assign a probability to each node in the nodes_accepted list.
				 * The probability for connecting to the same neuron (i.e., the axon's neuron) is set 0.
				 * Nodes with 0 probability are removed.
				 * The probabilities of all list elements sum up to 1.
				 */
                create_interval(axon->neuron_id, axon->xyz_pos, axon->dendrite_type_needed, axon->nodes_accepted);

                /**
				 * Select node with target neuron
				 */
                auto* node_selected = select_subinterval(axon->nodes_accepted);

                // Clear nodes_accepted list for next interval creation
                axon->nodes_accepted.clear();

                // Now nodes_accepted and nodes_to_visit are empty

                // Node was selected
                if (nullptr != node_selected) {
                    // Selected node is parent. A parent cannot be a target neuron.
                    // So append its children to nodes_to_visit
                    if (node_selected->is_parent) {
                        append_children(node_selected, axon->nodes_to_visit, access_epochs_started);
                    } else {
                        // Target neuron found
                        // Create synapse creation request for the target neuron
                        map_synapse_creation_requests_outgoing[node_selected->rank].append(
                            axon->neuron_id,
                            node_selected->cell.get_neuron_id(),
                            axon->dendrite_type_needed);
                        delete_axon = true;
                    }
                }
                // No node selected
                else {
                    // No target neuron found for axon
                    delete_axon = true;
                }
            }

            // Remove current axon from front of list
            vacant_axons.pop_front();

            if (!delete_axon) {
                vacant_axons.emplace_back(std::move(axon));
            }
        }
    } while (axon_added || !vacant_axons.empty());

    // Complete all started access epochs
    for (auto i = 0; i < access_epochs_started.size(); i++) {
        if (access_epochs_started[i]) {
            MPIWrapper::unlock_window(i);
        }
    }
}

// Insert neuron into the tree
OctreeNode* Octree::insert(const Vec3d& position, size_t neuron_id, int rank) {
    // Create new tree node for the neuron
    OctreeNode* new_node = mpi_rma_node_allocator.newObject(); // new OctreeNode();
    RelearnException::check(new_node != nullptr);

    new_node->cell.set_neuron_position(position, true);
    new_node->cell.set_neuron_id(neuron_id);
    new_node->rank = rank;

    // Tree is empty
    if (nullptr == root) {
        // Init cell size with simulation box size
        new_node->cell.set_size(this->xyz_min, this->xyz_max);

        // Init root with tree's root level
        new_node->level = root_level;
        root = new_node;

        return new_node;
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
        my_idx = curr->cell.get_octant_for_position(position);

        prev = curr;
        curr = curr->children[my_idx];
    }

    RelearnException::check(prev != nullptr);

    /**
	* Found my octant, but
	* I'm the very first child of that node.
	* I.e., the node is a leaf.
	*/
    if (!prev->is_parent) {
        do {
            /**
			* Make node containing my octant a parent by
			* adding neuron in this node as child node
			*/

            // Determine octant for neuron
            idx = prev->cell.get_neuron_octant();
            OctreeNode* new_node = mpi_rma_node_allocator.newObject(); // new OctreeNode();
            prev->children[idx] = new_node;

            /**
			* Init this new node properly
			*/
            // Cell size
            Vec3d xyz_min;
            Vec3d xyz_max;
            std::tie(xyz_min, xyz_max) = prev->cell.get_size_for_octant(idx);

            new_node->cell.set_size(xyz_min, xyz_max);

            // Neuron position
            Vec3d inner_pos;
            bool valid_pos = false;
            std::tie(inner_pos, valid_pos) = prev->cell.get_neuron_position();
            new_node->cell.set_neuron_position(inner_pos, valid_pos);

            // Neuron ID
            const auto prev_neuron_id = prev->cell.get_neuron_id();
            new_node->cell.set_neuron_id(prev_neuron_id);
            /**
			* Set neuron ID of parent (inner node) to uninitialized.
			* It is not used for inner nodes.
			*/
            prev->cell.set_neuron_id(Constants::uninitialized);
            prev->is_parent = true; // Mark node as parent

            // MPI rank who owns this node
            new_node->rank = prev->rank;

            // New node is one level below
            new_node->level = prev->level + 1;

            // Determine my octant
            my_idx = prev->cell.get_octant_for_position(position);

            if (my_idx == idx) {
                prev = new_node;
            }
        } while (my_idx == idx);
    }

    /**
	* Found my position in children array,
	* add myself to the array now
	*/
    prev->children[my_idx] = new_node;
    new_node->level = prev->level + 1; // Now we know level of me

    Vec3d xyz_min;
    Vec3d xyz_max;
    std::tie(xyz_min, xyz_max) = prev->cell.get_size_for_octant(my_idx);
    prev->children[my_idx]->cell.set_size(xyz_min, xyz_max);

    return new_node;
}

// Insert an octree node with its subtree into the tree
void Octree::insert(OctreeNode* node_to_insert) {
    const auto target_level = node_to_insert->level;

    // Tree is empty
    if (nullptr == root) {
        // Node should become root of the tree
        if (root_level == target_level) {
            root = node_to_insert;
            // NOTE: We assume that the tree's and the node's
            // box size are the same. That's why we don't set the tree's
            // box size explicitly here.

            //LogMessages::print_debug("ROOT: Me as root inserted.");

            return;
        }
        // Create tree's root

        // Create root node
        root = mpi_rma_node_allocator.newObject();

        // Init octree node
        root->rank = MPIWrapper::my_rank;
        root->level = root_level;
        root->is_parent = true; // node will become parent

        // Init cell in octree node
        // cell size becomes tree's box size
        root->cell.set_size(this->xyz_min, this->xyz_max);
        root->cell.set_neuron_id(Constants::uninitialized);

        //LogMessages::print_debug("ROOT: new node as root inserted.");
    }

    auto* curr = root;
    auto next_level = curr->level + 1; // next_level is the current level we consider for inserting the node
        // It's called next_level as it is the next level below the current node
        // "curr" in the tree

    unsigned char my_idx = 0;
    // Calc midpoint of node's cell
    Vec3d cell_xyz_min;
    Vec3d cell_xyz_max;
    std::tie(cell_xyz_min, cell_xyz_max) = node_to_insert->cell.get_size();
    const double cell_length_half = node_to_insert->cell.get_length() / 2;
    const auto cell_xyz_mid = cell_xyz_min + cell_length_half;

    while (true) {
        /**
		* My parent already exists.
		* Calc which child to follow, i.e., determine
		* my octant (index in the children array)
		* based on the midpoint of my cell
		*/
        my_idx = curr->cell.get_octant_for_position(cell_xyz_mid);

        // Target level reached, so insert me
        if (next_level == target_level) {
            //LogMessages::print_debug("Target level reached.");

            // Make sure that no other node is already
            // on my index in the children array
            //
            // NOTE:
            // This assertion is not valid anymore as the same branch nodes
            // are inserted repeatedly at the same position
            // RelearnException::check(curr->children[my_idx] == nullptr);

            curr->children[my_idx] = node_to_insert;

            //LogMessages::print_debug("  Target level reached... inserted me");
            break;
        }
        // Target level not yet reached

        //LogMessages::print_debug("Target level not yet reached.");

        // A node exists on my index in the
        // children array, so follow this node.
        if (curr->children[my_idx] != nullptr) {
            curr = curr->children[my_idx];
            //LogMessages::print_debug("  I follow node on my index.");
        }
        // New node must be created which
        // I can then follow
        else {
            //LogMessages::print_debug("  New node must be created which I can then follow.");
            Vec3d new_node_xyz_min;
            Vec3d new_node_xyz_max;

            //LogMessages::print_debug("    Trying to allocate node.");
            // Create node
            auto* new_node = mpi_rma_node_allocator.newObject();
            //LogMessages::print_debug("    Node allocated.");

            // Init octree node
            new_node->rank = MPIWrapper::my_rank;
            new_node->level = next_level;
            new_node->is_parent = true; // node will become parent

            // Init cell in octree node
            // cell size becomes size of new node's octant
            std::tie(new_node_xyz_min, new_node_xyz_max) = curr->cell.get_size_for_octant(my_idx);
            new_node->cell.set_size(new_node_xyz_min, new_node_xyz_max);
            new_node->cell.set_neuron_id(Constants::uninitialized);

            curr->children[my_idx] = new_node;
            curr = new_node;
        }
        next_level++;

    } // while
}

void Octree::insert_local_tree(Octree* node_to_insert) {

    OctreeNode* local_root = node_to_insert->get_root();
    if (local_root == nullptr) {
        std::stringstream s;
        s << "Local tree is empty, probably because the corresponding subdomain contains no neuron. "
          << "Currently, it is a requirement that every subdomain contains at least one neuron.\n";
        LogMessages::print_error(s.str().c_str());
        std::abort();
    }

    insert(local_root);
    local_trees.emplace_back(node_to_insert);
}

void Octree::print() {
    postorder_print();
}

void Octree::free() {
    // Provide allocator so that it can be used to free memory again
    const FunctorFreeNode free_node(mpi_rma_node_allocator);

    // The functor containing the visit function is of type FunctorFreeNode
    tree_walk_postorder<FunctorFreeNode>(free_node);
}

void Octree::update(const std::vector<double>& dendrites_exc_cnts, const std::vector<double>& dendrites_exc_connected_cnts,
    const std::vector<double>& dendrites_inh_cnts, const std::vector<double>& dendrites_inh_connected_cnts, size_t num_neurons) {
    // Init parameters to be used in function object
    const FunctorUpdateNode update_node(dendrites_exc_cnts, dendrites_exc_connected_cnts,
        dendrites_inh_cnts, dendrites_inh_connected_cnts, num_neurons);

    // The functor containing the visit function is of type FunctorUpdateNode
    tree_walk_postorder<FunctorUpdateNode>(update_node);
}

// The caller must ensure that only inner nodes are visited. "max_level" must be chosen correctly for this
void Octree::update_from_level(size_t max_level) {
    std::vector<double> dendrites_exc_cnts;
    std::vector<double> dendrites_exc_connected_cnts;
    std::vector<double> dendrites_inh_cnts;
    std::vector<double> dendrites_inh_connected_cnts;

    const FunctorUpdateNode update_node(dendrites_exc_cnts, dendrites_exc_connected_cnts, dendrites_inh_cnts, dendrites_inh_connected_cnts, 0);

    /**
	* NOTE: It *must* be ensured that in tree_walk_postorder() only inner nodes
	* are visited as update_node cannot update leaf nodes here
	*/

    // The functor containing the visit function is of type FunctorUpdateNode
    tree_walk_postorder<FunctorUpdateNode>(update_node, max_level);
}

bool Octree::find_target_neuron(size_t src_neuron_id, const Vec3d& axon_pos_xyz, Cell::DendriteType dendrite_type_needed, size_t& target_neuron_id, int& target_rank) {
    OctreeNode* node_selected = nullptr;
    OctreeNode* root_of_subtree = root;

    while (true) {
        /**
		* Create list with nodes that have at least one dendrite and are
		* precise enough given the position of an axon
		*/
        ProbabilitySubintervalList list;
        get_nodes_for_interval(axon_pos_xyz, root_of_subtree, dendrite_type_needed, list, naive_method);

        /**
		* Assign a probability to each node in the list.
		* The probability for connecting to the same neuron (i.e., the axon's neuron) is set 0.
		* Nodes with 0 probability are removed.
		* The probabilities of all list elements sum up to 1.
		*/
        create_interval(src_neuron_id, axon_pos_xyz, dendrite_type_needed, list);

        /**
		* Select node with target neuron
		*/
        node_selected = select_subinterval(list);

        /**
		* Leave loop if no node was selected OR
		* the selected node is leaf node, i.e., contains normal neuron.
		*
		* No node is selected when all nodes in the interval, created in
		* get_nodes_for_interval(), have probability 0 to connect.
		*/
        const auto done = (nullptr == node_selected) || (!node_selected->is_parent);

        // Update root of subtree
        root_of_subtree = node_selected;

        if (done) {
            break;
        }
    }

    const auto found = nullptr != node_selected;

    // Return neuron ID and rank of target neuron
    if (found) {
        target_neuron_id = node_selected->cell.get_neuron_id();
        target_rank = node_selected->rank;
    }

    return found;
}

void Octree::empty_remote_nodes_cache() {
    for (auto& remode_node_in_cache : remote_nodes_cache) {
        mpi_rma_node_allocator.deleteObject(remode_node_in_cache.second);
    }

    remote_nodes_cache.clear();
}
