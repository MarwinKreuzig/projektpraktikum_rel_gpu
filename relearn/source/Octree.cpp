/*
 * File:   Octree.cpp
 * Author: rinke
 *
 * Created on October 10, 2014
 */

 /*********************************************************************************
  * NOTE: We include Neurons.h here as the class Octree uses types from Neurons.h *
  * Neurons.h also includes Octree.h as it uses it too                            *
  *********************************************************************************/
#include "Neurons.h"
#include "randomNumberSeeds.h"
#include "Random.h"

Octree::Octree() :
	root(nullptr),
	root_level(0),
	no_free_in_destructor(0),
	level_of_branch_nodes(-1),
	random_number_distribution(0.0, std::nextafter(1.0, 2.0)),
	random_number_generator(RandomHolder<Octree>::get_random_generator()) {

	random_number_generator.seed(randomNumberSeeds::octree);
}

Octree::Octree(const Partition& part, const Parameters& params) :
	root(nullptr),
	root_level(0),
	no_free_in_destructor(0),
	level_of_branch_nodes(part.get_level_of_subdomain_trees()),
	acceptance_criterion(params.accept_criterion),
	sigma(params.sigma),
	naive_method(params.naive_method),
	max_num_pending_vacant_axons(params.max_num_pending_vacant_axons),
	random_number_distribution(0.0, std::nextafter(1.0, 2.0)),
	random_number_generator(RandomHolder<Octree>::get_random_generator()) {

	random_number_generator.seed(randomNumberSeeds::octree);

	Vec3d xyz_min, xyz_max;
	part.get_simulation_box_size(xyz_min, xyz_max);

	set_size(xyz_min, xyz_max);
}

Octree::~Octree() {
	if (!no_free_in_destructor) {
		// Free all nodes
		free();
	}
}

void Octree::append_node(OctreeNode* node, ProbabilitySubintervalList& list) {
	list.push_back(new ProbabilitySubinterval(node));
}

void Octree::append_children(OctreeNode* node, ProbabilitySubintervalList& list, AccessEpochsStarted& epochs_started) {
	// Node is local
	if (node_is_local(*node)) {
		// Append all children != nullptr
		for (size_t j = 0; j < 8; j++) {
			if (node->children[j]) {
				list.push_back(new ProbabilitySubinterval(node->children[j]));
			}
		}
	}
	// Node is remote
	else {
		MPI_Aint target_child_displ;
		NodesCacheKey rank_addr_pair;
		std::pair<NodesCache::iterator, bool> ret;
		std::pair<NodesCacheKey, NodesCacheValue> cache_key_val_pair;
		const MPI_Aint* base_pointers = mpi_rma_node_allocator->get_base_pointers();
		int target_rank;

		target_rank = node->rank;
		rank_addr_pair.first = target_rank;

		// Start access epoch if necessary
		if (!epochs_started[target_rank]) {
			// Start access epoch to remote rank
			//std::cout << MPIInfos::my_rank << ": Epoch started to: " << target_rank << std::endl;
			MPI_Win_lock(MPI_LOCK_SHARED, target_rank, MPI_MODE_NOCHECK, mpi_rma_node_allocator->mpi_window);
			epochs_started[target_rank] = true;
		}

		for (size_t j = 0; j < 8; j++) {
			if (node->children[j]) {
				// Create new subinterval
				ProbabilitySubinterval* child = new ProbabilitySubinterval();

				rank_addr_pair.second = node->children[j];
				cache_key_val_pair.first = rank_addr_pair;
				cache_key_val_pair.second = nullptr;

				// Get cache entry for "cache_key_val_pair"
				ret = remote_nodes_cache.insert(cache_key_val_pair);

				// Cache entry just inserted as it was not in cache
				// So, we still need to init the entry by fetching
				// from the target rank
				if (ret.second == true) {
					// Create new object which contains the remote node's information
					ret.first->second = child->ptr = mpi_rma_node_allocator->newObject();

					// Calc displacement from absolute address
					target_child_displ = (MPI_Aint)(node->children[j]) - base_pointers[target_rank];

					MPI_Get((char*)(child->ptr), sizeof(OctreeNode), MPI_CHAR,
						target_rank, target_child_displ, sizeof(OctreeNode), MPI_CHAR,
						mpi_rma_node_allocator->mpi_window);
					child->mpi_request = (MPI_Request)!MPI_REQUEST_NULL;
					child->request_rank = target_rank;
				}
				else {
					child->ptr = ret.first->second;
				}
				list.push_back(child);
			}
		} // for all children
	} // node is remote
}

void Octree::find_target_neurons(MapSynapseCreationRequests& map_synapse_creation_requests_outgoing,
	Neurons<NeuronModels, SynapticElements, SynapticElements, SynapticElements>& neurons) {
	VacantAxonList vacant_axons;
	size_t source_neuron_id;
	double xyz_pos[3];
	Cell::DendriteType dendrite_type_needed;
	NodesCacheKey rank_addr_pair;
	std::pair<NodesCacheKey, NodesCacheValue> cache_key_val_pair;
	const MPI_Aint* base_pointers = mpi_rma_node_allocator->get_base_pointers();
	bool axon_added, has_vacant_dendrites;
	std::list<ProbabilitySubinterval*>::iterator it;
	OctreeNode* node_selected;

	AccessEpochsStarted access_epochs_started(MPIInfos::num_ranks, false);

	do {
		axon_added = false;

		// Append one vacant axon to list of pending axons if too few are pending
		if ((vacant_axons.size() < max_num_pending_vacant_axons) &&
			(neurons.get_vacant_axon(source_neuron_id, xyz_pos, dendrite_type_needed))) {
			VacantAxon& axon = *(new VacantAxon(source_neuron_id, xyz_pos[0], xyz_pos[1], xyz_pos[2], dendrite_type_needed));
			vacant_axons.push_back(&axon);

			if (root->is_parent) {
				append_children(root, axon.nodes_to_visit, access_epochs_started);
			}
			else {
				append_node(root, axon.nodes_to_visit);
			}

			axon_added = true;
		}

		// Vacant axons exist
		if (!vacant_axons.empty()) {
			VacantAxon& axon = *(vacant_axons.front());
			bool delete_axon = false;

			// Go through all nodes to visit of this axon
			for (size_t i = 0; i < axon.nodes_to_visit.size(); i++) {
				ProbabilitySubinterval& node_to_visit = *(axon.nodes_to_visit.front());

				// Node is from different rank and MPI request still open
				// So complete getting the contents of the remote node
				if (MPI_REQUEST_NULL != node_to_visit.mpi_request) {
					MPI_Wait(&node_to_visit.mpi_request, MPI_STATUS_IGNORE);
				}

				// Check if the node is accepted and if yes, append it to nodes_accepted
				if (acceptance_criterion_test(axon.xyz_pos, node_to_visit.ptr,
					axon.dendrite_type_needed,
					false, has_vacant_dendrites)) {
					axon.nodes_accepted.push_back(&node_to_visit);
				}
				else {
					// Node was rejected only because it's too close
					if (has_vacant_dendrites) {
						append_children(node_to_visit.ptr, axon.nodes_to_visit, access_epochs_started);
					}
					delete& node_to_visit;
				}

				// Node is visited now, so remove it
				axon.nodes_to_visit.pop_front();
			}

			// No nodes to visit anymore
			if (axon.nodes_to_visit.empty()) {
				/**
				 * Assign a probability to each node in the nodes_accepted list.
				 * The probability for connecting to the same neuron (i.e., the axon's neuron) is set 0.
				 * Nodes with 0 probability are removed.
				 * The probabilities of all list elements sum up to 1.
				 */
				create_interval(axon.neuron_id, axon.xyz_pos, axon.dendrite_type_needed, axon.nodes_accepted);

				/**
				 * Select node with target neuron
				 */
				select_subinterval(axon.nodes_accepted, node_selected);

				// Clear nodes_accepted list for next interval creation
				for (it = axon.nodes_accepted.begin(); it != axon.nodes_accepted.end(); ) {
					delete* it;
					it = axon.nodes_accepted.erase(it);
				}

				// Now nodes_accepted and nodes_to_visit are empty

				// Node was selected
				if (nullptr != node_selected) {
					// Selected node is parent. A parent cannot be a target neuron.
					// So append its children to nodes_to_visit
					if (node_selected->is_parent) {
						append_children(node_selected, axon.nodes_to_visit, access_epochs_started);
					}
					else {
						// Target neuron found
						// Create synapse creation request for the target neuron
						map_synapse_creation_requests_outgoing[node_selected->rank].append(
							axon.neuron_id,
							node_selected->cell.get_neuron_id(),
							axon.dendrite_type_needed);
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

			if (delete_axon) {
				// Delete contents of axon
				delete& axon;
			}
			else {
				// Move axon to end of list so that another axon
				// is examined in the next iteration
				vacant_axons.push_back(&axon);
			}
		}
	} while (axon_added || !vacant_axons.empty());

	// Complete all started access epochs
	for (size_t i = 0; i < access_epochs_started.size(); i++) {
		if (access_epochs_started[i]) {
			MPI_Win_unlock((int)i, mpi_rma_node_allocator->mpi_window);
		}
	}
}
