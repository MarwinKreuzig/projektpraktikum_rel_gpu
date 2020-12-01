/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Neurons.h"

#include "MPIWrapper.h"
#include "Partition.h"
#include "RelearnException.h"

#include <array>

Neurons::Neurons(size_t num_neurons, const Parameters& params, const Partition& partition)
	: Neurons{ num_neurons, params, partition,
	NeuronModels::create<models::ModelA>(num_neurons, params.k, params.tau_C, params.beta, params.h, params.x_0, params.tau_x, params.refrac_time) } {
}

Neurons::Neurons(size_t num_neurons, const Parameters& params, const Partition& partition, std::unique_ptr<NeuronModels> model)
	: num_neurons(num_neurons),
	partition(partition),
	neuron_models(std::move(model)),
	axons(SynapticElements::AXON, num_neurons, params.eta_A, params.C_target, params.nu, params.vacant_retract_ratio),
	dendrites_exc(SynapticElements::DENDRITE, num_neurons, params.eta_D_ex, params.C_target, params.nu, params.vacant_retract_ratio),
	dendrites_inh(SynapticElements::DENDRITE, num_neurons, params.eta_D_in, params.C_target, params.nu, params.vacant_retract_ratio),
	positions(num_neurons),
	calcium(num_neurons),
	area_names(num_neurons),
	random_number_generator(RandomHolder<Neurons /*<NeuronModels, Axons, DendritesExc, DendritesInh>*/>::get_random_generator()),
	random_number_distribution(0.0, std::nextafter(1.0, 2.0))

{
	// Init member variables
	for (size_t i = 0; i < num_neurons; i++) {
		// Set calcium concentration
		const auto fired = neuron_models->get_fired(i);
		calcium[i] = fired ? neuron_models->get_beta() : 0.0;
	}
}

// NOTE: The static variables must be reset to 0 before this function can be used
// for the synapse creation phase in the next connectivity update
std::tuple<bool, size_t, Vec3d, Cell::DendriteType> Neurons::get_vacant_axon() const noexcept {
	static size_t i = 0, j = 0;

	size_t neuron_id;
	Vec3d xyz_pos;
	Cell::DendriteType dendrite_type_needed;

	const std::vector<double>& axons_cnts = axons.get_cnts();
	const std::vector<double>& axons_connected_cnts = axons.get_connected_cnts();
	const std::vector<SynapticElements::SignalType>& axons_signal_types = axons.get_signal_types();
	const std::vector<double>& axons_x_dims = positions.get_x_dims();
	const std::vector<double>& axons_y_dims = positions.get_y_dims();
	const std::vector<double>& axons_z_dims = positions.get_z_dims();

	while (i < num_neurons) {
		// neuron's vacant axons
		const auto num_vacant_axons = static_cast<unsigned int>(axons_cnts[i] - axons_connected_cnts[i]);

		if (j < num_vacant_axons) {
			j++;
			// Vacant axon found
			// set neuron id of vacant axon
			neuron_id = i;

			// set neuron's position
			xyz_pos.x = axons_x_dims[i];
			xyz_pos.y = axons_y_dims[i];
			xyz_pos.z = axons_z_dims[i];

			// set dendrite type matching this axon
			// DendriteType::INHIBITORY axon
			if (SynapticElements::INHIBITORY == axons_signal_types[i]) {
				dendrite_type_needed = Cell::DendriteType::INHIBITORY;
			}
			// DendriteType::EXCITATORY axon
			else {
				dendrite_type_needed = Cell::DendriteType::EXCITATORY;
			}

			return std::make_tuple(true, neuron_id, xyz_pos, dendrite_type_needed);
		}

		i++;
		j = 0;
	} // while

	return std::make_tuple(false, neuron_id, xyz_pos, dendrite_type_needed);
}

void Neurons::init_synaptic_elements() {
	/**
	* Mark dendrites as exc./inh.
	*/
	for (auto i = 0; i < num_neurons; i++) {
		dendrites_exc.set_signal_type(i, SynapticElements::EXCITATORY);
		dendrites_inh.set_signal_type(i, SynapticElements::INHIBITORY);
	}

	// Give unbound synaptic elements as well

	//            int num_axons = 1;
	//            int num_dends = 1;
	const int num_axons = 0;
	const int num_dends = 0;

	const std::vector<double>& axons_cnts = axons.get_cnts();
	const std::vector<double>& dendrites_inh_cnts = dendrites_inh.get_cnts();
	const std::vector<double>& dendrites_exc_cnts = dendrites_exc.get_cnts();

	for (auto i = 0; i < num_neurons; i++) {
		axons.update_cnt(i, num_axons);
		dendrites_inh.update_cnt(i, num_dends);
		dendrites_exc.update_cnt(i, num_dends);

		RelearnException::check(axons_cnts[i] >= axons.get_connected_cnts()[i]);
		RelearnException::check(dendrites_inh_cnts[i] >= dendrites_inh.get_connected_cnts()[i]);
		RelearnException::check(dendrites_exc_cnts[i] >= dendrites_exc.get_connected_cnts()[i]);
	}

}

void Neurons::delete_synapses(size_t& num_synapses_deleted, NetworkGraph& network_graph) {
	/**
	* 1. Update number of synaptic elements and delete synapses if necessary
	*/

	GlobalTimers::timers.start(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

	num_synapses_deleted = 0;

	debug_check_counts();

	std::list<PendingSynapseDeletion> list_with_pending_deletions;

	/**
	* Create list with synapses to delete (pending synapse deletions)
	*/

	// For all synaptic element types (axons, dends exc., dends inh.)
	for (SynapticElements* synaptic_elements : { &axons, &dendrites_exc, &dendrites_inh }) {
		const auto element_type = synaptic_elements->get_element_type();

		// For my neurons
		for (auto neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
			/**
			* Create and delete synaptic elements as required.
			* This function only deletes elements (bound and unbound), no synapses.
			*/
			const auto num_synapses_to_delete = synaptic_elements->update_number_elements(neuron_id);
			if (num_synapses_to_delete == 0) {
				continue;
			}
			/**
			* Create a list with all pending synapse deletions.
			* During creating this list, the possibility that neurons want to delete the same
			* synapse is considered.
			*/
			const auto signal_type = synaptic_elements->get_signal_type(neuron_id);
			find_synapses_for_deletion(neuron_id, element_type, signal_type, num_synapses_to_delete, network_graph, list_with_pending_deletions);

		} // For my neurons
	} // For all synaptic element types


	/**
	* - Go through list with pending synapse deletions and copy those into map "map_synapse_deletion_requests_outgoing"
	*   where the other neuron affected by the deletion is not one of my neurons
	* - Tell every rank how many deletion requests to receive from me
	* - Prepare for corresponding number of deletion requests from every rank and receive them
	* - Add received deletion requests to the list with pending deletions
	* - Execute pending deletions
	*/

	/**
	* Go through list with pending synapse deletions and copy those into
	* map "map_synapse_deletion_requests_outgoing" where the other neuron
	* affected by the deletion is not one of my neurons
	*/

	MapSynapseDeletionRequests map_synapse_deletion_requests_outgoing;
	// All pending deletion requests
	for (const auto& list_it : list_with_pending_deletions) {
		const auto target_rank = list_it.affected_neuron_id.rank;

		// Affected neuron of deletion request resides on different rank.
		// Thus the request needs to be communicated.
		if (target_rank != MPIWrapper::my_rank) {
			map_synapse_deletion_requests_outgoing[target_rank].append(
				list_it.src_neuron_id.neuron_id,
				list_it.tgt_neuron_id.neuron_id,
				list_it.affected_neuron_id.neuron_id,
				list_it.affected_element_type,
				list_it.signal_type,
				list_it.synapse_id);
		}
	}

	/**
	* Send to every rank the number of deletion requests it should prepare for from me.
	* Likewise, receive the number of deletion requests that I should prepare for from every rank.
	*/

	std::vector<size_t> num_synapse_deletion_requests_for_ranks(MPIWrapper::num_ranks, 0);
	// Fill vector with my number of synapse deletion requests for every rank
	// Requests to myself are kept local and not sent to myself again.
	for (const auto& map_it : map_synapse_deletion_requests_outgoing) {
		auto rank = map_it.first;
		auto num_requests = map_it.second.size();

		num_synapse_deletion_requests_for_ranks[rank] = num_requests;
	}


	std::vector<size_t> num_synapse_deletion_requests_from_ranks(MPIWrapper::num_ranks, 112233);
	// Send and receive the number of synapse deletion requests
	MPIWrapper::all_to_all(num_synapse_deletion_requests_for_ranks, num_synapse_deletion_requests_from_ranks, MPIWrapper::Scope::global);

	MapSynapseDeletionRequests map_synapse_deletion_requests_incoming;
	// Now I know how many requests I will get from every rank.
	// Allocate memory for all incoming synapse deletion requests.
	for (auto rank = 0; rank < MPIWrapper::num_ranks; ++rank) {
		auto num_requests = num_synapse_deletion_requests_from_ranks[rank];
		if (0 != num_requests) {
			map_synapse_deletion_requests_incoming[rank].resize(num_requests);
		}
	}

	std::vector<MPIWrapper::AsyncToken> mpi_requests(map_synapse_deletion_requests_outgoing.size() + map_synapse_deletion_requests_incoming.size());

	/**
	* Send and receive actual synapse deletion requests
	*/

	auto mpi_requests_index = 0;

	// Receive actual synapse deletion requests
	for (auto& map_it : map_synapse_deletion_requests_incoming) {
		const auto rank = map_it.first;
		auto buffer = map_it.second.get_requests();
		const auto size_in_bytes = static_cast<int>(map_it.second.get_requests_size_in_bytes());

		MPIWrapper::async_receive(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

		++mpi_requests_index;
	}

	// Send actual synapse deletion requests
	for (const auto& map_it : map_synapse_deletion_requests_outgoing) {
		const auto rank = map_it.first;
		const auto buffer = map_it.second.get_requests();
		const auto size_in_bytes = static_cast<int>(map_it.second.get_requests_size_in_bytes());

		MPIWrapper::async_send(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

		++mpi_requests_index;
	}

	// Wait for all sends and receives to complete
	MPIWrapper::wait_all_tokens(mpi_requests);

	/**
	* Go through all received deletion requests and add them to the list with pending requests.
	*/

	// From smallest to largest rank that sent deletion request
	for (const auto& map_it : map_synapse_deletion_requests_incoming) {
		const SynapseDeletionRequests& requests = map_it.second;
		const int other_rank = map_it.first;
		const auto num_requests = requests.size();

		// All requests of a rank
		for (auto request_index = 0; request_index < num_requests; ++request_index) {
			std::array<size_t, 6> arr = requests.get_request(request_index);

			size_t src_neuron_id = arr[0];
			size_t tgt_neuron_id = arr[1];
			size_t affected_neuron_id = arr[2];
			size_t affected_element_type = arr[3];
			size_t signal_type = arr[4];
			size_t synapse_id = arr[5];

			/**
			* Add received synapse deletion request to list with pending synapse deletions
			*/

			// My affected neuron is the source neuron of the synapse
			if (SynapticElements::ElementType::AXON == affected_element_type) {
				add_synapse_to_pending_deletions(
					RankNeuronId(MPIWrapper::my_rank, src_neuron_id),
					RankNeuronId(other_rank, tgt_neuron_id),
					RankNeuronId(MPIWrapper::my_rank, affected_neuron_id),
					static_cast<SynapticElements::ElementType>(affected_element_type),
					static_cast<SynapticElements::SignalType>(signal_type),
					static_cast<unsigned int>(synapse_id),
					list_with_pending_deletions);
			}
			// My affected neuron is the target neuron of the synapse
			else if (SynapticElements::ElementType::DENDRITE == affected_element_type) {
				add_synapse_to_pending_deletions(
					RankNeuronId(other_rank, src_neuron_id),
					RankNeuronId(MPIWrapper::my_rank, tgt_neuron_id),
					RankNeuronId(MPIWrapper::my_rank, affected_neuron_id),
					static_cast<SynapticElements::ElementType>(affected_element_type),
					static_cast<SynapticElements::SignalType>(signal_type),
					static_cast<unsigned int>(synapse_id),
					list_with_pending_deletions);
			}
			else {
				std::cout << "Invalid type of affected element." << std::endl;
			}
		} // All requests of a rank
	} // All ranks that sent deletion requests

	/**
	* Now the list with pending synapse deletions contains all deletion requests
	* of synapses that are connected to at least one of my neurons
	*
	* NOTE:
	* (i)  A synapse can be connected to two of my neurons
	* (ii) A synapse can be connected to one of my neurons and the other neuron belongs to another rank
	*/

	/* Delete all synapses pending for deletion */
	delete_synapses(list_with_pending_deletions, axons, dendrites_exc, dendrites_inh, network_graph, num_synapses_deleted);

	debug_check_counts();

	GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);
}

void Neurons::create_synapses(size_t& num_synapses_created, Octree& global_tree, NetworkGraph& network_graph) {
	/**
	* 2. Create Synapses
	*
	* - Update region trees (num dendrites in leaves and inner nodes) - postorder traversal (input: cnts, connected_cnts arrays)
	* - Determine target region for every axon
	* - Find target neuron for every axon (input: position, type; output: target neuron_id)
	* - Update synaptic elements (no connection when target neuron's dendrites have already been taken by previous axon)
	* - Update network
	*/
	num_synapses_created = 0;

	debug_check_counts();

	/**
	* Update global tree bottom-up with current number
	* of vacant dendrites and resulting positions
	*/

	/**********************************************************************************/

	// Lock local RMA memory for local stores
	MPIWrapper::lock_window(MPIWrapper::my_rank, MPI_Locktype::exclusive);

	// Update my local trees bottom-up
	GlobalTimers::timers.start(TimerRegion::UPDATE_LOCAL_TREES);
	global_tree.update_local_trees(dendrites_exc, dendrites_inh, num_neurons);
	GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_LOCAL_TREES);

	/**
	* Exchange branch nodes
	*/
	GlobalTimers::timers.start(TimerRegion::EXCHANGE_BRANCH_NODES);
	OctreeNode* rma_buffer_branch_nodes = MPIWrapper::rma_buffer_branch_nodes.ptr;
	// Copy local trees' root nodes to correct positions in receive buffer

	const size_t num_local_trees = global_tree.get_num_local_trees();
	for (size_t i = 0; i < num_local_trees; i++) {
		const size_t global_subdomain_id = partition.get_my_subdomain_id_start() + i;
		const OctreeNode* root_node = global_tree.get_local_root(i);

		// This assignment copies memberwise
		rma_buffer_branch_nodes[global_subdomain_id] = *root_node;
	}

	// Allgather in-place branch nodes from every rank
	MPIWrapper::all_gather_inline(rma_buffer_branch_nodes, num_local_trees, MPIWrapper::Scope::global);

	GlobalTimers::timers.stop_and_add(TimerRegion::EXCHANGE_BRANCH_NODES);

	// Insert only received branch nodes into global tree
	// The local ones are already in the global tree
	GlobalTimers::timers.start(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);
	const size_t num_rma_buffer_branch_nodes = MPIWrapper::rma_buffer_branch_nodes.num_nodes;
	for (size_t i = 0; i < num_rma_buffer_branch_nodes; i++) {
		if (i < partition.get_my_subdomain_id_start() ||
			i > partition.get_my_subdomain_id_end()) {
			global_tree.insert(rma_buffer_branch_nodes + i);
		}
	}
	GlobalTimers::timers.stop_and_add(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);

	// Update global tree
	GlobalTimers::timers.start(TimerRegion::UPDATE_GLOBAL_TREE);
	const auto level_branches = global_tree.get_level_of_branch_nodes();

	// Only update whenever there are other branches to update
	if (level_branches > 0) {
		global_tree.update_from_level(level_branches - 1);
	}
	GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_GLOBAL_TREE);

	// Unlock local RMA memory and make local stores visible in public window copy
	MPIWrapper::unlock_window(MPIWrapper::my_rank);

	/**********************************************************************************/

	// Makes sure that all ranks finished their local access epoch
	// before a remote origin opens an access epoch
	MPIWrapper::barrier(MPIWrapper::Scope::global);

	/**
	* Find target neuron for every vacant axon
	*/
	GlobalTimers::timers.start(TimerRegion::FIND_TARGET_NEURONS);

	const std::vector<double>* dendrites_cnts = nullptr; // TODO(fabian) find a nicer solution
	const std::vector<double>* dendrites_connected_cnts = nullptr;

	int num_axons_connected_increment = 0;
	MapSynapseCreationRequests map_synapse_creation_requests_outgoing;

	const std::vector<double>& axons_cnts = axons.get_cnts();
	const std::vector<double>& axons_connected_cnts = axons.get_connected_cnts();
	const std::vector<SynapticElements::SignalType>& axons_signal_types = axons.get_signal_types();

	// For my neurons
	for (size_t neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
		// Number of vacant axons
		const auto num_vacant_axons = static_cast<unsigned int>(axons_cnts[neuron_id] - axons_connected_cnts[neuron_id]);
		RelearnException::check(num_vacant_axons >= 0);

		Cell::DendriteType dendrite_type_needed = Cell::DendriteType::INHIBITORY;
		// DendriteType::INHIBITORY axon
		if (SynapticElements::INHIBITORY == axons_signal_types[neuron_id]) {
			dendrite_type_needed = Cell::DendriteType::INHIBITORY;
		}
		// DendriteType::EXCITATORY axon
		else {
			dendrite_type_needed = Cell::DendriteType::EXCITATORY;
		}

		// Position of current neuron
		const Vec3d axon_xyz_pos = positions.get_position(neuron_id);

		// For all vacant axons of neuron "neuron_id"
		for (size_t j = 0; j < num_vacant_axons; j++) {
			/**
			* Find target neuron for connecting and
			* connect if target neuron has still dendrite available.
			*
			* The target neuron might not have any dendrites std::left
			* as other axons might already have connected to them.
			* Right now, those collisions are handled in a first-come-first-served fashion.
			*/
			size_t target_neuron_id;
			int target_rank;
			bool target_neuron_found;
			target_neuron_found = global_tree.find_target_neuron(neuron_id, axon_xyz_pos, dendrite_type_needed,target_neuron_id, target_rank);

			if (target_neuron_found) {
				/*
				* Append request for synapse creation to rank "target_rank"
				* Note that "target_rank" could also be my own rank.
				*/
				map_synapse_creation_requests_outgoing[target_rank].append(neuron_id, target_neuron_id, dendrite_type_needed);
			}
		} /* all vacant axons of a neuron */
	} /* my neurons */

	GlobalTimers::timers.stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

	// Make cache empty for next connectivity update
	GlobalTimers::timers.start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
	global_tree.empty_remote_nodes_cache();
	GlobalTimers::timers.stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
	GlobalTimers::timers.start(TimerRegion::CREATE_SYNAPSES);
	{

		/**
		* At this point "map_synapse_creation_requests_outgoing" contains
		* all synapse creation requests from this rank
		*
		* The next step is to send the requests to the target ranks and
		* receive the requests from other ranks (including myself)
		*/

		/**
		* Send to every rank the number of requests it should prepare for from me.
		* Likewise, receive the number of requests that I should prepare for from every rank.
		*/
		std::vector<size_t> num_synapse_requests_for_ranks(MPIWrapper::num_ranks, 0);
		// Fill vector with my number of synapse requests for every rank (including me)
		for (const auto& it : map_synapse_creation_requests_outgoing) {
			auto rank = it.first;
			auto num_requests = (it.second).size();

			num_synapse_requests_for_ranks[rank] = num_requests;
		}

		std::vector<size_t> num_synapse_requests_from_ranks(MPIWrapper::num_ranks, 112233);
		// Send and receive the number of synapse requests
		MPIWrapper::all_to_all(num_synapse_requests_for_ranks, num_synapse_requests_from_ranks, MPIWrapper::Scope::global);

		MapSynapseCreationRequests map_synapse_creation_requests_incoming;
		// Now I know how many requests I will get from every rank.
		// Allocate memory for all incoming synapse requests.
		for (auto rank = 0; rank < MPIWrapper::num_ranks; rank++) {
			auto num_requests = num_synapse_requests_from_ranks[rank];
			if (0 != num_requests) {
				map_synapse_creation_requests_incoming[rank].resize(num_requests);
			}
		}

		std::vector<MPIWrapper::AsyncToken>
			mpi_requests(map_synapse_creation_requests_outgoing.size() + map_synapse_creation_requests_incoming.size());

		/**
		* Send and receive actual synapse requests
		*/
		auto mpi_requests_index = 0;

		// Receive actual synapse requests
		for (auto& it : map_synapse_creation_requests_incoming) {
			const auto rank = it.first;
			auto buffer = it.second.get_requests();
			const auto size_in_bytes = static_cast<int>(it.second.get_requests_size_in_bytes());

			MPIWrapper::async_receive(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

			mpi_requests_index++;
		}
		// Send actual synapse requests
		for (const auto& it : map_synapse_creation_requests_outgoing) {
			const auto rank = it.first;
			const auto buffer = it.second.get_requests();
			const auto size_in_bytes = static_cast<int>(it.second.get_requests_size_in_bytes());

			MPIWrapper::async_send(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

			mpi_requests_index++;
		}

		// Wait for all sends and receives to complete
		MPIWrapper::wait_all_tokens(mpi_requests);

		/**
		* Go through all received requests and try to connect.
		*
		* The order is from the smallest to the largest neuron id
		* as we start with the smallest rank which has the smallest neuron ids.
		*/
		// From smallest to largest rank that sent request
		for (auto& it : map_synapse_creation_requests_incoming) {
			const auto source_rank = it.first;
			SynapseCreationRequests& requests = it.second;
			const auto num_requests = requests.size();

			// All requests of a rank
			for (auto request_index = 0; request_index < num_requests; request_index++) {
				size_t source_neuron_id;
				size_t target_neuron_id;
				size_t dendrite_type_needed;
				std::tie(source_neuron_id, target_neuron_id, dendrite_type_needed) = requests.get_request(request_index);

				// Sanity check: if the request received is targeted for me
				if (target_neuron_id >= num_neurons) {
					RelearnException::fail("Target_neuron_id exceeds my neurons");
					exit(EXIT_FAILURE);
				}
				// DendriteType::INHIBITORY dendrite requested
				if (Cell::DendriteType::INHIBITORY == dendrite_type_needed) {
					dendrites_cnts = &dendrites_inh.get_cnts();
					dendrites_connected_cnts = &dendrites_inh.get_connected_cnts();
					num_axons_connected_increment = -1;
				}
				// DendriteType::EXCITATORY dendrite requested
				else {
					dendrites_cnts = &dendrites_exc.get_cnts();
					dendrites_connected_cnts = &dendrites_exc.get_connected_cnts();
					num_axons_connected_increment = +1;
				}

				// Target neuron has still dendrite available, so connect
				RelearnException::check((*dendrites_cnts)[target_neuron_id] - (*dendrites_connected_cnts)[target_neuron_id] >= 0);

				const auto diff = static_cast<unsigned int>((*dendrites_cnts)[target_neuron_id] - (*dendrites_connected_cnts)[target_neuron_id]);
				if (diff != 0) {
					// Increment num of connected dendrites
					//dendrites_connected_cnts[target_neuron_id]++;

					if (Cell::DendriteType::INHIBITORY == dendrite_type_needed) {
						dendrites_inh.update_conn_cnt(target_neuron_id, 1.0, "inh");
					}
					else {
						dendrites_exc.update_conn_cnt(target_neuron_id, 1.0, "exc");
					}

					// Update network
					network_graph.add_edge_weight(target_neuron_id, MPIWrapper::my_rank, source_neuron_id, source_rank, num_axons_connected_increment);

					// Set response to "connected" (success)
					requests.set_response(request_index, 1);
					num_synapses_created++;
					//std::cout << " [CONNECTED]\n";
				}
				else {
					// Set response to "not connected" (not success)
					requests.set_response(request_index, 0);

					// Other axons were faster and came first
					//std::cout << " [NOT CONNECTED] (dendrites already occupied)\n";
				}
			} // All requests of a rank
		} // Increasing order of ranks that sent requests


		  /**
		  * Send and receive responses for synapse requests
		  */
		mpi_requests_index = 0;

		// Receive responses
		for (auto& it : map_synapse_creation_requests_outgoing) {
			const auto rank = it.first;
			auto buffer = it.second.get_responses();
			const auto size_in_bytes = static_cast<int>(it.second.get_responses_size_in_bytes());

			MPIWrapper::async_receive(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

			mpi_requests_index++;
		}
		// Send responses
		for (const auto& it : map_synapse_creation_requests_incoming) {
			const auto rank = it.first;
			const auto buffer = it.second.get_responses();
			const auto size_in_bytes = static_cast<int>(it.second.get_responses_size_in_bytes());

			MPIWrapper::async_send(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

			mpi_requests_index++;
		}
		// Wait for all sends and receives to complete
		MPIWrapper::wait_all_tokens(mpi_requests);

		/**
		* Register which axons could be connected
		*
		* NOTE: Do not create synapses in the network for my own responses as the corresponding synapses, if possible,
		* would have been created before sending the response to myself (see above).
		*/
		for (const auto& it : map_synapse_creation_requests_outgoing) {
			const auto target_rank = it.first;
			const SynapseCreationRequests& requests = it.second;
			const auto num_requests = requests.size();

			// All responses from a rank
			for (auto request_index = 0; request_index < num_requests; request_index++) {
				char connected = requests.get_response(request_index);
				size_t source_neuron_id;
				size_t target_neuron_id;
				size_t dendrite_type_needed;
				std::tie(source_neuron_id, target_neuron_id, dendrite_type_needed) = requests.get_request(request_index);

				//std::cout << "From: " << source_neuron_id << " to " << target_neuron_id << ": " << dendrite_type_needed << std::endl;

				// Request to form synapse succeeded
				if (connected != 0) {
					// Increment num of connected axons
					axons.update_conn_cnt(source_neuron_id, 1.0, "ax");
					//axons_connected_cnts[source_neuron_id]++;
					num_synapses_created++;

					const double delta = axons.get_cnt(source_neuron_id) - axons.get_connected_cnt(source_neuron_id);
					RelearnException::check(delta >= 0, std::to_string(delta));

					// I have already created the synapse in the network
					// if the response comes from myself
					if (target_rank != MPIWrapper::my_rank) {
						// Update network
						num_axons_connected_increment = (Cell::DendriteType::INHIBITORY == dendrite_type_needed) ? -1 : +1;
						network_graph.add_edge_weight(target_neuron_id, target_rank, source_neuron_id, MPIWrapper::my_rank, num_axons_connected_increment);
					}
				}
				else {
					// Other axons were faster and came first
					//std::cout << " [NOT CONNECTED] (dendrites already occupied)\n";
				}
			} // All responses from a rank
		} // All outgoing requests
	}

	GlobalTimers::timers.stop_and_add(TimerRegion::CREATE_SYNAPSES);
	debug_check_counts();
}

void Neurons::debug_check_counts() {
	const std::vector<double>& axs_count = axons.get_cnts();
	const std::vector<double>& axs_conn_count = axons.get_connected_cnts();
	const std::vector<double>& de_count = dendrites_exc.get_cnts();
	const std::vector<double>& de_conn_count = dendrites_exc.get_connected_cnts();
	const std::vector<double>& di_count = dendrites_inh.get_cnts();
	const std::vector<double>& di_conn_count = dendrites_inh.get_connected_cnts();

	for (size_t i = 0; i < num_neurons; i++) {
		const double diff_axs = axs_count[i] - axs_conn_count[i];
		const double diff_de = de_count[i] - de_conn_count[i];
		const double diff_di = di_count[i] - di_conn_count[i];

		RelearnException::check(diff_axs >= 0.0, std::to_string(diff_axs));
		RelearnException::check(diff_de >= 0.0, std::to_string(diff_de));
		RelearnException::check(diff_di >= 0.0, std::to_string(diff_di));
	}
}

void Neurons::print_sums_of_synapses_and_elements_to_log_file_on_rank_0(size_t step, LogFiles& log_file, const Parameters& params, size_t sum_synapses_deleted, size_t sum_synapses_created) {
	unsigned int sum_axons_exc_cnts, sum_axons_exc_connected_cnts;
	unsigned int sum_axons_inh_cnts, sum_axons_inh_connected_cnts;
	unsigned int sum_dends_exc_cnts, sum_dends_exc_connected_cnts;
	unsigned int sum_dends_inh_cnts, sum_dends_inh_connected_cnts;
	unsigned int sum_axons_exc_vacant, sum_axons_inh_vacant;
	unsigned int sum_dends_exc_vacant, sum_dends_inh_vacant;

	// My vacant axons (exc./inh.)
	sum_axons_exc_cnts = sum_axons_exc_connected_cnts = 0;
	sum_axons_inh_cnts = sum_axons_inh_connected_cnts = 0;

	const std::vector<double>& cnts_ax = axons.get_cnts();
	const std::vector<double>& connected_cnts_ax = axons.get_connected_cnts();
	const std::vector<SynapticElements::SignalType>& signal_types = axons.get_signal_types();

	for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
		if (SynapticElements::EXCITATORY == signal_types[neuron_id]) {
			sum_axons_exc_cnts += static_cast<unsigned int>(cnts_ax[neuron_id]);
			sum_axons_exc_connected_cnts += static_cast<unsigned int>(connected_cnts_ax[neuron_id]);
		}
		else {
			sum_axons_inh_cnts += static_cast<unsigned int>(cnts_ax[neuron_id]);
			sum_axons_inh_connected_cnts += static_cast<unsigned int>(connected_cnts_ax[neuron_id]);
		}
	}
	sum_axons_exc_vacant = sum_axons_exc_cnts - sum_axons_exc_connected_cnts;
	sum_axons_inh_vacant = sum_axons_inh_cnts - sum_axons_inh_connected_cnts;

	// My vacant dendrites
	// Exc.
	sum_dends_exc_cnts = sum_dends_exc_connected_cnts = 0;
	const std::vector<double>& cnts_den_ex = dendrites_exc.get_cnts();
	const std::vector<double>& connected_cnts_den_ex = dendrites_exc.get_connected_cnts();
	for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
		sum_dends_exc_cnts += static_cast<unsigned int>(cnts_den_ex[neuron_id]);
		sum_dends_exc_connected_cnts += static_cast<unsigned int>(connected_cnts_den_ex[neuron_id]);
	}
	sum_dends_exc_vacant = sum_dends_exc_cnts - sum_dends_exc_connected_cnts;

	// Inh.
	sum_dends_inh_cnts = sum_dends_inh_connected_cnts = 0;
	const std::vector<double>& cnts_den_in = dendrites_inh.get_cnts();
	const std::vector<double>& connected_cnts_den_in = dendrites_inh.get_connected_cnts();
	for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
		sum_dends_inh_cnts += static_cast<unsigned int>(cnts_den_in[neuron_id]);
		sum_dends_inh_connected_cnts += static_cast<unsigned int>(connected_cnts_den_in[neuron_id]);
	}
	sum_dends_inh_vacant = sum_dends_inh_cnts - sum_dends_inh_connected_cnts;

	// Get global sums at rank 0
	std::array<unsigned int, 6> sums_local = { sum_axons_exc_vacant,
		sum_axons_inh_vacant,
		sum_dends_exc_vacant,
		sum_dends_inh_vacant,
		static_cast<unsigned int>(sum_synapses_deleted),
		static_cast<unsigned int>(sum_synapses_created) };

	std::array<unsigned int, 6> sums_global{ 0, 0, 0, 0, 0, 0 }; // Init all to zero

	MPIWrapper::reduce(sums_local, sums_global, MPIWrapper::ReduceFunction::sum, 0, MPIWrapper::Scope::global);

	// Output data
	if (0 == MPIWrapper::my_rank) {
		std::ofstream& file = log_file.get_file(0);
		const int cwidth = 20;  // Column width

								// Write headers to file if not already done so
		if (0 == step) {
			file << params << std::endl;
			file << "# SUMS OVER ALL NEURONS\n";
			file << std::left
				<< std::setw(cwidth) << "# step"
				<< std::setw(cwidth) << "Axons exc. (vacant)"
				<< std::setw(cwidth) << "Axons inh. (vacant)"
				<< std::setw(cwidth) << "Dends exc. (vacant)"
				<< std::setw(cwidth) << "Dends inh. (vacant)"
				<< std::setw(cwidth) << "Synapses deleted"
				<< std::setw(cwidth) << "Synapses created"
				<< "\n";
		}

		// Write data at step "step"
		file << std::left
			<< std::setw(cwidth) << step
			<< std::setw(cwidth) << sums_global[0]
			<< std::setw(cwidth) << sums_global[1]
			<< std::setw(cwidth) << sums_global[2]
			<< std::setw(cwidth) << sums_global[3]
			<< std::setw(cwidth) << sums_global[4] / 2 // As counted on both of the neurons
			<< std::setw(cwidth) << sums_global[5] / 2 // As counted on both of the neurons
			<< "\n";
	}
}

// Print global information about all neurons at rank 0

void Neurons::print_neurons_overview_to_log_file_on_rank_0(size_t step, LogFiles& log_file, const Parameters& params) {
	const StatisticalMeasures<double> calcium_statistics =
		global_statistics(calcium.data(), num_neurons, params.num_neurons, 0, MPIWrapper::Scope::global);

	const StatisticalMeasures<double> activity_statistics =
		global_statistics(neuron_models->get_x().data(), num_neurons, params.num_neurons, 0, MPIWrapper::Scope::global);

	// Output data
	if (0 == MPIWrapper::my_rank) {
		std::ofstream& file = log_file.get_file(0);
		const int cwidth = 16;  // Column width

								// Write headers to file if not already done so
		if (0 == step) {
			file << params << std::endl;
			file << "# ALL NEURONS\n";
			file << std::left
				<< std::setw(cwidth) << "# step"
				<< std::setw(cwidth) << "C (avg)"
				<< std::setw(cwidth) << "C (min)"
				<< std::setw(cwidth) << "C (max)"
				<< std::setw(cwidth) << "C (var)"
				<< std::setw(cwidth) << "C (std_dev)"
				<< std::setw(cwidth) << "activity (avg)"
				<< std::setw(cwidth) << "activity (min)"
				<< std::setw(cwidth) << "activity (max)"
				<< std::setw(cwidth) << "activity (var)"
				<< std::setw(cwidth) << "activity (std_dev)"
				<< "\n";
		}

		// Write data at step "step"
		file << std::left
			<< std::setw(cwidth) << step
			<< std::setw(cwidth) << calcium_statistics.avg
			<< std::setw(cwidth) << calcium_statistics.min
			<< std::setw(cwidth) << calcium_statistics.max
			<< std::setw(cwidth) << calcium_statistics.var
			<< std::setw(cwidth) << calcium_statistics.std
			<< std::setw(cwidth) << activity_statistics.avg
			<< std::setw(cwidth) << activity_statistics.min
			<< std::setw(cwidth) << activity_statistics.max
			<< std::setw(cwidth) << activity_statistics.var
			<< std::setw(cwidth) << activity_statistics.std
			<< "\n";
	}
}

void Neurons::print_network_graph_to_log_file(LogFiles& log_file, const NetworkGraph& network_graph, const Parameters& params, const NeuronIdMap& neuron_id_map) {
	std::ofstream& file = log_file.get_file(0);

	// Write output format to file
	file << "# " << params.num_neurons << std::endl; // Total number of neurons
	file << "# <target neuron id> <source neuron id> <weight>" << std::endl;

	// Write network graph to file
	//*file << network_graph << std::endl;
	network_graph.print(file, neuron_id_map);
}

void Neurons::print_positions_to_log_file(LogFiles& log_file, const Parameters& params, const NeuronIdMap& neuron_id_map) {
	std::ofstream& file = log_file.get_file(0);

	// Write total number of neurons to log file
	file << "# " << params.num_neurons << std::endl;
	file << "# " << "<global id> <pos x> <pos y> <pos z> <area>" << std::endl;

	const std::vector<double>& axons_x_dims = positions.get_x_dims();
	const std::vector<double>& axons_y_dims = positions.get_y_dims();
	const std::vector<double>& axons_z_dims = positions.get_z_dims();

	// Print global ids, positions, and areas of local neurons
	bool ret = false;
	size_t glob_id = 0;
	NeuronIdMap::RankNeuronId rank_neuron_id{ 0, 0 };

	rank_neuron_id.rank = MPIWrapper::my_rank;
	file << std::fixed << std::setprecision(6);
	for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
		rank_neuron_id.neuron_id = neuron_id;
		std::tie(ret, glob_id) = neuron_id_map.rank_neuron_id2glob_id(rank_neuron_id);
		RelearnException::check(ret);

		file << glob_id << " "
			<< axons_x_dims[neuron_id] << " "
			<< axons_y_dims[neuron_id] << " "
			<< axons_z_dims[neuron_id] << " "
			<< area_names[neuron_id] << "\n";
	}

	file << std::flush;
	file << std::defaultfloat;
}

void Neurons::print() {
	// Column widths
	const int cwidth_left = 6;
	const int cwidth = 16;

	// Heading
	std::cout << std::left << std::setw(cwidth_left) << "gid" << std::setw(cwidth) << "x" << std::setw(cwidth) << "AP";
	std::cout << std::setw(cwidth) << "refrac" << std::setw(cwidth) << "C" << std::setw(cwidth) << "A" << std::setw(cwidth) << "D_ex" << std::setw(cwidth) << "D_in" << "\n";

	// Values
	for (size_t i = 0; i < num_neurons; i++) {
		std::cout << std::left << std::setw(cwidth_left) << i << std::setw(cwidth) << neuron_models->get_x(i) << std::setw(cwidth) << neuron_models->get_fired(i);
		std::cout << std::setw(cwidth) << neuron_models->get_secondary_variable(i) << std::setw(cwidth) << calcium[i] << std::setw(cwidth) << axons.get_cnt(i);
		std::cout << std::setw(cwidth) << dendrites_exc.get_cnt(i) << std::setw(cwidth) << dendrites_inh.get_cnt(i) << "\n";
	}
}

void Neurons::print_info_for_barnes_hut() {
	const std::vector<double>& x_dims = positions.get_x_dims();
	const std::vector<double>& y_dims = positions.get_y_dims();
	const std::vector<double>& z_dims = positions.get_z_dims();

	const std::vector<double>& axons_cnts = axons.get_cnts();
	const std::vector<double>& dendrites_exc_cnts = dendrites_exc.get_cnts();
	const std::vector<double>& dendrites_inh_cnts = dendrites_inh.get_cnts();

	const std::vector<double>& axons_connected_cnts = axons.get_connected_cnts();
	const std::vector<double>& dendrites_exc_connected_cnts = dendrites_exc.get_connected_cnts();
	const std::vector<double>& dendrites_inh_connected_cnts = dendrites_inh.get_connected_cnts();

	// Column widths
	const int cwidth_small = 8;
	const int cwidth_medium = 16;
	const int cwidth_big = 27;

	std::string my_string;


	// Heading
	std::cout << std::left << std::setw(cwidth_small) << "gid" << std::setw(cwidth_small) << "region" << std::setw(cwidth_medium) << "position";
	std::cout << std::setw(cwidth_big) << "axon (exist|connected)" << std::setw(cwidth_big) << "exc_den (exist|connected)";
	std::cout << std::setw(cwidth_big) << "inh_den (exist|connected)" << "\n";

	// Values
	for (size_t i = 0; i < num_neurons; i++) {
		std::cout << std::left << std::setw(cwidth_small) << i;

		const auto x = static_cast<unsigned int>(x_dims[i]);
		const auto y = static_cast<unsigned int>(y_dims[i]);
		const auto z = static_cast<unsigned int>(z_dims[i]);

		my_string = "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")";
		std::cout << std::setw(cwidth_medium) << my_string;

		my_string = std::to_string(axons_cnts[i]) + "|" + std::to_string(axons_connected_cnts[i]);
		std::cout << std::setw(cwidth_big) << my_string;

		my_string = std::to_string(dendrites_exc_cnts[i]) + "|" + std::to_string(dendrites_exc_connected_cnts[i]);
		std::cout << std::setw(cwidth_big) << my_string;

		my_string = std::to_string(dendrites_inh_cnts[i]) + "|" + std::to_string(dendrites_inh_connected_cnts[i]);
		std::cout << std::setw(cwidth_big) << my_string;

		std::cout << std::endl;
	}
}

/**
* Returns iterator to randomly chosen synapse from list
*/

typename std::list<Neurons::Synapse>::const_iterator Neurons::select_synapse(const std::list<Synapse>& list) {
	// Point to first synapse
	auto it = list.begin();

	// Draw random number from [0,1)
	const double random_number = random_number_distribution(random_number_generator);

	// Make iterator point to selected element
	std::advance(it, static_cast<int>(list.size() * random_number));

	return it;
}

void Neurons::add_synapse_to_pending_deletions(const RankNeuronId& src_neuron_id,
	const RankNeuronId& tgt_neuron_id,
	const RankNeuronId& affected_neuron_id,
	SynapticElements::ElementType affected_element_type,
	SynapticElements::SignalType signal_type,
	unsigned int synapse_id,
	std::list<PendingSynapseDeletion>& list) {

	typename std::list<PendingSynapseDeletion>::iterator it;
	bool found = false;

	// Check if synapse is already pending for deletion
	for (it = list.begin(); it != list.end() && !found; ++it) {
		if ((it->src_neuron_id == src_neuron_id) &&
			(it->tgt_neuron_id == tgt_neuron_id) &&
			(it->synapse_id == synapse_id)) {
			/**
			* As the synapse was selected by both neurons connected through it for deletion,
			* both already deleted their respective synaptic elements of this synapse.
			* I.e., no element is std::left to be set vacant.
			*/
			it->affected_element_already_deleted = true;

			found = true;
		}
	}

	// Synapse not pending yet, so add it to pending deletions
	if (!found) {
		PendingSynapseDeletion pending_deletion{
			src_neuron_id,
			tgt_neuron_id,
			affected_neuron_id,
			affected_element_type,
			signal_type,
			synapse_id,
			false };

		list.emplace_back(pending_deletion);
	}
}

/**
* Determines which synapses should be deleted.
* The selected synapses connect with neuron "neuron_id" and the type of
* those synapses is given by "signal_type".
*
* NOTE: The semantics of the function is not nice but used to postpone all updates
* due to synapse deletion until all neurons have decided *independently* which synapse
* to delete. This should reflect how it's done for a distributed memory implementation.
*/

void Neurons::find_synapses_for_deletion(size_t neuron_id,
	SynapticElements::ElementType element_type,
	SynapticElements::SignalType signal_type,
	unsigned int num_synapses_to_delete,
	const NetworkGraph& network_graph,
	std::list<PendingSynapseDeletion>& list_pending_deletions) {

	// Only do something if necessary
	if (0 == num_synapses_to_delete) {
		return;
	}

	std::list<Synapse> list_synapses;


	/**
	* Bound elements to delete: Axons
	*/
	if (SynapticElements::AXON == element_type) {
		/**
		* Create list with synapses
		*/
		const NetworkGraph::Edges& out_edges = network_graph.get_out_edges(neuron_id);
		// Walk through outgoing edges
		for (const auto& it : out_edges) {
			/**
			* Create "edge weight" number of synapses and add them to the synapse list
			* NOTE: We take abs(it->second) here as DendriteType::INHIBITORY synapses have count < 0
			*/
			const auto rank = it.first.first;
			const auto id = it.first.second;

			const auto abs_synapse_weight = abs(it.second);
			if (abs_synapse_weight == 0) {
				continue;
			}

			for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
				RankNeuronId rank_neuron_id(rank, id);
				list_synapses.emplace_back(rank_neuron_id, synapse_id);
			}
		}

		/**
		* Select synapses for deletion
		*/
		RelearnException::check(num_synapses_to_delete <= list_synapses.size(), "num_synapses_to_delete > last_synapses.size()");

		for (unsigned int num_synapses_selected = 0; num_synapses_selected < num_synapses_to_delete; ++num_synapses_selected) {
			// Randomly select synapse for deletion
			typename std::list<Synapse>::const_iterator synapse_selected = select_synapse(list_synapses);
			RelearnException::check(synapse_selected != list_synapses.end()); // Make sure that valid synapse was selected

															 // Check if synapse is already in pending deletions, if not, add it.
			add_synapse_to_pending_deletions(
				RankNeuronId(MPIWrapper::my_rank, neuron_id),
				synapse_selected->rank_neuron_id,
				synapse_selected->rank_neuron_id,
				SynapticElements::DENDRITE,
				signal_type,
				synapse_selected->synapse_id,
				list_pending_deletions);

			// Remove selected synapse from synapse list
			list_synapses.erase(synapse_selected);
		}
		// Empty list of synapses
		list_synapses.clear();
	}

	/**
	* Bound elements to delete: DendriteType::EXCITATORY dendrites
	*/
	if (SynapticElements::DENDRITE == element_type && SynapticElements::EXCITATORY == signal_type) {

		/**
		* Create list with synapses
		*/
		const NetworkGraph::Edges& in_edges = network_graph.get_in_edges(neuron_id);
		// Walk through ingoing edges
		for (const auto& it : in_edges) {
			/**
			* Create "edge weight" number of synapses and add them to the synapse list
			* NOTE: We take positive entries only as those are DendriteType::EXCITATORY synapses
			*/
			const auto rank = it.first.first;
			const auto id = it.first.second;

			const auto abs_synapse_weight = abs(it.second);
			if (abs_synapse_weight == 0) {
				continue;
			}

			for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
				RankNeuronId rank_neuron_id(rank, id);
				list_synapses.emplace_back(rank_neuron_id, synapse_id);
			}
		}

		/**
		* Select synapses for deletion
		*/
		RelearnException::check(num_synapses_to_delete <= list_synapses.size(), "num_synapses_to_delete > last_synapses.size()");

		for (unsigned int num_synapses_selected = 0; num_synapses_selected < num_synapses_to_delete; ++num_synapses_selected) {
			// Randomly select synapse for deletion
			typename std::list<Synapse>::const_iterator synapse_selected = select_synapse(list_synapses);
			RelearnException::check(synapse_selected != list_synapses.end()); // Make sure that valid synapse was selected

			// Check if synapse is already in pending deletions, if not, add it.
			add_synapse_to_pending_deletions(
				synapse_selected->rank_neuron_id,
				RankNeuronId(MPIWrapper::my_rank, neuron_id),
				synapse_selected->rank_neuron_id,
				SynapticElements::AXON,
				signal_type,
				synapse_selected->synapse_id,
				list_pending_deletions);

			// Remove selected synapse from synapse list
			list_synapses.erase(synapse_selected);
		}
		// Empty list of synapses
		list_synapses.clear();
	}

	/**
	* Bound elements to delete: DendriteType::INHIBITORY dendrites
	*/
	if (SynapticElements::DENDRITE == element_type && SynapticElements::INHIBITORY == signal_type) {
		/**
		* Create list with synapses
		*/
		const NetworkGraph::Edges& in_edges = network_graph.get_in_edges(neuron_id);
		// Walk through ingoing edges
		for (const auto& it : in_edges) {
			/**
			* Create "edge weight" number of synapses and add them to the synapse list
			*
			* NOTE: We take negative entries only as those are DendriteType::INHIBITORY synapses
			*/
			const auto rank = it.first.first;
			const auto id = it.first.second;

			const auto abs_synapse_weight = abs(it.second);
			if (abs_synapse_weight == 0) {
				continue;
			}

			for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
				RankNeuronId rank_neuron_id(rank, id);
				list_synapses.emplace_back(rank_neuron_id, synapse_id);
			}
		}

		/**
		* Select synapses for deletion
		*/
		RelearnException::check(num_synapses_to_delete <= list_synapses.size(), "num_synapses_to_delete > last_synapses.size()");

		for (unsigned int num_synapses_selected = 0; num_synapses_selected < num_synapses_to_delete; ++num_synapses_selected) {
			// Randomly select synapse for deletion
			typename std::list<Synapse>::const_iterator synapse_selected= select_synapse(list_synapses);
			RelearnException::check(synapse_selected != list_synapses.end()); // Make sure that valid synapse was selected

															 // Check if synapse is already in pending deletions, if not, add it.
			add_synapse_to_pending_deletions(
				synapse_selected->rank_neuron_id,
				RankNeuronId(MPIWrapper::my_rank, neuron_id),
				synapse_selected->rank_neuron_id,
				SynapticElements::AXON,
				signal_type,
				synapse_selected->synapse_id,
				list_pending_deletions);

			// Remove selected synapse from synapse list
			list_synapses.erase(synapse_selected);
		}
		// Empty list of synapses
		list_synapses.clear();
	}
}

void Neurons::print_pending_synapse_deletions(const std::list<PendingSynapseDeletion>& list) {
	for (const auto& it : list) {
		std::cout << "src_neuron_id: " << it.src_neuron_id << "\n";
		std::cout << "tgt_neuron_id: " << it.tgt_neuron_id << "\n";
		std::cout << "affected_neuron_id: " << it.affected_neuron_id << "\n";
		std::cout << "affected_element_type: " << it.affected_element_type << "\n";
		std::cout << "signal_type: " << it.signal_type << "\n";
		std::cout << "synapse_id: " << it.synapse_id << "\n";
		std::cout << "affected_element_already_deleted: " << it.affected_element_already_deleted << "\n" << std::endl;
	}
}

void Neurons::delete_synapses(std::list<PendingSynapseDeletion>& list,
	SynapticElements& axons,
	SynapticElements& dendrites_exc,
	SynapticElements& dendrites_inh,
	NetworkGraph& network_graph,
	size_t& num_synapses_deleted) {

	debug_check_counts();

	//double* axons_connected_cnts = axons.get_connected_cnts();
	//double* dendrites_exc_connected_cnts = dendrites_exc.get_connected_cnts();
	//double* dendrites_inh_connected_cnts = dendrites_inh.get_connected_cnts();

	/* Execute pending synapse deletions */
	for (const auto& it : list) {
		// Pending synapse deletion is valid (not completely) if source or
		// target neuron belong to me. To be completely valid, things such as
		// the neuron id need to be validated as well.
		RelearnException::check(it.src_neuron_id.rank == MPIWrapper::my_rank || it.tgt_neuron_id.rank == MPIWrapper::my_rank);

		if (it.src_neuron_id.rank == MPIWrapper::my_rank && it.tgt_neuron_id.rank == MPIWrapper::my_rank) {
			/**
			* Count the deleted synapse once for each connected neuron.
			* The reason is that synapses where neurons are on different ranks are also
			* counted once on each rank
			*/
			num_synapses_deleted += 2;
		}
		else {
			num_synapses_deleted += 1;
		}

		/**
		*  Update network graph
		*/
		// DendriteType::EXCITATORY synapses have positive count, so decrement
		int weight_increment = 0;
		if (SynapticElements::EXCITATORY == it.signal_type) {
			weight_increment = -1;
		}
		// DendriteType::INHIBITORY synapses have negative count, so increment
		else {
			weight_increment = +1;
		}
		network_graph.add_edge_weight(it.tgt_neuron_id.neuron_id, it.tgt_neuron_id.rank,
			it.src_neuron_id.neuron_id, it.src_neuron_id.rank, weight_increment);

		/**
		* Set element of affected neuron vacant if necessary,
		* i.e., only if the affected neuron belongs to me and the
		* element of the affected neuron still exists.
		*
		* NOTE: Checking that the affected neuron belongs to me is important
		* because the list of pending deletion requests also contains requests whose
		* affected neuron belongs to a different rank.
		*/
		const auto affected_neuron_id = it.affected_neuron_id.neuron_id;

		if (it.affected_neuron_id.rank == MPIWrapper::my_rank && !it.affected_element_already_deleted) {
			if (SynapticElements::AXON == it.affected_element_type) {
				//--axons_connected_cnts[affected_neuron_id];
				axons.update_conn_cnt(affected_neuron_id, -1.0, "ax");
			}
			else if ((SynapticElements::DENDRITE == it.affected_element_type) &&
				(SynapticElements::EXCITATORY == it.signal_type)) {
				//--dendrites_exc_connected_cnts[affected_neuron_id];
				dendrites_exc.update_conn_cnt(affected_neuron_id, -1.0, "exc");

				if (!(dendrites_exc.get_cnts()[affected_neuron_id] >=
					dendrites_exc.get_connected_cnts()[affected_neuron_id])) {
					std::cout << "neuron_id: " << affected_neuron_id << "\n"
						<< "cnt: " << dendrites_exc.get_cnts()[affected_neuron_id] << "\n"
						<< "connected_cnt: " << dendrites_exc.get_connected_cnts()[affected_neuron_id] << "\n";
				}

			}
			else if ((SynapticElements::DENDRITE == it.affected_element_type) &&
				(SynapticElements::INHIBITORY == it.signal_type)) {
				//--dendrites_inh_connected_cnts[affected_neuron_id];
				dendrites_inh.update_conn_cnt(affected_neuron_id, -1.0, "inh");
			}
			else {
				std::cout << "Invalid list element for pending synapse deletion." << std::endl;
			}
		}
		debug_check_counts();
	}

	debug_check_counts();
}
