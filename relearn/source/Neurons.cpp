#include "Neurons.h"
#include "Partition.h"
#include "RelearnException.h"

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
		calcium[i] = neuron_models->get_beta() * neuron_models->get_fired(i);
	}
}

// NOTE: The static variables must be reset to 0 before this function can be used
// for the synapse creation phase in the next connectivity update
bool Neurons::get_vacant_axon(size_t& neuron_id, Vec3d& xyz_pos, Cell::DendriteType& dendrite_type_needed) noexcept {
	static size_t i = 0, j = 0;

	const double* axons_cnts = axons.get_cnts();
	const double* axons_connected_cnts = axons.get_connected_cnts();
	const SynapticElements::SignalType* axons_signal_types = axons.get_signal_types();
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

			return true;
		}
		else {
			i++;
			j = 0;
		}
	} // while

	return false;
}

void Neurons::init_synaptic_elements() {
	SynapticElements::SignalType* dendrites_exc_signal_types = dendrites_exc.get_signal_types();
	SynapticElements::SignalType* dendrites_inh_signal_types = dendrites_inh.get_signal_types();

	/**
	* Mark dendrites as exc./inh.
	*/
	for (auto i = 0; i < num_neurons; i++) {
		dendrites_exc_signal_types[i] = SynapticElements::EXCITATORY;  // Mark DendriteType::EXCITATORY dendrites as DendriteType::EXCITATORY
		dendrites_inh_signal_types[i] = SynapticElements::INHIBITORY;  // Mark DendriteType::INHIBITORY dendrites as DendriteType::INHIBITORY
	}

	// Give unbound synaptic elements as well

	//            int num_axons = 1;
	//            int num_dends = 1;
	const int num_axons = 0;
	const int num_dends = 0;

	// TODO: Look here

	const double* axons_cnts = axons.get_cnts();
	const double* dendrites_inh_cnts = dendrites_inh.get_cnts();
	const double* dendrites_exc_cnts = dendrites_exc.get_cnts();

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
		if (target_rank != MPIInfos::my_rank) {
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

	std::vector<size_t> num_synapse_deletion_requests_for_ranks(MPIInfos::num_ranks, 0);
	// Fill vector with my number of synapse deletion requests for every rank
	// Requests to myself are kept local and not sent to myself again.
	for (const auto& map_it : map_synapse_deletion_requests_outgoing) {
		auto rank = map_it.first;
		auto num_requests = map_it.second.size();

		num_synapse_deletion_requests_for_ranks[rank] = num_requests;
	}


	std::vector<size_t> num_synapse_deletion_requests_from_ranks(MPIInfos::num_ranks, 112233);
	// Send and receive the number of synapse deletion requests
	MPI_Alltoall(num_synapse_deletion_requests_for_ranks.data(), sizeof(size_t), MPI_CHAR,
		num_synapse_deletion_requests_from_ranks.data(), sizeof(size_t), MPI_CHAR,
		MPI_COMM_WORLD);


	MapSynapseDeletionRequests map_synapse_deletion_requests_incoming;
	// Now I know how many requests I will get from every rank.
	// Allocate memory for all incoming synapse deletion requests.
	for (auto rank = 0; rank < MPIInfos::num_ranks; ++rank) {
		auto num_requests = num_synapse_deletion_requests_from_ranks[rank];
		if (0 != num_requests) {
			map_synapse_deletion_requests_incoming[rank].resize(num_requests);
		}
	}

	std::vector<MPI_Request> mpi_requests(map_synapse_deletion_requests_outgoing.size() + map_synapse_deletion_requests_incoming.size());

	/**
	* Send and receive actual synapse deletion requests
	*/

	auto mpi_requests_index = 0;

	// Receive actual synapse deletion requests
	for (auto& map_it : map_synapse_deletion_requests_incoming) {
		const auto rank = map_it.first;
		auto buffer = map_it.second.get_requests();
		const auto size_in_bytes = static_cast<int>(map_it.second.get_requests_size_in_bytes());

		MPI_Irecv(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
		++mpi_requests_index;
	}

	// Send actual synapse deletion requests
	for (const auto& map_it : map_synapse_deletion_requests_outgoing) {
		const auto rank = map_it.first;
		const auto buffer = map_it.second.get_requests();
		const auto size_in_bytes = static_cast<int>(map_it.second.get_requests_size_in_bytes());

		MPI_Isend(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
		++mpi_requests_index;
	}

	// Wait for all sends and receives to complete
	MPI_Waitall(mpi_requests_index, mpi_requests.data(), MPI_STATUSES_IGNORE);

	/**
	* Go through all received deletion requests and add them to the list with pending requests.
	*/

	// From smallest to largest rank that sent deletion request
	for (auto map_it = map_synapse_deletion_requests_incoming.begin(); map_it != map_synapse_deletion_requests_incoming.end(); ++map_it) {
		const SynapseDeletionRequests& requests = map_it->second;
		const int other_rank = map_it->first;
		const auto num_requests = requests.size();

		// All requests of a rank
		for (auto request_index = 0; request_index < num_requests; ++request_index) {
			size_t src_neuron_id, tgt_neuron_id, affected_neuron_id, affected_element_type, signal_type, synapse_id;
			requests.get_request(
				request_index,
				src_neuron_id,
				tgt_neuron_id,
				affected_neuron_id,
				affected_element_type,
				signal_type,
				synapse_id);

			/**
			* Add received synapse deletion request to list with pending synapse deletions
			*/

			// My affected neuron is the source neuron of the synapse
			if (SynapticElements::ElementType::AXON == affected_element_type) {
				add_synapse_to_pending_deletions(
					RankNeuronId(MPIInfos::my_rank, src_neuron_id),
					RankNeuronId(other_rank, tgt_neuron_id),
					RankNeuronId(MPIInfos::my_rank, affected_neuron_id),
					(SynapticElements::ElementType) affected_element_type,
					(SynapticElements::SignalType) signal_type,
					(unsigned int)synapse_id,
					list_with_pending_deletions);
			}
			// My affected neuron is the target neuron of the synapse
			else if (SynapticElements::ElementType::DENDRITE == affected_element_type) {
				add_synapse_to_pending_deletions(
					RankNeuronId(other_rank, src_neuron_id),
					RankNeuronId(MPIInfos::my_rank, tgt_neuron_id),
					RankNeuronId(MPIInfos::my_rank, affected_neuron_id),
					(SynapticElements::ElementType) affected_element_type,
					(SynapticElements::SignalType) signal_type,
					(unsigned int)synapse_id,
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
	MPIInfos::lock_window(MPIInfos::my_rank, MPI_Locktype::exclusive);

	// Update my local trees bottom-up
	GlobalTimers::timers.start(TimerRegion::UPDATE_LOCAL_TREES);
	global_tree.update_local_trees(dendrites_exc, dendrites_inh, num_neurons);
	GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_LOCAL_TREES);

	/**
	* Exchange branch nodes
	*/
	GlobalTimers::timers.start(TimerRegion::EXCHANGE_BRANCH_NODES);
	OctreeNode* rma_buffer_branch_nodes = MPIInfos::rma_buffer_branch_nodes.ptr;
	// Copy local trees' root nodes to correct positions in receive buffer

	const size_t num_local_trees = global_tree.get_num_local_trees();
	for (size_t i = 0; i < num_local_trees; i++) {
		const size_t global_subdomain_id = partition.get_my_subdomain_id_start() + i;
		const OctreeNode* root_node = global_tree.get_local_root(i);

		// This assignment copies memberwise
		rma_buffer_branch_nodes[global_subdomain_id] = *root_node;
	}

	// Allgather in-place branch nodes from every rank
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rma_buffer_branch_nodes,
		static_cast<int>(num_local_trees) * sizeof(OctreeNode),
		MPI_CHAR, MPI_COMM_WORLD);
	GlobalTimers::timers.stop_and_add(TimerRegion::EXCHANGE_BRANCH_NODES);

	// Insert only received branch nodes into global tree
	// The local ones are already in the global tree
	GlobalTimers::timers.start(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);
	const size_t num_rma_buffer_branch_nodes = MPIInfos::rma_buffer_branch_nodes.num_nodes;
	for (size_t i = 0; i < num_rma_buffer_branch_nodes; i++) {
		if (i < partition.get_my_subdomain_id_start() ||
			i > partition.get_my_subdomain_id_end()) {
			global_tree.insert(&rma_buffer_branch_nodes[i]);
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
	MPIInfos::unlock_window(MPIInfos::my_rank);

	/**********************************************************************************/

	// Makes sure that all ranks finished their local access epoch
	// before a remote origin opens an access epoch
	MPI_Barrier(MPI_COMM_WORLD);


	/**
	* Find target neuron for every vacant axon
	*/
	GlobalTimers::timers.start(TimerRegion::FIND_TARGET_NEURONS);

	const double* dendrites_cnts = nullptr;
	const double* dendrites_connected_cnts = nullptr;

	int num_axons_connected_increment = 0;
	MapSynapseCreationRequests map_synapse_creation_requests_outgoing;

	const double* axons_cnts = axons.get_cnts();
	const double* axons_connected_cnts = axons.get_connected_cnts();
	const SynapticElements::SignalType* axons_signal_types = axons.get_signal_types();

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
			* The target neuron might not have any dendrites left
			* as other axons might already have connected to them.
			* Right now, those collisions are handled in a first-come-first-served fashion.
			*/
			size_t target_neuron_id;
			int target_rank;
			const auto target_neuron_found = global_tree.find_target_neuron(neuron_id, axon_xyz_pos, dendrite_type_needed, target_neuron_id, target_rank);
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
		std::vector<size_t> num_synapse_requests_for_ranks(MPIInfos::num_ranks, 0);
		// Fill vector with my number of synapse requests for every rank (including me)
		for (auto it = map_synapse_creation_requests_outgoing.begin(); it != map_synapse_creation_requests_outgoing.end(); it++) {
			auto rank = it->first;
			auto num_requests = (it->second).size();

			num_synapse_requests_for_ranks[rank] = num_requests;
		}

		std::vector<size_t> num_synapse_requests_from_ranks(MPIInfos::num_ranks, 112233);
		// Send and receive the number of synapse requests
		MPI_Alltoall(num_synapse_requests_for_ranks.data(), sizeof(size_t), MPI_CHAR,
			num_synapse_requests_from_ranks.data(), sizeof(size_t), MPI_CHAR,
			MPI_COMM_WORLD);

		MapSynapseCreationRequests map_synapse_creation_requests_incoming;
		// Now I know how many requests I will get from every rank.
		// Allocate memory for all incoming synapse requests.
		for (auto rank = 0; rank < MPIInfos::num_ranks; rank++) {
			auto num_requests = num_synapse_requests_from_ranks[rank];
			if (0 != num_requests) {
				map_synapse_creation_requests_incoming[rank].resize(num_requests);
			}
		}

		std::vector<MPI_Request>
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

			MPI_Irecv(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
			mpi_requests_index++;
		}
		// Send actual synapse requests
		for (const auto& it : map_synapse_creation_requests_outgoing) {
			const auto rank = it.first;
			const auto buffer = it.second.get_requests();
			const auto size_in_bytes = static_cast<int>(it.second.get_requests_size_in_bytes());

			MPI_Isend(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
			mpi_requests_index++;
		}
		// Wait for all sends and receives to complete
		MPI_Waitall(mpi_requests_index, mpi_requests.data(), MPI_STATUSES_IGNORE);


		/**
		* Go through all received requests and try to connect.
		*
		* The order is from the smallest to the largest neuron id
		* as we start with the smallest rank which has the smallest neuron ids.
		*/
		// From smallest to largest rank that sent request
		for (auto it = map_synapse_creation_requests_incoming.begin(); it != map_synapse_creation_requests_incoming.end(); it++) {
			const auto source_rank = it->first;
			SynapseCreationRequests& requests = it->second;
			const auto num_requests = requests.size();

			// All requests of a rank
			for (auto request_index = 0; request_index < num_requests; request_index++) {
				size_t source_neuron_id;
				size_t target_neuron_id;
				size_t dendrite_type_needed;
				requests.get_request(request_index, source_neuron_id, target_neuron_id, dendrite_type_needed);

				// Sanity check: if the request received is targeted for me
				if (target_neuron_id >= num_neurons) {
					std::stringstream sstream;
					sstream << __FUNCTION__ << ": \"target_neuron_id\": " << target_neuron_id << " exceeds my neuron ids";
					LogMessages::print_error(sstream.str().c_str());
					exit(EXIT_FAILURE);
				}
				// DendriteType::INHIBITORY dendrite requested
				if (Cell::DendriteType::INHIBITORY == dendrite_type_needed) {
					dendrites_cnts = dendrites_inh.get_cnts();
					dendrites_connected_cnts = dendrites_inh.get_connected_cnts();
					num_axons_connected_increment = -1;
				}
				// DendriteType::EXCITATORY dendrite requested
				else {
					dendrites_cnts = dendrites_exc.get_cnts();
					dendrites_connected_cnts = dendrites_exc.get_connected_cnts();
					num_axons_connected_increment = +1;
				}

				// Target neuron has still dendrite available, so connect
				RelearnException::check(dendrites_cnts[target_neuron_id] - dendrites_connected_cnts[target_neuron_id] >= 0);
				if (static_cast<unsigned int>(dendrites_cnts[target_neuron_id] - dendrites_connected_cnts[target_neuron_id])) {

					// Increment num of connected dendrites
					//dendrites_connected_cnts[target_neuron_id]++;

					if (Cell::DendriteType::INHIBITORY == dendrite_type_needed) {
						dendrites_inh.update_conn_cnt(target_neuron_id, 1.0);
					}
					else {
						dendrites_exc.update_conn_cnt(target_neuron_id, 1.0);
					}

					// Update network
					network_graph.add_edge_weight(target_neuron_id, MPIInfos::my_rank, source_neuron_id, source_rank, num_axons_connected_increment);

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

			MPI_Irecv(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
			mpi_requests_index++;
		}
		// Send responses
		for (const auto& it : map_synapse_creation_requests_incoming) {
			const auto rank = it.first;
			const auto buffer = it.second.get_responses();
			const auto size_in_bytes = static_cast<int>(it.second.get_responses_size_in_bytes());

			MPI_Isend(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
			mpi_requests_index++;
		}
		// Wait for all sends and receives to complete
		MPI_Waitall(mpi_requests_index, mpi_requests.data(), MPI_STATUSES_IGNORE);

		/**
		* Register which axons could be connected
		*
		* NOTE: Do not create synapses in the network for my own responses as the corresponding synapses, if possible,
		* would have been created before sending the response to myself (see above).
		*/
		for (auto it = map_synapse_creation_requests_outgoing.begin(); it != map_synapse_creation_requests_outgoing.end(); it++) {
			const auto target_rank = it->first;
			const SynapseCreationRequests& requests = it->second;
			const auto num_requests = requests.size();

			// All responses from a rank
			for (auto request_index = 0; request_index < num_requests; request_index++) {
				char connected;
				requests.get_response(request_index, connected);
				size_t source_neuron_id;
				size_t target_neuron_id;
				size_t dendrite_type_needed;
				requests.get_request(request_index, source_neuron_id, target_neuron_id, dendrite_type_needed);

				//std::cout << "From: " << source_neuron_id << " to " << target_neuron_id << ": " << dendrite_type_needed << std::endl;

				// Request to form synapse succeeded
				if (connected) {
					// Increment num of connected axons
					axons.update_conn_cnt(source_neuron_id, 1.0);
					//axons_connected_cnts[source_neuron_id]++;
					num_synapses_created++;

					const double delta = axons.get_cnt(source_neuron_id) - axons.get_connected_cnt(source_neuron_id);
					RelearnException::check(delta >= 0, std::to_string(delta));

					// I have already created the synapse in the network
					// if the response comes from myself
					if (target_rank != MPIInfos::my_rank) {
						// Update network
						num_axons_connected_increment = (Cell::DendriteType::INHIBITORY == dendrite_type_needed) ? -1 : +1;
						network_graph.add_edge_weight(target_neuron_id, target_rank, source_neuron_id, MPIInfos::my_rank, num_axons_connected_increment);
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
	const double* axs_count = axons.get_cnts();
	const double* axs_conn_count = axons.get_connected_cnts();
	const double* de_count = dendrites_exc.get_cnts();
	const double* de_conn_count = dendrites_exc.get_connected_cnts();
	const double* di_count = dendrites_inh.get_cnts();
	const double* di_conn_count = dendrites_inh.get_connected_cnts();

	for (size_t i = 0; i < num_neurons; i++) {
		const double diff_axs = axs_count[i] - axs_conn_count[i];
		const double diff_de = de_count[i] >= de_conn_count[i];
		const double diff_di = di_count[i] >= di_conn_count[i];

		RelearnException::check(diff_axs >= 0.0, std::to_string(diff_axs));
		RelearnException::check(diff_de >= 0.0, std::to_string(diff_de));
		RelearnException::check(diff_di >= 0.0, std::to_string(diff_di));
	}
}

void Neurons::print_sums_of_synapses_and_elements_to_log_file_on_rank_0(size_t step, LogFiles& log_file, const Parameters& params, size_t sum_synapses_deleted, size_t sum_synapses_created) {
	using namespace std;

	unsigned int sum_axons_exc_cnts, sum_axons_exc_connected_cnts;
	unsigned int sum_axons_inh_cnts, sum_axons_inh_connected_cnts;
	unsigned int sum_dends_exc_cnts, sum_dends_exc_connected_cnts;
	unsigned int sum_dends_inh_cnts, sum_dends_inh_connected_cnts;
	unsigned int sum_axons_exc_vacant, sum_axons_inh_vacant;
	unsigned int sum_dends_exc_vacant, sum_dends_inh_vacant;

	// My vacant axons (exc./inh.)
	sum_axons_exc_cnts = sum_axons_exc_connected_cnts = 0;
	sum_axons_inh_cnts = sum_axons_inh_connected_cnts = 0;

	const double * cnts = axons.get_cnts();
	const double * connected_cnts = axons.get_connected_cnts();
	const SynapticElements::SignalType* signal_types = axons.get_signal_types();

	for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
		if (SynapticElements::EXCITATORY == signal_types[neuron_id]) {
			sum_axons_exc_cnts += static_cast<unsigned int>(cnts[neuron_id]);
			sum_axons_exc_connected_cnts += static_cast<unsigned int>(connected_cnts[neuron_id]);
		}
		else {
			sum_axons_inh_cnts += static_cast<unsigned int>(cnts[neuron_id]);
			sum_axons_inh_connected_cnts += static_cast<unsigned int>(connected_cnts[neuron_id]);
		}
	}
	sum_axons_exc_vacant = sum_axons_exc_cnts - sum_axons_exc_connected_cnts;
	sum_axons_inh_vacant = sum_axons_inh_cnts - sum_axons_inh_connected_cnts;

	// My vacant dendrites
	// Exc.
	sum_dends_exc_cnts = sum_dends_exc_connected_cnts = 0;
	cnts = dendrites_exc.get_cnts();
	connected_cnts = dendrites_exc.get_connected_cnts();
	for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
		sum_dends_exc_cnts += static_cast<unsigned int>(cnts[neuron_id]);
		sum_dends_exc_connected_cnts += static_cast<unsigned int>(connected_cnts[neuron_id]);
	}
	sum_dends_exc_vacant = sum_dends_exc_cnts - sum_dends_exc_connected_cnts;

	// Inh.
	sum_dends_inh_cnts = sum_dends_inh_connected_cnts = 0;
	cnts = dendrites_inh.get_cnts();
	connected_cnts = dendrites_inh.get_connected_cnts();
	for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
		sum_dends_inh_cnts += static_cast<unsigned int>(cnts[neuron_id]);
		sum_dends_inh_connected_cnts += static_cast<unsigned int>(connected_cnts[neuron_id]);
	}
	sum_dends_inh_vacant = sum_dends_inh_cnts - sum_dends_inh_connected_cnts;

	// Get global sums at rank 0
	const unsigned int sums_local[6]{ sum_axons_exc_vacant,
		sum_axons_inh_vacant,
		sum_dends_exc_vacant,
		sum_dends_inh_vacant,
		static_cast<unsigned int>(sum_synapses_deleted),
		static_cast<unsigned int>(sum_synapses_created) };
	unsigned int sums_global[6]{ 123 }; // Init all to zero

	MPI_Reduce(sums_local, sums_global, 6, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);


	// Output data
	if (0 == MPIInfos::my_rank) {
		ofstream& file = log_file.get_file(0);
		const int cwidth = 20;  // Column width

								// Write headers to file if not already done so
		if (0 == step) {
			file << params << endl;
			file << "# SUMS OVER ALL NEURONS\n";
			file << left
				<< setw(cwidth) << "# step"
				<< setw(cwidth) << "Axons exc. (vacant)"
				<< setw(cwidth) << "Axons inh. (vacant)"
				<< setw(cwidth) << "Dends exc. (vacant)"
				<< setw(cwidth) << "Dends inh. (vacant)"
				<< setw(cwidth) << "Synapses deleted"
				<< setw(cwidth) << "Synapses created"
				<< "\n";
		}

		// Write data at step "step"
		file << left
			<< setw(cwidth) << step
			<< setw(cwidth) << sums_global[0]
			<< setw(cwidth) << sums_global[1]
			<< setw(cwidth) << sums_global[2]
			<< setw(cwidth) << sums_global[3]
			<< setw(cwidth) << sums_global[4] / 2 // As counted on both of the neurons
			<< setw(cwidth) << sums_global[5] / 2 // As counted on both of the neurons
			<< "\n";
	}
}

// Print global information about all neurons at rank 0

void Neurons::print_neurons_overview_to_log_file_on_rank_0(size_t step, LogFiles& log_file, const Parameters& params) {
	using namespace std;

	const StatisticalMeasures<double> calcium_statistics =
		global_statistics(calcium.data(), num_neurons, params.num_neurons, 0, MPI_COMM_WORLD);

	const StatisticalMeasures<double> activity_statistics =
		global_statistics(neuron_models->get_x().data(), num_neurons, params.num_neurons, 0, MPI_COMM_WORLD);

	// Output data
	if (0 == MPIInfos::my_rank) {
		ofstream& file = log_file.get_file(0);
		const int cwidth = 16;  // Column width

								// Write headers to file if not already done so
		if (0 == step) {
			file << params << endl;
			file << "# ALL NEURONS\n";
			file << left
				<< setw(cwidth) << "# step"
				<< setw(cwidth) << "C (avg)"
				<< setw(cwidth) << "C (min)"
				<< setw(cwidth) << "C (max)"
				<< setw(cwidth) << "C (var)"
				<< setw(cwidth) << "C (std_dev)"
				<< setw(cwidth) << "activity (avg)"
				<< setw(cwidth) << "activity (min)"
				<< setw(cwidth) << "activity (max)"
				<< setw(cwidth) << "activity (var)"
				<< setw(cwidth) << "activity (std_dev)"
				<< "\n";
		}

		// Write data at step "step"
		file << left
			<< setw(cwidth) << step
			<< setw(cwidth) << calcium_statistics.avg
			<< setw(cwidth) << calcium_statistics.min
			<< setw(cwidth) << calcium_statistics.max
			<< setw(cwidth) << calcium_statistics.var
			<< setw(cwidth) << calcium_statistics.std
			<< setw(cwidth) << activity_statistics.avg
			<< setw(cwidth) << activity_statistics.min
			<< setw(cwidth) << activity_statistics.max
			<< setw(cwidth) << activity_statistics.var
			<< setw(cwidth) << activity_statistics.std
			<< "\n";
	}
}

void Neurons::print_network_graph_to_log_file(LogFiles& log_file, const NetworkGraph& network_graph, const Parameters& params, const NeuronIdMap& neuron_id_map) {
	using namespace std;
	ofstream& file = log_file.get_file(0);

	// Write output format to file
	file << "# " << params.num_neurons << endl; // Total number of neurons
	file << "# <target neuron id> <source neuron id> <weight>" << endl;

	// Write network graph to file
	//*file << network_graph << endl;
	network_graph.print(file, neuron_id_map);
}

void Neurons::print_positions_to_log_file(LogFiles& log_file, const Parameters& params, const NeuronIdMap& neuron_id_map) {
	using namespace std;

	ofstream& file = log_file.get_file(0);

	// Write total number of neurons to log file
	file << "# " << params.num_neurons << endl;
	file << "# " << "<global id> <pos x> <pos y> <pos z> <area>" << endl;

	const std::vector<double>& axons_x_dims = positions.get_x_dims();
	const std::vector<double>& axons_y_dims = positions.get_y_dims();
	const std::vector<double>& axons_z_dims = positions.get_z_dims();

	// Print global ids, positions, and areas of local neurons
	bool ret = false;
	size_t glob_id = 0;
	NeuronIdMap::RankNeuronId rank_neuron_id{ 0, 0 };

	rank_neuron_id.rank = MPIInfos::my_rank;
	file << std::fixed << std::setprecision(6);
	for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
		rank_neuron_id.neuron_id = neuron_id;
		ret = neuron_id_map.rank_neuron_id2glob_id(rank_neuron_id, glob_id);
		RelearnException::check(ret);

		file << glob_id << " "
			<< axons_x_dims[neuron_id] << " "
			<< axons_y_dims[neuron_id] << " "
			<< axons_z_dims[neuron_id] << " "
			<< area_names[neuron_id] << "\n";
	}

	file << flush;
	file << std::defaultfloat;
}

void Neurons::print() {
	using namespace std;

	// Column widths
	const int cwidth_left = 6;
	const int cwidth = 16;

	// Heading
	cout << left << setw(cwidth_left) << "gid" << setw(cwidth) << "x" << setw(cwidth) << "AP";
	cout << setw(cwidth) << "refrac" << setw(cwidth) << "C" << setw(cwidth) << "A" << setw(cwidth) << "D_ex" << setw(cwidth) << "D_in" << "\n";

	// Values
	for (size_t i = 0; i < num_neurons; i++) {
		cout << left << setw(cwidth_left) << i << setw(cwidth) << neuron_models->get_x(i) << setw(cwidth) << neuron_models->get_fired(i);
		cout << setw(cwidth) << neuron_models->get_secondary_variable(i) << setw(cwidth) << calcium[i] << setw(cwidth) << axons.get_cnt(i);
		cout << setw(cwidth) << dendrites_exc.get_cnt(i) << setw(cwidth) << dendrites_inh.get_cnt(i) << "\n";
	}
}

void Neurons::print_info_for_barnes_hut() {
	using namespace std;

	const std::vector<double>& x_dims = positions.get_x_dims();
	const std::vector<double>& y_dims = positions.get_y_dims();
	const std::vector<double>& z_dims = positions.get_z_dims();

	const double* axons_cnts = axons.get_cnts();
	const double* dendrites_exc_cnts = dendrites_exc.get_cnts();
	const double* dendrites_inh_cnts = dendrites_inh.get_cnts();

	const double* axons_connected_cnts = axons.get_connected_cnts();
	const double* dendrites_exc_connected_cnts = dendrites_exc.get_connected_cnts();
	const double* dendrites_inh_connected_cnts = dendrites_inh.get_connected_cnts();

	// Column widths
	const int cwidth_small = 8;
	const int cwidth_medium = 16;
	const int cwidth_big = 27;

	string my_string;


	// Heading
	cout << left << setw(cwidth_small) << "gid" << setw(cwidth_small) << "region" << setw(cwidth_medium) << "position";
	cout << setw(cwidth_big) << "axon (exist|connected)" << setw(cwidth_big) << "exc_den (exist|connected)";
	cout << setw(cwidth_big) << "inh_den (exist|connected)" << "\n";

	// Values
	for (size_t i = 0; i < num_neurons; i++) {
		cout << left << setw(cwidth_small) << i;

		my_string = "(" + to_string((unsigned int)x_dims[i]) + "," + to_string((unsigned int)y_dims[i]) + "," + to_string((unsigned int)z_dims[i]) + ")";
		cout << setw(cwidth_medium) << my_string;

		my_string = to_string(axons_cnts[i]) + "|" + to_string(axons_connected_cnts[i]);
		cout << setw(cwidth_big) << my_string;

		my_string = to_string(dendrites_exc_cnts[i]) + "|" + to_string(dendrites_exc_connected_cnts[i]);
		cout << setw(cwidth_big) << my_string;

		my_string = to_string(dendrites_inh_cnts[i]) + "|" + to_string(dendrites_inh_connected_cnts[i]);
		cout << setw(cwidth_big) << my_string;

		cout << endl;
	}
}

/**
* Returns iterator to randomly chosen synapse from list
*/

void Neurons::select_synapse(std::list<Synapse>& list, typename std::list<Synapse>::iterator& it) {
	// Point to first synapse
	it = list.begin();

	// Draw random number from [0,1)
	const double random_number = random_number_distribution(random_number_generator);

	// Make iterator point to selected element
	std::advance(it, static_cast<int>(list.size() * random_number));
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
			* I.e., no element is left to be set vacant.
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

		list.emplace_back(std::move(pending_deletion));
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
		for (auto it = out_edges.begin(); it != out_edges.end(); ++it) {
			/**
			* Create "edge weight" number of synapses and add them to the synapse list
			* NOTE: We take abs(it->second) here as DendriteType::INHIBITORY synapses have count < 0
			*/
			const unsigned int rounded = static_cast<unsigned int>(abs(it->second));
			for (unsigned int synapse_id = 0; synapse_id < rounded; ++synapse_id) {
				const RankNeuronId rank_neuron_id(it->first.first, it->first.second);
				list_synapses.push_back(Synapse(rank_neuron_id, synapse_id));
			}
		}

		/**
		* Select synapses for deletion
		*/
		{
			const bool valid = num_synapses_to_delete <= list_synapses.size();
			if (!valid) {
				std::cout << __func__
					<< "num_synapses_to_delete (" << num_synapses_to_delete << ") "
					<< "> "
					<< "list_synapses.size() (" << list_synapses.size() << ")\n";
			}
		}

		RelearnException::check(num_synapses_to_delete <= list_synapses.size());

		for (unsigned int num_synapses_selected = 0; num_synapses_selected < num_synapses_to_delete; ++num_synapses_selected) {
			// Randomly select synapse for deletion
			typename std::list<Synapse>::iterator synapse_selected;
			select_synapse(list_synapses, synapse_selected);
			RelearnException::check(synapse_selected != list_synapses.end()); // Make sure that valid synapse was selected

															 // Check if synapse is already in pending deletions, if not, add it.
			add_synapse_to_pending_deletions(
				RankNeuronId(MPIInfos::my_rank, neuron_id),
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
		for (auto it = in_edges.begin(); it != in_edges.end(); ++it) {
			/**
			* Create "edge weight" number of synapses and add them to the synapse list
			* NOTE: We take positive entries only as those are DendriteType::EXCITATORY synapses
			*/
			if (it->second > 0) {
				for (unsigned int synapse_id = 0; synapse_id < static_cast<unsigned int>(it->second); ++synapse_id) {
					const RankNeuronId rank_neuron_id(it->first.first, it->first.second);
					list_synapses.push_back(Synapse(rank_neuron_id, synapse_id));
				}
			}
		}

		/**
		* Select synapses for deletion
		*/
		{
			const bool valid = num_synapses_to_delete <= list_synapses.size();
			if (!valid) {
				std::cout << __func__
					<< "num_synapses_to_delete (" << num_synapses_to_delete << ") "
					<< "> "
					<< "list_synapses.size() (" << list_synapses.size() << ")\n";
			}
		}
		RelearnException::check(num_synapses_to_delete <= list_synapses.size());
		for (unsigned int num_synapses_selected = 0; num_synapses_selected < num_synapses_to_delete; ++num_synapses_selected) {
			// Randomly select synapse for deletion
			typename std::list<Synapse>::iterator synapse_selected;
			select_synapse(list_synapses, synapse_selected);
			RelearnException::check(synapse_selected != list_synapses.end()); // Make sure that valid synapse was selected

			// Check if synapse is already in pending deletions, if not, add it.
			add_synapse_to_pending_deletions(
				synapse_selected->rank_neuron_id,
				RankNeuronId(MPIInfos::my_rank, neuron_id),
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
		for (auto it = in_edges.begin(); it != in_edges.end(); ++it) {
			/**
			* Create "edge weight" number of synapses and add them to the synapse list
			*
			* NOTE: We take negative entries only as those are DendriteType::INHIBITORY synapses
			*/
			if (it->second < 0) {
				const unsigned int rounded = static_cast<unsigned int>(abs(it->second));
				for (unsigned int synapse_id = 0; synapse_id < rounded; ++synapse_id) {
					const RankNeuronId rank_neuron_id(it->first.first, it->first.second);
					list_synapses.push_back(Synapse(rank_neuron_id, synapse_id));
				}
			}
		}

		/**
		* Select synapses for deletion
		*/
		{
			const bool valid = num_synapses_to_delete <= list_synapses.size();
			if (!valid) {
				std::cout << __func__
					<< "num_synapses_to_delete (" << num_synapses_to_delete << ") "
					<< "> "
					<< "list_synapses.size() (" << list_synapses.size() << ")\n";
			}
		}
		RelearnException::check(num_synapses_to_delete <= list_synapses.size());
		for (unsigned int num_synapses_selected = 0; num_synapses_selected < num_synapses_to_delete; ++num_synapses_selected) {
			// Randomly select synapse for deletion
			typename std::list<Synapse>::iterator synapse_selected;
			select_synapse(list_synapses, synapse_selected);
			RelearnException::check(synapse_selected != list_synapses.end()); // Make sure that valid synapse was selected

															 // Check if synapse is already in pending deletions, if not, add it.
			add_synapse_to_pending_deletions(
				synapse_selected->rank_neuron_id,
				RankNeuronId(MPIInfos::my_rank, neuron_id),
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

void Neurons::print_pending_synapse_deletions(std::list<PendingSynapseDeletion>& list) {
	for (auto it = list.begin(); it != list.end(); it++) {
		std::cout << "src_neuron_id: " << it->src_neuron_id << "\n";
		std::cout << "tgt_neuron_id: " << it->tgt_neuron_id << "\n";
		std::cout << "affected_neuron_id: " << it->affected_neuron_id << "\n";
		std::cout << "affected_element_type: " << it->affected_element_type << "\n";
		std::cout << "signal_type: " << it->signal_type << "\n";
		std::cout << "synapse_id: " << it->synapse_id << "\n";
		std::cout << "affected_element_already_deleted: " << it->affected_element_already_deleted << "\n" << std::endl;
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
	for (auto it = list.begin(); it != list.end(); ++it) {
		// Pending synapse deletion is valid (not completely) if source or
		// target neuron belong to me. To be completely valid, things such as
		// the neuron id need to be validated as well.
		RelearnException::check(it->src_neuron_id.rank == MPIInfos::my_rank || it->tgt_neuron_id.rank == MPIInfos::my_rank);

		if (it->src_neuron_id.rank == MPIInfos::my_rank && it->tgt_neuron_id.rank == MPIInfos::my_rank) {
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
		if (SynapticElements::EXCITATORY == it->signal_type) {
			weight_increment = -1;
		}
		// DendriteType::INHIBITORY synapses have negative count, so increment
		else {
			weight_increment = +1;
		}
		network_graph.add_edge_weight(it->tgt_neuron_id.neuron_id, it->tgt_neuron_id.rank,
			it->src_neuron_id.neuron_id, it->src_neuron_id.rank, weight_increment);

		/**
		* Set element of affected neuron vacant if necessary,
		* i.e., only if the affected neuron belongs to me and the
		* element of the affected neuron still exists.
		*
		* NOTE: Checking that the affected neuron belongs to me is important
		* because the list of pending deletion requests also contains requests whose
		* affected neuron belongs to a different rank.
		*/
		const auto affected_neuron_id = it->affected_neuron_id.neuron_id;

		if (it->affected_neuron_id.rank == MPIInfos::my_rank && !it->affected_element_already_deleted) {
			if (SynapticElements::AXON == it->affected_element_type) {
				//--axons_connected_cnts[affected_neuron_id];
				axons.update_conn_cnt(affected_neuron_id, -1.0);
			}
			else if ((SynapticElements::DENDRITE == it->affected_element_type) &&
				(SynapticElements::EXCITATORY == it->signal_type)) {
				//--dendrites_exc_connected_cnts[affected_neuron_id];
				dendrites_exc.update_conn_cnt(affected_neuron_id, -1.0);

				if (!(dendrites_exc.get_cnts()[affected_neuron_id] >=
					dendrites_exc.get_connected_cnts()[affected_neuron_id])) {
					std::cout << "neuron_id: " << affected_neuron_id << "\n"
						<< "cnt: " << dendrites_exc.get_cnts()[affected_neuron_id] << "\n"
						<< "connected_cnt: " << dendrites_exc.get_connected_cnts()[affected_neuron_id] << "\n";
				}

			}
			else if ((SynapticElements::DENDRITE == it->affected_element_type) &&
				(SynapticElements::INHIBITORY == it->signal_type)) {
				//--dendrites_inh_connected_cnts[affected_neuron_id];
				dendrites_inh.update_conn_cnt(affected_neuron_id, -1.0);
			}
			else {
				std::cout << "Invalid list element for pending synapse deletion." << std::endl;
			}
		}
		debug_check_counts();
	}

	debug_check_counts();
}
