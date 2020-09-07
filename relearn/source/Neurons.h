/*
 * File:   Neurons.h
 * Author: naveau
 *
 * Created on September 26, 2014, 10:23 AM
 */

#ifndef NEURONS_H
#define	NEURONS_H

#include <random>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>
#include <mpi.h>


#include "Positions.h"
#include "Parameters.h"
#include "Octree.h"
#include "SynapticElements.h"
#include "LogFiles.h"
#include "Parameters.h"
#include "Timers.h"
#include "MPIInfos.h"
#include "NetworkGraph.h"
#include "Partition.h"
#include "NeuronIdMap.h"
#include "Random.h"

 /***************************************************************************************************
  * NOTE: The following two type declarations (SynapseCreationRequests, MapSynapseCreationRequests) *
  * are outside of the class Neurons so that the class Octree can use them                          *
  ***************************************************************************************************/

  /**
   * Type for synapse creation requests which are used with MPI
   */
class SynapseCreationRequests {
public:
	SynapseCreationRequests() : num_requests(0) {};

	// Return size
	size_t size() { return num_requests; }

	// Resize the number of requests
	void resize(size_t size) {
		num_requests = size;
		requests.resize(3 * size);
		responses.resize(size);
	}

	// Append request 3-tuple
	void append(size_t source_neuron_id, size_t target_neuron_id, size_t dendrite_type_needed) {
		num_requests++;

		requests.push_back(source_neuron_id);
		requests.push_back(target_neuron_id);
		requests.push_back(dendrite_type_needed);

		responses.resize(responses.size() + 1);
	}

	// Get request at index "request_index"
	void get_request(size_t request_index, size_t& source_neuron_id, size_t& target_neuron_id, size_t& dendrite_type_needed) {
		size_t base_index = 3 * request_index;

		source_neuron_id = requests[base_index];
		target_neuron_id = requests[base_index + 1];
		dendrite_type_needed = requests[base_index + 2];
	}

	// Set response at index "request_index"
	void set_response(size_t request_index, char connected) {
		responses[request_index] = connected;
	}

	// Get response at index "request_index"
	void get_response(size_t request_index, char& connected) {
		connected = responses[request_index];
	}

	// Get pointers to data
	size_t* get_requests() { return requests.data(); }
	char* get_responses() { return responses.data(); }

	size_t  get_requests_size_in_bytes() { return requests.size() * sizeof(size_t); }
	size_t  get_responses_size_in_bytes() { return responses.size() * sizeof(char); }

private:
	size_t num_requests;           // Number of synapse creation requests
	std::vector<size_t> requests;  // Each request to form a synapse is a 3-tuple: (source_neuron_id, target_neuron_id, dendrite_type_needed)
								   // That is why requests.size() == 3*responses.size()
								   // Note, a more memory-efficient implementation would use a smaller data type (not size_t) for dendrite_type_needed.
								   // This vector is used as MPI communication buffer
	std::vector<char> responses;   // Response if the corresponding request was accepted and thus the synapse was formed
								   // responses[i] refers to requests[3*i,...,3*i+2]
								   // This vector is used as MPI communication buffer
};

/**
 * Map of (MPI rank; SynapseCreationRequests)
 * The MPI rank specifies the corresponding process
 */
typedef std::map<int, SynapseCreationRequests> MapSynapseCreationRequests;

/**
 * Identifies a neuron by the MPI rank of its owner
 * and its neuron id on the owner, i.e., the pair <rank, neuron_id>
 */
class RankNeuronId {
public:
	//RankNeuronId() {};
	RankNeuronId(int rank, size_t neuron_id) :
		rank(rank), neuron_id(neuron_id) {
	};

	bool operator==(const RankNeuronId& other) {
		return (this->rank == other.rank &&
			this->neuron_id == other.neuron_id);
	}

	int rank;        // MPI rank of the owner
	size_t neuron_id; // Neuron id on the owner

	friend std::ostream& operator<< (std::ostream& os, const RankNeuronId& rni) {
		os << "Rank: " << rni.rank << "\t id: " << rni.neuron_id << "\n";

		return os;
	}
};


/**
 * Type for list element used to represent a synapse for synapse selection
 */
class Synapse {
public:
	Synapse(const RankNeuronId& rank_neuron_id, unsigned int synapse_id) :
		rank_neuron_id(rank_neuron_id), synapse_id(synapse_id) {
	};

	RankNeuronId rank_neuron_id;
	unsigned int synapse_id; // Id of the synapse. Used to distinguish multiple synapses between the same neuron pair,
							 // i.e., edge weight in the graph >1.
							 // E.g., edge weight "3" in the graph corresponds to the synapse IDs: 0, 1, 2
};


/**
 * Type for synapse deletion requests which are used with MPI
 */
class SynapseDeletionRequests {
public:
	SynapseDeletionRequests() : num_requests(0) {};

	// Return size
	size_t size() { return num_requests; }

	// Resize the number of requests
	void resize(size_t size) {
		num_requests = size;
		requests.resize(6 * size);
	}

	// Append request 6-tuple
	void append(size_t src_neuron_id,
		size_t tgt_neuron_id,
		size_t affected_neuron_id,
		size_t affected_element_type,
		size_t signal_type,
		size_t synapse_id) {
		num_requests++;

		requests.push_back(src_neuron_id);
		requests.push_back(tgt_neuron_id);
		requests.push_back(affected_neuron_id);
		requests.push_back(affected_element_type);
		requests.push_back(signal_type);
		requests.push_back(synapse_id);
	}

	// Get request at index "request_index"
	void get_request(size_t request_index,
		size_t& src_neuron_id,
		size_t& tgt_neuron_id,
		size_t& affected_neuron_id,
		size_t& affected_element_type,
		size_t& signal_type,
		size_t& synapse_id) {
		size_t base_index = 6 * request_index;

		src_neuron_id = requests[base_index];
		tgt_neuron_id = requests[base_index + 1];
		affected_neuron_id = requests[base_index + 2];
		affected_element_type = requests[base_index + 3];
		signal_type = requests[base_index + 4];
		synapse_id = requests[base_index + 5];
	}

	// Get pointer to data
	size_t* get_requests() { return requests.data(); }

	size_t  get_requests_size_in_bytes() { return requests.size() * sizeof(size_t); }

private:
	size_t num_requests;           // Number of synapse deletion requests
	std::vector<size_t> requests;  // Each request to delete a synapse is a 6-tuple:
								   // (src_neuron_id, tgt_neuron_id, affected_neuron_id, affected_element_type, signal_type, synapse_id)
								   // That is why requests.size() == 6*num_requests
								   // Note, a more memory-efficient implementation would use a smaller data type (not size_t)
								   // for affected_element_type, signal_type.
								   // This vector is used as MPI communication buffer
};


/**
 * Type for list element used to store pending synapse deletion
 */
struct PendingSynapseDeletion {
	RankNeuronId src_neuron_id;                               // Synapse source neuron id
	RankNeuronId tgt_neuron_id;                               // Synapse target neuron id
	RankNeuronId affected_neuron_id;                          // Neuron whose synaptic element should be set vacant
	SynapticElements::ElementType affected_element_type;  // Type of the element (axon/dendrite) to be set vacant
	SynapticElements::SignalType signal_type;             // Signal type (exc/inh) of the synapse
	unsigned int synapse_id;                              // Synapse id of the synapse to be deleted
	bool affected_element_already_deleted;                // "True" if the element to be set vacant was already deleted by the neuron owning it
														  // "False" if the element must be set vacant
};


template<class NeuronModels, class Axons, class DendritesExc, class DendritesInh>
class Neurons {

public:



	/**
	 * Map of (MPI rank; SynapseDeletionRequests)
	 * The MPI rank specifies the corresponding process
	 */
	typedef std::map<int, SynapseDeletionRequests> MapSynapseDeletionRequests;


	Neurons(size_t, Parameters);
	~Neurons();

	size_t                          get_num_neurons() const { return num_neurons; }
	Positions& get_positions() { return positions; }
	std::vector<std::string>& get_area_names() { return area_names; }
	Axons& get_axons() { return axons; }
	const DendritesExc& get_dendrites_exc() { return dendrites_exc; }
	const DendritesInh& get_dendrites_inh() { return dendrites_inh; }
	NeuronModels& get_neuron_models() { return neuron_models; }

	// NOTE: The static variables must be reset to 0 before this function can be used
	// for the synapse creation phase in the next connectivity update
	inline bool get_vacant_axon(size_t& neuron_id, double xyz_pos[], Cell::DendriteType& dendrite_type_needed) {
		static size_t i = 0, j = 0;

		double* axons_x_dims, * axons_y_dims, * axons_z_dims;
		double* axons_cnts, * axons_connected_cnts;
		SynapticElements::SignalType* axons_signal_types;
		unsigned int num_vacant_axons;
		bool found = false;

		axons_cnts = axons.get_cnts();
		axons_connected_cnts = axons.get_connected_cnts();
		axons_signal_types = axons.get_signal_types();
		axons_x_dims = positions.get_x_dims();
		axons_y_dims = positions.get_y_dims();
		axons_z_dims = positions.get_z_dims();

		while (i < num_neurons) {
			// neuron's vacant axons
			num_vacant_axons = (unsigned int)(axons_cnts[i] - axons_connected_cnts[i]);

			if (j < num_vacant_axons) {
				j++;
				found = true;
			}
			else {
				i++;
				j = 0;
			}

			// Vacant axon found
			if (found) {
				// set neuron id of vacant axon
				neuron_id = i;

				// set neuron's position
				xyz_pos[0] = axons_x_dims[i];
				xyz_pos[1] = axons_y_dims[i];
				xyz_pos[2] = axons_z_dims[i];

				// set dendrite type matching this axon
				// INHIBITORY axon
				if (SynapticElements::INHIBITORY == axons_signal_types[i]) {
					dendrite_type_needed = Cell::INHIBITORY;
				}
				// EXCITATORY axon
				else {
					dendrite_type_needed = Cell::EXCITATORY;
				}

				break;
			}
		} // while
		return found;
	}

	void init_synaptic_elements() {
		size_t i;
		double* axons_cnts = axons.get_cnts();
		double* dendrites_exc_cnts = dendrites_exc.get_cnts();
		double* dendrites_inh_cnts = dendrites_inh.get_cnts();
		SynapticElements::SignalType* dendrites_exc_signal_types = dendrites_exc.get_signal_types();
		SynapticElements::SignalType* dendrites_inh_signal_types = dendrites_inh.get_signal_types();

		/**
		 * Mark dendrites as exc./inh.
		 */
		for (i = 0; i < num_neurons; i++) {
			dendrites_exc_signal_types[i] = SynapticElements::EXCITATORY;  // Mark EXCITATORY dendrites as EXCITATORY
			dendrites_inh_signal_types[i] = SynapticElements::INHIBITORY;  // Mark INHIBITORY dendrites as INHIBITORY
		}

		// Give unbound synaptic elements as well
		{
			//            int num_axons = 1;
			//            int num_dends = 1;
			int num_axons = 0;
			int num_dends = 0;


			for (i = 0; i < num_neurons; i++) {
				axons_cnts[i] += num_axons;
				dendrites_inh_cnts[i] += num_dends;
				dendrites_exc_cnts[i] += num_dends;

				assert(axons.get_cnts()[i] >= axons.get_connected_cnts()[i]);
				assert(dendrites_exc.get_cnts()[i] >= dendrites_exc.get_connected_cnts()[i]);
				assert(dendrites_inh.get_cnts()[i] >= dendrites_inh.get_connected_cnts()[i]);
			}
		}
	}

	void update_electrical_activity(NetworkGraph& network_graph) {
		neuron_models.update_electrical_activity(network_graph, calcium);
	}

	void update_number_synaptic_elements_delta() {
		axons.update_number_elements_delta(calcium);
		dendrites_exc.update_number_elements_delta(calcium);
		dendrites_inh.update_number_elements_delta(calcium);
	}

	void update_connectivity(Octree& global_tree,
		std::vector<Octree*>& local_trees,
		NetworkGraph& network_graph,
		OctreeNode rma_buffer_branch_nodes[],
		size_t num_rma_buffer_branch_nodes,
		MPI_Win& mpi_window,
		Partition& partition,
		size_t& num_synapses_deleted, size_t& num_synapses_created) {
		//std::cout << "Before deletion:\n";
		//print_info_for_barnes_hut();

		GlobalTimers::timers.start(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);
		/**
		 * 1. Update number of synaptic elements and delete synapses if necessary
		 */
		{
#ifndef NDEBUG
			for (size_t i = 0; i < num_neurons; i++) {
				assert(axons.get_cnts()[i] >= axons.get_connected_cnts()[i]);
				assert(dendrites_exc.get_cnts()[i] >= dendrites_exc.get_connected_cnts()[i]);
				assert(dendrites_inh.get_cnts()[i] >= dendrites_inh.get_connected_cnts()[i]);
			}
#endif

			std::list<PendingSynapseDeletion> list_with_pending_deletions;
			num_synapses_deleted = 0;

			/**
			 * Create list with synapses to delete (pending synapse deletions)
			 */
			{
				SynapticElements* axons_dendsexc_dendsinh[3] = { &axons, &dendrites_exc, &dendrites_inh };
				SynapticElements* synaptic_elements;
				SynapticElements::ElementType element_type;
				SynapticElements::SignalType signal_type;
				unsigned int num_synapses_to_delete;
				size_t neuron_id;
				int i;


				// For all synaptic element types (axons, dends exc., dends inh.)
				for (i = 0; i < 3; i++) {
					synaptic_elements = axons_dendsexc_dendsinh[i];
					element_type = synaptic_elements->get_element_type();

					// For my neurons
					for (neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
						/**
						 * Create and delete synaptic elements as required.
						 * This function only deletes elements (bound and unbound), no synapses.
						 */
						 //timers.start(1);
						num_synapses_to_delete = synaptic_elements->update_number_elements(neuron_id);
						//timers.stop_and_add(1);

						/**
						 * Create a list with all pending synapse deletions.
						 * During creating this list, the possibility that neurons want to delete the same
						 * synapse is considered.
						 */
						 //timers.start(2);
						signal_type = synaptic_elements->get_signal_type(neuron_id);
						find_synapses_for_deletion(neuron_id, element_type, signal_type, num_synapses_to_delete, network_graph, list_with_pending_deletions);
						//timers.stop_and_add(2);

					} // For my neurons
				} // For all synaptic element types
			}

			/**
			 * - Go through list with pending synapse deletions and copy those into map "map_synapse_deletion_requests_outgoing"
			 *   where the other neuron affected by the deletion is not one of my neurons
			 * - Tell every rank how many deletion requests to receive from me
			 * - Prepare for corresponding number of deletion requests from every rank and receive them
			 * - Add received deletion requests to the list with pending deletions
			 * - Execute pending deletions
			 */
			{
				typename std::list<PendingSynapseDeletion>::iterator list_it, it_curr;
				typename MapSynapseDeletionRequests::iterator map_it;
				MapSynapseDeletionRequests map_synapse_deletion_requests_outgoing;
				MapSynapseDeletionRequests map_synapse_deletion_requests_incoming;
				std::vector<size_t> num_synapse_deletion_requests_for_ranks(MPIInfos::num_ranks, 0);
				std::vector<size_t> num_synapse_deletion_requests_from_ranks(MPIInfos::num_ranks, 112233);
				size_t src_neuron_id, tgt_neuron_id, affected_neuron_id, affected_element_type, signal_type, synapse_id;
				size_t num_requests, request_index;
				int target_rank, rank, mpi_requests_index, size_in_bytes;
				void* buffer;

				/**
				 * Go through list with pending synapse deletions and copy those into
				 * map "map_synapse_deletion_requests_outgoing" where the other neuron
				 * affected by the deletion is not one of my neurons
				 */

				 // All pending deletion requests
				for (list_it = list_with_pending_deletions.begin(); list_it != list_with_pending_deletions.end(); ++list_it) {
					target_rank = list_it->affected_neuron_id.rank;

					// Affected neuron of deletion request resides on different rank.
					// Thus the request needs to be communicated.
					if (target_rank != MPIInfos::my_rank) {
						map_synapse_deletion_requests_outgoing[target_rank].append(
							list_it->src_neuron_id.neuron_id,
							list_it->tgt_neuron_id.neuron_id,
							list_it->affected_neuron_id.neuron_id,
							list_it->affected_element_type,
							list_it->signal_type,
							list_it->synapse_id);
					}
				}

				/**
				 * Send to every rank the number of deletion requests it should prepare for from me.
				 * Likewise, receive the number of deletion requests that I should prepare for from every rank.
				 */

				 // Fill vector with my number of synapse deletion requests for every rank
				 // Requests to myself are kept local and not sent to myself again.
				for (map_it = map_synapse_deletion_requests_outgoing.begin(); map_it != map_synapse_deletion_requests_outgoing.end(); ++map_it) {
					rank = map_it->first;
					num_requests = (map_it->second).size();

					num_synapse_deletion_requests_for_ranks[rank] = num_requests;
				}
				// Send and receive the number of synapse deletion requests
				MPI_Alltoall((char*)num_synapse_deletion_requests_for_ranks.data(), sizeof(size_t), MPI_CHAR,
					(char*)num_synapse_deletion_requests_from_ranks.data(), sizeof(size_t), MPI_CHAR,
					MPI_COMM_WORLD);

				// Now I know how many requests I will get from every rank.
				// Allocate memory for all incoming synapse deletion requests.
				for (rank = 0; rank < MPIInfos::num_ranks; ++rank) {
					num_requests = num_synapse_deletion_requests_from_ranks[rank];
					if (0 != num_requests) {
						map_synapse_deletion_requests_incoming[rank].resize(num_requests);
					}
				}

				std::vector<MPI_Request>
					mpi_requests(map_synapse_deletion_requests_outgoing.size() + map_synapse_deletion_requests_incoming.size());

				/**
				 * Send and receive actual synapse deletion requests
				 */

				mpi_requests_index = 0;

				// Receive actual synapse deletion requests
				for (map_it = map_synapse_deletion_requests_incoming.begin(); map_it != map_synapse_deletion_requests_incoming.end(); ++map_it) {
					rank = map_it->first;
					buffer = (map_it->second).get_requests();
					size_in_bytes = (int)((map_it->second).get_requests_size_in_bytes());

					MPI_Irecv(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
					++mpi_requests_index;
				}
				// Send actual synapse deletion requests
				for (map_it = map_synapse_deletion_requests_outgoing.begin(); map_it != map_synapse_deletion_requests_outgoing.end(); ++map_it) {
					rank = map_it->first;
					buffer = (map_it->second).get_requests();
					size_in_bytes = (int)((map_it->second).get_requests_size_in_bytes());

					MPI_Isend(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
					++mpi_requests_index;
				}
				// Wait for all sends and receives to complete
				MPI_Waitall(mpi_requests_index, mpi_requests.data(), MPI_STATUSES_IGNORE);

				/**
				 * Go through all received deletion requests and add them to the list with pending requests.
				 */

				 // From smallest to largest rank that sent deletion request
				for (map_it = map_synapse_deletion_requests_incoming.begin(); map_it != map_synapse_deletion_requests_incoming.end(); ++map_it) {
					SynapseDeletionRequests& requests = map_it->second;
					int other_rank = map_it->first;
					num_requests = requests.size();

					// All requests of a rank
					for (request_index = 0; request_index < num_requests; ++request_index) {
						requests.get_request(
							request_index,
							src_neuron_id,
							tgt_neuron_id,
							affected_neuron_id,
							affected_element_type,
							signal_type,
							synapse_id);

						// Sanity check: if the affected neuron which I received the
						// synapse deletion request for is actually mine
						//
						// NOTE: This sanity check would be helpful, however, as only local neuron IDs are communicated in
						// the deletion requests, there is currently no proper way to check if I received correct requests.
						// Local neuron IDs are not unique.
						//
						//if (!MPIInfos::neuron_id_is_mine(affected_neuron_id)) {
						//    stringstream sstream;
						//    sstream << __FUNCTION__ << ": \"affected_neuron_id\": " << affected_neuron_id << " is not one of my neurons";
						//    LogMessages::print_error(sstream.str().c_str());
						//    exit(EXIT_FAILURE);
						//}

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
			} // Completing list with pending synapse deletions

			/**
			 * Now the list with pending synapse deletions contains all deletion requests
			 * of synapses that are connected to at least one of my neurons
			 *
			 * NOTE:
			 * (i)  A synapse can be connected to two of my neurons
			 * (ii) A synapse can be connected to one of my neurons and the other neuron belongs to another rank
			 */

			 /* Delete all synapses pending for deletion */
			 //print_pending_synapse_deletions(list_with_pending_deletions);
			 //timers.start(6);
			delete_synapses(list_with_pending_deletions, axons, dendrites_exc, dendrites_inh, network_graph, num_synapses_deleted);
			list_with_pending_deletions.clear(); // Empty list of pending synapse deletions
			//timers.stop_and_add(6);
#ifndef NDEBUG
			for (size_t i = 0; i < num_neurons; i++) {
				assert(axons.get_cnts()[i] >= axons.get_connected_cnts()[i]);
				assert(dendrites_exc.get_cnts()[i] >= dendrites_exc.get_connected_cnts()[i]);
				assert(dendrites_inh.get_cnts()[i] >= dendrites_inh.get_connected_cnts()[i]);
			}
#endif
		}
		GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);


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
		{
#ifndef NDEBUG
			for (size_t i = 0; i < num_neurons; i++) {
				assert(axons.get_cnts()[i] >= axons.get_connected_cnts()[i]);
				assert(dendrites_exc.get_cnts()[i] >= dendrites_exc.get_connected_cnts()[i]);
				assert(dendrites_inh.get_cnts()[i] >= dendrites_inh.get_connected_cnts()[i]);
			}
#endif

			/**
			 * Update global tree bottom-up with current number
			 * of vacant dendrites and resulting positions
			 */

			 /**********************************************************************************/

			 // Lock local RMA memory for local stores
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, MPIInfos::my_rank, MPI_MODE_NOCHECK, mpi_window);

			// Update my local trees bottom-up
			GlobalTimers::timers.start(TimerRegion::UPDATE_LOCAL_TREES);
			for (size_t i = 0; i < local_trees.size(); i++) {
				local_trees[i]->update(
					dendrites_exc.get_cnts(), dendrites_exc.get_connected_cnts(),
					dendrites_inh.get_cnts(), dendrites_inh.get_connected_cnts(),
					num_neurons);
			}
			GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_LOCAL_TREES);

			/**
			 * Exchange branch nodes
			 */
			GlobalTimers::timers.start(TimerRegion::EXCHANGE_BRANCH_NODES);
			// Copy local trees' root nodes to correct positions in receive buffer
			for (size_t i = 0; i < local_trees.size(); i++) {
				size_t global_subdomain_id = partition.get_my_subdomain_id_start() + i;
				OctreeNode& root_node = *(local_trees[i]->get_root());

				// This assignment copies memberwise
				rma_buffer_branch_nodes[global_subdomain_id] = root_node;
			}

			// Allgather in-place branch nodes from every rank
			MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, (char*)rma_buffer_branch_nodes,
				(int)local_trees.size() * sizeof(OctreeNode),
				MPI_CHAR, MPI_COMM_WORLD);
			GlobalTimers::timers.stop_and_add(TimerRegion::EXCHANGE_BRANCH_NODES);

			// Insert only received branch nodes into global tree
			// The local ones are already in the global tree
			GlobalTimers::timers.start(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);
			for (size_t i = 0; i < num_rma_buffer_branch_nodes; i++) {
				if (i < partition.get_my_subdomain_id_start() ||
					i > partition.get_my_subdomain_id_end()) {
					global_tree.insert(&rma_buffer_branch_nodes[i]);
				}
			}
			GlobalTimers::timers.stop_and_add(TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);

			// Update global tree
			GlobalTimers::timers.start(TimerRegion::UPDATE_GLOBAL_TREE);
			auto level_branches = global_tree.get_level_of_branch_nodes();

			// Only update whenever there are other branches to update
			if (level_branches > 0) {
				global_tree.update_from_level(level_branches - 1);
			}
			GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_GLOBAL_TREE);

			/*
			if (MPIInfos::my_rank == 0) {
				global_tree.get_root()->print();
			}
			 */

			 // Unlock local RMA memory and make local stores visible in public window copy
			MPI_Win_unlock(MPIInfos::my_rank, mpi_window);

			/**********************************************************************************/

			// Makes sure that all ranks finished their local access epoch
			// before a remote origin opens an access epoch
			MPI_Barrier(MPI_COMM_WORLD);


			/**
			 * Find target neuron for every vacant axon
			 */
			GlobalTimers::timers.start(TimerRegion::FIND_TARGET_NEURONS);

			double* axons_cnts, * axons_connected_cnts;
			SynapticElements::SignalType* axons_signal_types;
			double* dendrites_cnts, * dendrites_connected_cnts;
			double* axons_x_dims, * axons_y_dims, * axons_z_dims;
			double axon_xyz_pos[3];
			unsigned int num_vacant_axons;
			int num_axons_connected_increment;
			size_t target_neuron_id;
			Cell::DendriteType dendrite_type_needed;
			bool target_neuron_found;
			int target_rank;
			MapSynapseCreationRequests map_synapse_creation_requests_outgoing;
			MapSynapseCreationRequests map_synapse_creation_requests_incoming;

			axons_cnts = axons.get_cnts();
			axons_connected_cnts = axons.get_connected_cnts();
			axons_signal_types = axons.get_signal_types();
			axons_x_dims = positions.get_x_dims();
			axons_y_dims = positions.get_y_dims();
			axons_z_dims = positions.get_z_dims();

			// For my neurons
			for (size_t neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
				// Number of vacant axons
				num_vacant_axons = (unsigned int)(axons_cnts[neuron_id] - axons_connected_cnts[neuron_id]);
				assert(num_vacant_axons >= 0);

				// INHIBITORY axon
				if (SynapticElements::INHIBITORY == axons_signal_types[neuron_id]) {
					dendrite_type_needed = Cell::INHIBITORY;
					//dendrites_cnts           = dendrites_inh.get_cnts();
					//dendrites_connected_cnts = dendrites_inh.get_connected_cnts();
					//num_axons_connected_increment = -1;
				}
				// EXCITATORY axon
				else {
					dendrite_type_needed = Cell::EXCITATORY;
					//dendrites_cnts           = dendrites_exc.get_cnts();
					//dendrites_connected_cnts = dendrites_exc.get_connected_cnts();
					//num_axons_connected_increment = +1;
				}

				// Position of current neuron
				axon_xyz_pos[0] = axons_x_dims[neuron_id];
				axon_xyz_pos[1] = axons_y_dims[neuron_id];
				axon_xyz_pos[2] = axons_z_dims[neuron_id];

				//timers.start(5);
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
					target_neuron_found = global_tree.find_target_neuron(neuron_id, axon_xyz_pos, dendrite_type_needed, target_neuron_id, target_rank);
					if (target_neuron_found) {
						//std::cout << __func__ << ": Trying to connect: (" << neuron_id << " -> " << target_neuron_id << ")\n";

						/*
						 * Append request for synapse creation to rank "target_rank"
						 * Note that "target_rank" could also be my own rank.
						 */
						map_synapse_creation_requests_outgoing[target_rank].append(neuron_id, target_neuron_id, dendrite_type_needed);
					}
					else {
						//std::cout << __func__ << ": No target neuron for connecting found (no vacant dendrites of correct signal type in octree)\n";
					}
				} /* all vacant axons of a neuron */
				//timers.stop_and_add(5);
			} /* my neurons */

			GlobalTimers::timers.stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

			// Make cache empty for next connectivity update
			GlobalTimers::timers.start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
			global_tree.empty_remote_nodes_cache();
			GlobalTimers::timers.stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
			GlobalTimers::timers.start(TimerRegion::CREATE_SYNAPSES);
			{
				typename MapSynapseCreationRequests::iterator it;
				std::vector<size_t> num_synapse_requests_for_ranks(MPIInfos::num_ranks, 0);
				std::vector<size_t> num_synapse_requests_from_ranks(MPIInfos::num_ranks, 112233);
				size_t num_requests, request_index, source_neuron_id, target_neuron_id, dendrite_type_needed;
				int mpi_requests_index, rank, source_rank, target_rank, size_in_bytes;
				void* buffer;

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
				  // Fill vector with my number of synapse requests for every rank (including me)
				for (it = map_synapse_creation_requests_outgoing.begin(); it != map_synapse_creation_requests_outgoing.end(); it++) {
					rank = it->first;
					num_requests = (it->second).size();

					num_synapse_requests_for_ranks[rank] = num_requests;
				}
				// Send and receive the number of synapse requests
				MPI_Alltoall((char*)num_synapse_requests_for_ranks.data(), sizeof(size_t), MPI_CHAR,
					(char*)num_synapse_requests_from_ranks.data(), sizeof(size_t), MPI_CHAR,
					MPI_COMM_WORLD);

				// Now I know how many requests I will get from every rank.
				// Allocate memory for all incoming synapse requests.
				for (rank = 0; rank < MPIInfos::num_ranks; rank++) {
					num_requests = num_synapse_requests_from_ranks[rank];
					if (0 != num_requests) {
						map_synapse_creation_requests_incoming[rank].resize(num_requests);
					}
				}

				std::vector<MPI_Request>
					mpi_requests(map_synapse_creation_requests_outgoing.size() + map_synapse_creation_requests_incoming.size());

				/**
				 * Send and receive actual synapse requests
				 */
				mpi_requests_index = 0;

				// Receive actual synapse requests
				for (it = map_synapse_creation_requests_incoming.begin(); it != map_synapse_creation_requests_incoming.end(); it++) {
					rank = it->first;
					buffer = (it->second).get_requests();
					size_in_bytes = (int)((it->second).get_requests_size_in_bytes());

					MPI_Irecv(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
					mpi_requests_index++;
				}
				// Send actual synapse requests
				for (it = map_synapse_creation_requests_outgoing.begin(); it != map_synapse_creation_requests_outgoing.end(); it++) {
					rank = it->first;
					buffer = (it->second).get_requests();
					size_in_bytes = (int)((it->second).get_requests_size_in_bytes());

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
				for (it = map_synapse_creation_requests_incoming.begin(); it != map_synapse_creation_requests_incoming.end(); it++) {
					source_rank = it->first;
					SynapseCreationRequests& requests = it->second;
					num_requests = requests.size();

					// All requests of a rank
					for (request_index = 0; request_index < num_requests; request_index++) {
						requests.get_request(request_index, source_neuron_id, target_neuron_id, dendrite_type_needed);

						// Sanity check: if the request received is targeted for me
						if (target_neuron_id >= num_neurons) {
							std::stringstream sstream;
							sstream << __FUNCTION__ << ": \"target_neuron_id\": " << target_neuron_id << " exceeds my neuron ids";
							LogMessages::print_error(sstream.str().c_str());
							exit(EXIT_FAILURE);
						}
						// INHIBITORY dendrite requested
						if (Cell::INHIBITORY == dendrite_type_needed) {
							dendrites_cnts = dendrites_inh.get_cnts();
							dendrites_connected_cnts = dendrites_inh.get_connected_cnts();
							num_axons_connected_increment = -1;
						}
						// EXCITATORY dendrite requested
						else {
							dendrites_cnts = dendrites_exc.get_cnts();
							dendrites_connected_cnts = dendrites_exc.get_connected_cnts();
							num_axons_connected_increment = +1;
						}

						// Target neuron has still dendrite available, so connect
						assert(dendrites_cnts[target_neuron_id] - dendrites_connected_cnts[target_neuron_id] >= 0);
						if ((unsigned int)(dendrites_cnts[target_neuron_id] - dendrites_connected_cnts[target_neuron_id])) {

							// Increment num of connected dendrites
							dendrites_connected_cnts[target_neuron_id]++;

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
				for (it = map_synapse_creation_requests_outgoing.begin(); it != map_synapse_creation_requests_outgoing.end(); it++) {
					rank = it->first;
					buffer = (it->second).get_responses();
					size_in_bytes = (int)((it->second).get_responses_size_in_bytes());

					MPI_Irecv(buffer, size_in_bytes, MPI_CHAR, rank, 0, MPI_COMM_WORLD, &mpi_requests[mpi_requests_index]);
					mpi_requests_index++;
				}
				// Send responses
				for (it = map_synapse_creation_requests_incoming.begin(); it != map_synapse_creation_requests_incoming.end(); it++) {
					rank = it->first;
					buffer = (it->second).get_responses();
					size_in_bytes = (int)((it->second).get_responses_size_in_bytes());

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
				char connected;
				for (it = map_synapse_creation_requests_outgoing.begin(); it != map_synapse_creation_requests_outgoing.end(); it++) {
					target_rank = it->first;
					SynapseCreationRequests& requests = it->second;
					num_requests = requests.size();

					// All responses from a rank
					for (request_index = 0; request_index < num_requests; request_index++) {
						requests.get_response(request_index, connected);
						requests.get_request(request_index, source_neuron_id, target_neuron_id, dendrite_type_needed);

						// Request to form synapse succeeded
						if (connected) {
							// Increment num of connected axons
							axons_connected_cnts[source_neuron_id]++;
							num_synapses_created++;

							assert(axons.get_cnts()[source_neuron_id] - axons.get_connected_cnts()[source_neuron_id] >= 0);

							// I have already created the synapse in the network
							// if the response comes from myself
							if (target_rank != MPIInfos::my_rank) {
								// Update network
								num_axons_connected_increment = (Cell::INHIBITORY == dendrite_type_needed) ? -1 : +1;
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
#ifndef NDEBUG
			for (size_t i = 0; i < num_neurons; i++) {
				assert(axons.get_cnts()[i] >= axons.get_connected_cnts()[i]);
				assert(dendrites_exc.get_cnts()[i] >= dendrites_exc.get_connected_cnts()[i]);
				assert(dendrites_inh.get_cnts()[i] >= dendrites_inh.get_connected_cnts()[i]);
			}
#endif
		}


	}

	void print_sums_of_synapses_and_elements_to_log_file_on_rank_0(size_t step, LogFiles& log_file, Parameters& params, size_t sum_synapses_deleted, size_t sum_synapses_created) {
		using namespace std;

		unsigned int sum_axons_exc_cnts, sum_axons_exc_connected_cnts;
		unsigned int sum_axons_inh_cnts, sum_axons_inh_connected_cnts;
		unsigned int sum_dends_exc_cnts, sum_dends_exc_connected_cnts;
		unsigned int sum_dends_inh_cnts, sum_dends_inh_connected_cnts;
		unsigned int sum_axons_exc_vacant, sum_axons_inh_vacant;
		unsigned int sum_dends_exc_vacant, sum_dends_inh_vacant;
		double* cnts, * connected_cnts;
		SynapticElements::SignalType* signal_types;

		// My vacant axons (exc./inh.)
		sum_axons_exc_cnts = sum_axons_exc_connected_cnts = 0;
		sum_axons_inh_cnts = sum_axons_inh_connected_cnts = 0;
		cnts = axons.get_cnts();
		connected_cnts = axons.get_connected_cnts();
		signal_types = axons.get_signal_types();
		for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
			if (SynapticElements::EXCITATORY == signal_types[neuron_id]) {
				sum_axons_exc_cnts += (unsigned int)cnts[neuron_id];
				sum_axons_exc_connected_cnts += (unsigned int)connected_cnts[neuron_id];
			}
			else {
				sum_axons_inh_cnts += (unsigned int)cnts[neuron_id];
				sum_axons_inh_connected_cnts += (unsigned int)connected_cnts[neuron_id];
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
			sum_dends_exc_cnts += (unsigned int)cnts[neuron_id];
			sum_dends_exc_connected_cnts += (unsigned int)connected_cnts[neuron_id];
		}
		sum_dends_exc_vacant = sum_dends_exc_cnts - sum_dends_exc_connected_cnts;

		// Inh.
		sum_dends_inh_cnts = sum_dends_inh_connected_cnts = 0;
		cnts = dendrites_inh.get_cnts();
		connected_cnts = dendrites_inh.get_connected_cnts();
		for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
			sum_dends_inh_cnts += (unsigned int)cnts[neuron_id];
			sum_dends_inh_connected_cnts += (unsigned int)connected_cnts[neuron_id];
		}
		sum_dends_inh_vacant = sum_dends_inh_cnts - sum_dends_inh_connected_cnts;

		// Get global sums at rank 0
		unsigned int sums_local[6]{ sum_axons_exc_vacant,
									sum_axons_inh_vacant,
									sum_dends_exc_vacant,
									sum_dends_inh_vacant,
									(unsigned int)sum_synapses_deleted,
									(unsigned int)sum_synapses_created };
		unsigned int sums_global[6]{ 123 }; // Init all to zero

		MPI_Reduce(sums_local, sums_global, 6, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);


		// Output data
		if (0 == MPIInfos::my_rank) {
			ofstream* file = log_file.get_file(0);
			int cwidth = 20;  // Column width

			// Write headers to file if not already done so
			if (0 == step) {
				*file << params << endl;
				*file << "# SUMS OVER ALL NEURONS\n";
				*file << left
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
			*file << left
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
	void print_neurons_overview_to_log_file_on_rank_0(size_t step, LogFiles& log_file, Parameters& params) {
		using namespace std;

		StatisticalMeasures<double> calcium_statistics =
			global_statistics(calcium, num_neurons, params.num_neurons, 0, MPI_COMM_WORLD);

		StatisticalMeasures<double> activity_statistics =
			global_statistics(neuron_models.get_x(), num_neurons, params.num_neurons, 0, MPI_COMM_WORLD);

		// Output data
		if (0 == MPIInfos::my_rank) {
			ofstream* file = log_file.get_file(0);
			int cwidth = 16;  // Column width

			// Write headers to file if not already done so
			if (0 == step) {
				*file << params << endl;
				*file << "# ALL NEURONS\n";
				*file << left
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
			*file << left
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

	void print_network_graph_to_log_file(LogFiles& log_file,
		const NetworkGraph& network_graph,
		const Parameters& params,
		const NeuronIdMap& neuron_id_map) {
		using namespace std;
		ofstream* file;

		file = log_file.get_file(0);

		// Write output format to file
		*file << "# " << params.num_neurons << endl; // Total number of neurons
		*file << "# <target neuron id> <source neuron id> <weight>" << endl;

		// Write network graph to file
		//*file << network_graph << endl;
		network_graph.print(*file, neuron_id_map);
	}

	void print_positions_to_log_file(LogFiles& log_file, Parameters& params,
		const NeuronIdMap& neuron_id_map) {
		using namespace std;

		double* xyz_dims[3];
		ofstream* file;

		file = log_file.get_file(0);

		// Write total number of neurons to log file
		*file << "# " << params.num_neurons << endl;
		*file << "# " << "<global id> <pos x> <pos y> <pos z> <area>" << endl;

		xyz_dims[0] = positions.get_x_dims();
		xyz_dims[1] = positions.get_y_dims();
		xyz_dims[2] = positions.get_z_dims();

		// Print global ids, positions, and areas of local neurons
		bool ret;
		size_t glob_id;
		NeuronIdMap::RankNeuronId rank_neuron_id;

		rank_neuron_id.rank = MPIInfos::my_rank;
		*file << std::fixed << std::setprecision(6);
		for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
			rank_neuron_id.neuron_id = neuron_id;
			ret = neuron_id_map.rank_neuron_id2glob_id(rank_neuron_id, glob_id);
			assert(ret);

			*file << glob_id << " "
				<< xyz_dims[0][neuron_id] << " "
				<< xyz_dims[1][neuron_id] << " "
				<< xyz_dims[2][neuron_id] << " "
				<< area_names[neuron_id] << "\n";
		}
		*file << endl;
		*file << std::defaultfloat;
	}

	void print() {
		using namespace std;

		// Column widths
		int cwidth_left = 6;
		int cwidth = 16;

		// Heading
		cout << left << setw(cwidth_left) << "gid" << setw(cwidth) << "x" << setw(cwidth) << "AP";
		cout << setw(cwidth) << "refrac" << setw(cwidth) << "C" << setw(cwidth) << "A" << setw(cwidth) << "D_ex" << setw(cwidth) << "D_in" << "\n";

		// Values
		for (size_t i = 0; i < num_neurons; i++) {
			cout << left << setw(cwidth_left) << i << setw(cwidth) << neuron_models.get_x(i) << setw(cwidth) << neuron_models.get_fired(i);
			cout << setw(cwidth) << neuron_models.get_refrac(i) << setw(cwidth) << calcium[i] << setw(cwidth) << axons.get_cnt(i);
			cout << setw(cwidth) << dendrites_exc.get_cnt(i) << setw(cwidth) << dendrites_inh.get_cnt(i) << "\n";
		}
	}

	void print_info_for_barnes_hut() {
		using namespace std;

		double* x_dims = positions.get_x_dims();
		double* y_dims = positions.get_y_dims();
		double* z_dims = positions.get_z_dims();

		double* axons_cnts = axons.get_cnts();
		double* dendrites_exc_cnts = dendrites_exc.get_cnts();
		double* dendrites_inh_cnts = dendrites_inh.get_cnts();

		double* axons_connected_cnts = axons.get_connected_cnts();
		double* dendrites_exc_connected_cnts = dendrites_exc.get_connected_cnts();
		double* dendrites_inh_connected_cnts = dendrites_inh.get_connected_cnts();

		// Column widths
		int cwidth_small = 8;
		int cwidth_medium = 16;
		int cwidth_big = 27;

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

private:
	template<typename T>
	struct StatisticalMeasures {
		T min;
		T max;
		double avg;
		double var;
		double std;
	};

	template<typename T>
	StatisticalMeasures<T> global_statistics(const T* local_values, size_t num_local_values, size_t total_num_values, int root, MPI_Comm mpi_comm) {
		auto result = std::minmax_element(&local_values[0], &local_values[num_neurons]);
		T my_min = *result.first;
		T my_max = *result.second;

		double my_avg = std::accumulate(&local_values[0], &local_values[num_neurons], double{ 0 });
		my_avg /= total_num_values;

		// Get global min and max at rank "root"
		double d_my_min = double{ my_min };
		double d_my_max = double{ my_max };
		double d_min, d_max;
		MPI_Reduce(&d_my_min, &d_min, 1, MPI_DOUBLE, MPI_MIN, root, mpi_comm);
		MPI_Reduce(&d_my_max, &d_max, 1, MPI_DOUBLE, MPI_MAX, root, mpi_comm);

		// Get global avg at all ranks (needed for variance)
		double avg;
		MPI_Allreduce(&my_avg, &avg, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

		/**
		 * Calc variance
		 */
		double my_var = 0;
		for (size_t neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
			my_var += (local_values[neuron_id] - avg) * (local_values[neuron_id] - avg);
		}
		my_var /= total_num_values;

		// Get global variance at rank "root"
		double var;
		MPI_Reduce(&my_var, &var, 1, MPI_DOUBLE, MPI_SUM, root, mpi_comm);

		// Calc standard deviation
		double std = sqrt(var);

		return { T{d_min}, T{d_max}, avg, var, std };
	}

	/**
	 * Returns iterator to randomly chosen synapse from list
	 */
	void select_synapse(std::list<Synapse>& list, typename std::list<Synapse>::iterator& it) {
		// Point to first synapse
		it = list.begin();

		// Draw random number from [0,1)
		double random_number = random_number_distribution(random_number_generator);

		// Make iterator point to selected element
		std::advance(it, static_cast<int>(list.size() * random_number));

		//    	The code above does the same as the following code
		//
		//        // Point to first synapse
		//        it = list.begin();
		//
		//        // Does list contain synapses?
		//        if (!list.empty()) {
		//            double sum_probabilities, random_number, probability;
		//
		//            // Same probability for being selected for every synapse
		//            probability = ((double) 1) / list.size();
		//
		//            // Draw random number from [0,1]
		//            random_number = random_number_distribution(random_number_generator);
		//
		//            /**
		//             * Also check for it != list.end() to account for that, due to numeric inaccuracies in summation,
		//             * it might happen that random_number > sum_probabilities in the end
		//             */
		//            sum_probabilities = probability;
		//            ++it; // Point to second synapse
		//            while (random_number > sum_probabilities && it != list.end())
		//            {
		//                sum_probabilities += probability;
		//                ++it;
		//            }
		//            --it; // Undo ++it before or in loop to get correct synapse
		//        }

	}

	void add_synapse_to_pending_deletions(const RankNeuronId& src_neuron_id,
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
	void find_synapses_for_deletion(size_t neuron_id,
		SynapticElements::ElementType element_type,
		SynapticElements::SignalType signal_type,
		unsigned int num_synapses_to_delete,
		NetworkGraph& network_graph,
		std::list<PendingSynapseDeletion>& list_pending_deletions) {
		std::list<Synapse> list_synapses;
		typename std::list<Synapse>::iterator synapse_selected;
		unsigned int num_synapses_selected;
		unsigned int synapse_id;

		//std::cout << __func__ << ": neuron_id: " << neuron_id << " num_synapses_to_delete: " << num_synapses_to_delete << std::endl;

		// Only do something if necessary
		if (0 != num_synapses_to_delete) {

			/**
			 * Bound elements to delete: Axons
			 */
			if (SynapticElements::AXON == element_type) {
				const NetworkGraph::Edges& out_edges = network_graph.get_out_edges(neuron_id);
				NetworkGraph::Edges::const_iterator it;

				/**
				 * Create list with synapses
				 */
				 // Walk through outgoing edges
				for (it = out_edges.begin(); it != out_edges.end(); ++it) {
					/**
					 * Create "edge weight" number of synapses and add them to the synapse list
					 *
					 * NOTE: We take abs(it->second) here as INHIBITORY synapses have count < 0
					 */
					for (synapse_id = 0; synapse_id < (unsigned int)abs(it->second); ++synapse_id) {
						RankNeuronId rank_neuron_id(it->first.first, it->first.second);
						list_synapses.push_back(Synapse(rank_neuron_id, synapse_id));
					}
				}

				/**
				 * Select synapses for deletion
				 */
				{
					bool valid = num_synapses_to_delete <= list_synapses.size();
					if (!valid) {
						std::cout << __func__
							<< "num_synapses_to_delete (" << num_synapses_to_delete << ") "
							<< "> "
							<< "list_synapses.size() (" << list_synapses.size() << ")\n";
					}
				}
				assert(num_synapses_to_delete <= list_synapses.size());
				for (num_synapses_selected = 0; num_synapses_selected < num_synapses_to_delete; ++num_synapses_selected) {
					// Randomly select synapse for deletion
					select_synapse(list_synapses, synapse_selected);
					assert(synapse_selected != list_synapses.end()); // Make sure that valid synapse was selected

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
			 * Bound elements to delete: EXCITATORY dendrites
			 */
			if (SynapticElements::DENDRITE == element_type && SynapticElements::EXCITATORY == signal_type) {
				const NetworkGraph::Edges& in_edges = network_graph.get_in_edges(neuron_id);
				NetworkGraph::Edges::const_iterator it;

				/**
				 * Create list with synapses
				 */
				 // Walk through ingoing edges
				for (it = in_edges.begin(); it != in_edges.end(); ++it) {
					/**
					 * Create "edge weight" number of synapses and add them to the synapse list
					 *
					 * NOTE: We take positive entries only as those are EXCITATORY synapses
					 */
					if (it->second > 0) {
						for (synapse_id = 0; synapse_id < (unsigned int)it->second; ++synapse_id) {
							RankNeuronId rank_neuron_id(it->first.first, it->first.second);
							list_synapses.push_back(Synapse(rank_neuron_id, synapse_id));
						}
					}
				}

				/**
				 * Select synapses for deletion
				 */
				{
					bool valid = num_synapses_to_delete <= list_synapses.size();
					if (!valid) {
						std::cout << __func__
							<< "num_synapses_to_delete (" << num_synapses_to_delete << ") "
							<< "> "
							<< "list_synapses.size() (" << list_synapses.size() << ")\n";
					}
				}
				assert(num_synapses_to_delete <= list_synapses.size());
				for (num_synapses_selected = 0; num_synapses_selected < num_synapses_to_delete; ++num_synapses_selected) {
					// Randomly select synapse for deletion
					select_synapse(list_synapses, synapse_selected);
					assert(synapse_selected != list_synapses.end()); // Make sure that valid synapse was selected

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
			 * Bound elements to delete: INHIBITORY dendrites
			 */
			if (SynapticElements::DENDRITE == element_type && SynapticElements::INHIBITORY == signal_type) {
				const NetworkGraph::Edges& in_edges = network_graph.get_in_edges(neuron_id);
				NetworkGraph::Edges::const_iterator it;

				/**
				 * Create list with synapses
				 */
				 // Walk through ingoing edges
				for (it = in_edges.begin(); it != in_edges.end(); ++it) {

					/**
					 * Create "edge weight" number of synapses and add them to the synapse list
					 *
					 * NOTE: We take negative entries only as those are INHIBITORY synapses
					 */
					if (it->second < 0) {
						for (synapse_id = 0; synapse_id < (unsigned int)abs(it->second); ++synapse_id) {
							RankNeuronId rank_neuron_id(it->first.first, it->first.second);
							list_synapses.push_back(Synapse(rank_neuron_id, synapse_id));
						}
					}
				}

				/**
				 * Select synapses for deletion
				 */
				{
					bool valid = num_synapses_to_delete <= list_synapses.size();
					if (!valid) {
						std::cout << __func__
							<< "num_synapses_to_delete (" << num_synapses_to_delete << ") "
							<< "> "
							<< "list_synapses.size() (" << list_synapses.size() << ")\n";
					}
				}
				assert(num_synapses_to_delete <= list_synapses.size());
				for (num_synapses_selected = 0; num_synapses_selected < num_synapses_to_delete; ++num_synapses_selected) {
					// Randomly select synapse for deletion
					select_synapse(list_synapses, synapse_selected);
					assert(synapse_selected != list_synapses.end()); // Make sure that valid synapse was selected

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
	}

	void print_pending_synapse_deletions(std::list<PendingSynapseDeletion>& list) {
		typename std::list<PendingSynapseDeletion>::iterator it;

		for (it = list.begin(); it != list.end(); it++) {
			std::cout << "src_neuron_id: " << it->src_neuron_id << "\n";
			std::cout << "tgt_neuron_id: " << it->tgt_neuron_id << "\n";
			std::cout << "affected_neuron_id: " << it->affected_neuron_id << "\n";
			std::cout << "affected_element_type: " << it->affected_element_type << "\n";
			std::cout << "signal_type: " << it->signal_type << "\n";
			std::cout << "synapse_id: " << it->synapse_id << "\n";
			std::cout << "affected_element_already_deleted: " << it->affected_element_already_deleted << "\n" << std::endl;
		}
	}

	void delete_synapses(std::list<PendingSynapseDeletion>& list,
		SynapticElements& axons,
		SynapticElements& dendrites_exc,
		SynapticElements& dendrites_inh,
		NetworkGraph& network_graph,
		size_t& num_synapses_deleted) {
#ifndef NDEBUG
		for (size_t i = 0; i < num_neurons; i++) {
			assert(axons.get_cnts()[i] >= axons.get_connected_cnts()[i]);
			assert(dendrites_exc.get_cnts()[i] >= dendrites_exc.get_connected_cnts()[i]);
			assert(dendrites_inh.get_cnts()[i] >= dendrites_inh.get_connected_cnts()[i]);
		}
#endif

		typename std::list<PendingSynapseDeletion>::iterator it;
		double* axons_connected_cnts = axons.get_connected_cnts();
		double* dendrites_exc_connected_cnts = dendrites_exc.get_connected_cnts();
		double* dendrites_inh_connected_cnts = dendrites_inh.get_connected_cnts();

		/* Execute pending synapse deletions */
		for (it = list.begin(); it != list.end(); ++it) {
			// Pending synapse deletion is valid (not completely) if source or
			// target neuron belong to me. To be completely valid, things such as
			// the neuron id need to be validated as well.
			assert(it->src_neuron_id.rank == MPIInfos::my_rank || it->tgt_neuron_id.rank == MPIInfos::my_rank);

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
			 // EXCITATORY synapses have positive count, so decrement
			int weight_increment;
			if (SynapticElements::EXCITATORY == it->signal_type) {
				weight_increment = -1;
			}
			// INHIBITORY synapses have negative count, so increment
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
			if (it->affected_neuron_id.rank == MPIInfos::my_rank && !it->affected_element_already_deleted) {
				if (SynapticElements::AXON == it->affected_element_type) {
					--axons_connected_cnts[it->affected_neuron_id.neuron_id];
				}
				else if ((SynapticElements::DENDRITE == it->affected_element_type) &&
					(SynapticElements::EXCITATORY == it->signal_type)) {
					--dendrites_exc_connected_cnts[it->affected_neuron_id.neuron_id];
#ifndef NDEBUG
					if (!(dendrites_exc.get_cnts()[it->affected_neuron_id.neuron_id] >=
						dendrites_exc.get_connected_cnts()[it->affected_neuron_id.neuron_id])) {
						std::cout << "neuron_id: " << it->affected_neuron_id.neuron_id << "\n"
							<< "cnt: " << dendrites_exc.get_cnts()[it->affected_neuron_id.neuron_id] << "\n"
							<< "connected_cnt: " << dendrites_exc.get_connected_cnts()[it->affected_neuron_id.neuron_id] << "\n";
					}
#endif

		}
				else if ((SynapticElements::DENDRITE == it->affected_element_type) &&
					(SynapticElements::INHIBITORY == it->signal_type)) {
					--dendrites_inh_connected_cnts[it->affected_neuron_id.neuron_id];
				}
				else {
					std::cout << "Invalid list element for pending synapse deletion." << std::endl;
				}
	}
#ifndef NDEBUG
			for (size_t i = 0; i < num_neurons; i++) {
				assert(axons.get_cnts()[i] >= axons.get_connected_cnts()[i]);
				assert(dendrites_exc.get_cnts()[i] >= dendrites_exc.get_connected_cnts()[i]);
				assert(dendrites_inh.get_cnts()[i] >= dendrites_inh.get_connected_cnts()[i]);
			}
#endif
}
#ifndef NDEBUG
		for (size_t i = 0; i < num_neurons; i++) {
			assert(axons.get_cnts()[i] >= axons.get_connected_cnts()[i]);
			assert(dendrites_exc.get_cnts()[i] >= dendrites_exc.get_connected_cnts()[i]);
			assert(dendrites_inh.get_cnts()[i] >= dendrites_inh.get_connected_cnts()[i]);
		}
#endif
	}


	size_t num_neurons;       // Local number of neurons

	NeuronModels neuron_models;

	Axons axons;
	DendritesExc dendrites_exc;
	DendritesInh dendrites_inh;

	Positions positions;  // Position of every neuron
	double* calcium;      // Intracellular calcium concentration of every neuron
	std::vector<std::string> area_names;  // Area name of every neuron


	// Randpm number generator for this class (C++11)
	std::mt19937& random_number_generator;
	// Random number distribution used together with "random_number_generator" (C++11)
	// Uniform distribution for interval [0, 1) (see constructor for initialization)
	std::uniform_real_distribution<double> random_number_distribution;

	// Timers for timing code sections
	//Timers timers;
};


template<class NeuronModels, class Axons, class DendritesExc, class DendritesInh>
Neurons<NeuronModels, Axons, DendritesExc, DendritesInh>::Neurons(size_t s, Parameters params) :
	num_neurons(s),
	neuron_models(s, params.x_0, params.tau_x, params.k, params.tau_C, params.beta, params.h, params.refrac_time),
	axons(SynapticElements::AXON, s, params.eta_A, params.C_target, params.nu, params.vacant_retract_ratio),
	dendrites_exc(SynapticElements::DENDRITE, s, params.eta_D_ex, params.C_target, params.nu, params.vacant_retract_ratio),
	dendrites_inh(SynapticElements::DENDRITE, s, params.eta_D_in, params.C_target, params.nu, params.vacant_retract_ratio),
	positions(s),
	area_names(s),
	random_number_distribution(0.0, std::nextafter(1.0, 2.0)),
	random_number_generator(RandomHolder<Neurons<NeuronModels, Axons, DendritesExc, DendritesInh>>::get_random_generator()) {
	calcium = new double[num_neurons];

	// Init member variables
	for (size_t i = 0; i < num_neurons; i++) {
		// Set calcium concentration
		calcium[i] = neuron_models.get_beta() * neuron_models.get_fired(i);
	}
}

template<class NeuronModels, class Axons, class DendritesExc, class DendritesInh>
Neurons<NeuronModels, Axons, DendritesExc, DendritesInh>::~Neurons() {
	delete[] calcium;
}

#endif	/* NEURONS_H */
