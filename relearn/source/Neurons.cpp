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
#include "Random.h"
#include "RelearnException.h"
#include "SynapseCreationRequests.h"

#include <algorithm>
#include <array>
#include <optional>

Neurons::Neurons(const Partition& partition, std::unique_ptr<NeuronModels> model)
    : partition(&partition)
    , neuron_model(std::move(model))
    , axons(ElementType::AXON, SynapticElements::default_eta_Axons)
    , dendrites_exc(ElementType::DENDRITE, SynapticElements::default_eta_Dendrites_exc)
    , dendrites_inh(ElementType::DENDRITE, SynapticElements::default_eta_Dendrites_inh)
    // NOLINTNEXTLINE
    , random_number_distribution(0.0, std::nextafter(1.0, 2.0)) {
}

// NOTE: The static variables must be reset to 0 before this function can be used
// for the synapse creation phase in the next connectivity update
std::tuple<bool, size_t, Vec3d, SignalType> Neurons::get_vacant_axon() const noexcept {
    static size_t i = 0;
    static size_t j = 0;

    size_t neuron_id{ Constants::uninitialized };
    Vec3d xyz_pos;
    SignalType dendrite_type_needed;

    const std::vector<double>& axons_cnts = axons.get_cnts();
    const std::vector<unsigned int>& axons_connected_cnts = axons.get_connected_cnts();
    const std::vector<SignalType>& axons_signal_types = axons.get_signal_types();
    const std::vector<double>& axons_x_dims = positions.get_x_dims();
    const std::vector<double>& axons_y_dims = positions.get_y_dims();
    const std::vector<double>& axons_z_dims = positions.get_z_dims();

    while (i < num_neurons) {
        // neuron's vacant axons
        const auto num_vacant_axons = static_cast<unsigned int>(axons_cnts[i]) - axons_connected_cnts[i];

        if (j < num_vacant_axons) {
            j++;
            // Vacant axon found
            // set neuron id of vacant axon
            neuron_id = i;

            // set neuron's position
            xyz_pos.set_x(axons_x_dims[i]);
            xyz_pos.set_y(axons_y_dims[i]);
            xyz_pos.set_z(axons_z_dims[i]);

            // set dendrite type matching this axon
            // DendriteType::INHIBITORY axon
            if (SignalType::INHIBITORY == axons_signal_types[i]) {
                dendrite_type_needed = SignalType::INHIBITORY;
            }
            // DendriteType::EXCITATORY axon
            else {
                dendrite_type_needed = SignalType::EXCITATORY;
            }

            return std::make_tuple(true, neuron_id, xyz_pos, dendrite_type_needed);
        }

        i++;
        j = 0;
    } // while

    return std::make_tuple(false, neuron_id, xyz_pos, dendrite_type_needed);
}

void Neurons::init_synaptic_elements(const NetworkGraph& network_graph) {
    // Give unbound synaptic elements as well
    const double num_axons_offset = 0;
    const double num_dends_offset = 0;

    const std::vector<double>& axons_cnts = axons.get_cnts();
    const std::vector<double>& dendrites_inh_cnts = dendrites_inh.get_cnts();
    const std::vector<double>& dendrites_exc_cnts = dendrites_exc.get_cnts();

    for (auto i = 0; i < num_neurons; i++) {
        const size_t axon_connections = network_graph.get_num_out_edges(i);
        const size_t dendrites_ex_connections = network_graph.get_num_in_edges_ex(i);
        const size_t dendrites_in_connections = network_graph.get_num_in_edges_in(i);

        axons.update_cnt(i, axon_connections);
        dendrites_exc.update_cnt(i, dendrites_ex_connections);
        dendrites_inh.update_cnt(i, dendrites_in_connections);

        axons.update_conn_cnt(i, axon_connections);
        dendrites_exc.update_conn_cnt(i, dendrites_ex_connections);
        dendrites_inh.update_conn_cnt(i, dendrites_in_connections);

        RelearnException::check(axons_cnts[i] >= axons.get_connected_cnts()[i], "Error is with: %d", i);
        RelearnException::check(dendrites_inh_cnts[i] >= dendrites_inh.get_connected_cnts()[i], "Error is with: %d", i);
        RelearnException::check(dendrites_exc_cnts[i] >= dendrites_exc.get_connected_cnts()[i], "Error is with: %d", i);
    }
}

size_t Neurons::delete_synapses(NetworkGraph& network_graph) {
    /**
	* 1. Update number of synaptic elements and delete synapses if necessary
	*/

    GlobalTimers::timers.start(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

    /**
	* Create list with synapses to delete (pending synapse deletions)
	*/
    std::list<PendingSynapseDeletion> pending_deletions;

    // For all synaptic element types (axons, dends exc., dends inh.)
    delete_synapses_find_synapses(axons, network_graph, pending_deletions);
    delete_synapses_find_synapses(dendrites_exc, network_graph, pending_deletions);
    delete_synapses_find_synapses(dendrites_inh, network_graph, pending_deletions);

    MapSynapseDeletionRequests synapse_deletion_requests_incoming = delete_synapses_exchange_requests(pending_deletions);
    delete_synapses_process_requests(synapse_deletion_requests_incoming, pending_deletions);

    /**
	* Now the list with pending synapse deletions contains all deletion requests
	* of synapses that are connected to at least one of my neurons
	*
	* NOTE:
	* (i)  A synapse can be connected to two of my neurons
	* (ii) A synapse can be connected to one of my neurons and the other neuron belongs to another rank
	*/

    /* Delete all synapses pending for deletion */
    size_t num_synapses_deleted = delete_synapses_commit_deletions(pending_deletions, network_graph);

    GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

    return num_synapses_deleted;
}

void Neurons::delete_synapses_find_synapses(SynapticElements& synaptic_elements, const NetworkGraph& network_graph, std::list<Neurons::PendingSynapseDeletion>& pending_deletions) {
    const auto element_type = synaptic_elements.get_element_type();

    // For my neurons
    for (size_t neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
        /**
		* Create and delete synaptic elements as required.
		* This function only deletes elements (bound and unbound), no synapses.
		*/
        const auto num_synapses_to_delete = synaptic_elements.update_number_elements(neuron_id);
        if (num_synapses_to_delete == 0) {
            continue;
        }

        /**
		* Create a list with all pending synapse deletions.
		* During creating this list, the possibility that neurons want to delete the same
		* synapse is considered.
		*/

        const auto signal_type = synaptic_elements.get_signal_type(neuron_id);
        auto local = delete_synapses_find_synapses_on_neuron(neuron_id, element_type, signal_type, num_synapses_to_delete, network_graph, pending_deletions);
    }
}

std::list<Neurons::PendingSynapseDeletion> Neurons::delete_synapses_find_synapses_on_neuron(size_t neuron_id,
    ElementType element_type,
    SignalType signal_type,
    unsigned int num_synapses_to_delete,
    const NetworkGraph& network_graph,
    std::list<Neurons::PendingSynapseDeletion>& pending_deletions) {

    // Only do something if necessary
    if (0 == num_synapses_to_delete) {
        return {};
    }

    const bool is_axon = element_type == ElementType::AXON;
    const ElementType other_element_type = is_axon ? ElementType::DENDRITE : ElementType::AXON;

    const bool is_exc = signal_type == SignalType::EXCITATORY;
    const SignalType other_signal_type = is_exc ? SignalType::INHIBITORY : SignalType::EXCITATORY;

    std::list<Synapse> list_synapses;

    if (is_axon) {
        // Walk through outgoing edges
        const NetworkGraph::Edges& out_edges = network_graph.get_out_edges(neuron_id);
        list_synapses = delete_synapses_register_edges(out_edges);
    } else {
        // Walk through ingoing edges
        const NetworkGraph::Edges& in_edges = network_graph.get_in_edges(neuron_id, signal_type);
        list_synapses = delete_synapses_register_edges(in_edges);
    }

    RelearnException::check(num_synapses_to_delete <= list_synapses.size(), "num_synapses_to_delete > last_synapses.size()");

    /**
	* Select synapses for deletion
	*/
    std::mt19937& random_number_generator = RandomHolder::get_random_generator(RandomHolderKey::Neurons);

    for (unsigned int num_synapses_selected = 0; num_synapses_selected < num_synapses_to_delete; ++num_synapses_selected) {
        // Randomly select synapse for deletion
        auto synapse_selected = list_synapses.cbegin();
        // Draw random number from [0,1)
        const double random_number = random_number_distribution(random_number_generator);

        // Make iterator point to selected element
        std::advance(synapse_selected, static_cast<int>(list_synapses.size() * random_number));

        RelearnException::check(synapse_selected != list_synapses.cend(), "Didn't select a synapse to delete");

        RankNeuronId src_neuron_id = RankNeuronId(MPIWrapper::get_my_rank(), neuron_id);
        RankNeuronId tgt_neuron_id = synapse_selected->get_rank_neuron_id();
        auto synapse_id = synapse_selected->get_synapse_id();

        if (!is_axon) {
            src_neuron_id = synapse_selected->get_rank_neuron_id();
            tgt_neuron_id = RankNeuronId(MPIWrapper::get_my_rank(), neuron_id);
        }

        // Check if synapse is already in pending deletions, if not, add it.
        auto pending_deletion = std::find_if(pending_deletions.begin(), pending_deletions.end(), [&](auto param) {
            return param.check_light_equality(src_neuron_id, tgt_neuron_id, synapse_id);
        });

        if (pending_deletion == pending_deletions.end()) {
            pending_deletions.emplace_back(src_neuron_id, tgt_neuron_id, synapse_selected->get_rank_neuron_id(),
                other_element_type, signal_type, synapse_selected->get_synapse_id());
        } else {
            pending_deletion->set_affected_element_already_deleted();
        }

        // Remove selected synapse from synapse list
        list_synapses.erase(synapse_selected);
    }

    return pending_deletions;
}

std::list<Neurons::Synapse> Neurons::delete_synapses_register_edges(const NetworkGraph::Edges& edges) {
    std::list<Neurons::Synapse> list_synapses;

    for (const auto& it : edges) {
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

    return list_synapses;
}

Neurons::MapSynapseDeletionRequests Neurons::delete_synapses_exchange_requests(const std::list<PendingSynapseDeletion>& pending_deletions) {
    /**
	* - Go through list with pending synapse deletions and copy those into map "synapse_deletion_requests_outgoing"
	*   where the other neuron affected by the deletion is not one of my neurons
	* - Tell every rank how many deletion requests to receive from me
	* - Prepare for corresponding number of deletion requests from every rank and receive them
	* - Add received deletion requests to the list with pending deletions
	* - Execute pending deletions
	*/

    /**
	* Go through list with pending synapse deletions and copy those into
	* map "synapse_deletion_requests_outgoing" where the other neuron
	* affected by the deletion is not one of my neurons
	*/

    MapSynapseDeletionRequests synapse_deletion_requests_incoming;
    MapSynapseDeletionRequests synapse_deletion_requests_outgoing;
    // All pending deletion requests
    for (const auto& list_it : pending_deletions) {
        const auto target_rank = list_it.get_affected_neuron_id().get_rank();

        // Affected neuron of deletion request resides on different rank.
        // Thus the request needs to be communicated.
        if (target_rank != MPIWrapper::get_my_rank()) {
            synapse_deletion_requests_outgoing[target_rank].append(list_it);
        }
    }

    /**
	* Send to every rank the number of deletion requests it should prepare for from me.
	* Likewise, receive the number of deletion requests that I should prepare for from every rank.
	*/

    std::vector<size_t> num_synapse_deletion_requests_for_ranks(MPIWrapper::get_num_ranks(), 0);
    // Fill vector with my number of synapse deletion requests for every rank
    // Requests to myself are kept local and not sent to myself again.
    for (const auto& it : synapse_deletion_requests_outgoing) {
        auto rank = it.first;
        auto num_requests = it.second.size();

        num_synapse_deletion_requests_for_ranks[rank] = num_requests;
    }

    std::vector<size_t> num_synapse_deletion_requests_from_ranks(MPIWrapper::get_num_ranks(), Constants::uninitialized);
    // Send and receive the number of synapse deletion requests
    MPIWrapper::all_to_all(num_synapse_deletion_requests_for_ranks, num_synapse_deletion_requests_from_ranks, MPIWrapper::Scope::global);
    // Now I know how many requests I will get from every rank.
    // Allocate memory for all incoming synapse deletion requests.
    for (auto rank = 0; rank < MPIWrapper::get_num_ranks(); ++rank) {
        auto num_requests = num_synapse_deletion_requests_from_ranks[rank];
        if (0 != num_requests) {
            synapse_deletion_requests_incoming[rank].resize(num_requests);
        }
    }

    std::vector<MPIWrapper::AsyncToken> mpi_requests(synapse_deletion_requests_outgoing.size() + synapse_deletion_requests_incoming.size());

    /**
	* Send and receive actual synapse deletion requests
	*/

    auto mpi_requests_index = 0;

    // Receive actual synapse deletion requests
    for (auto& it : synapse_deletion_requests_incoming) {
        const auto rank = it.first;
        auto* buffer = it.second.get_requests();
        const auto size_in_bytes = static_cast<int>(it.second.get_requests_size_in_bytes());

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

        ++mpi_requests_index;
    }

    // Send actual synapse deletion requests
    for (const auto& it : synapse_deletion_requests_outgoing) {
        const auto rank = it.first;
        const auto* const buffer = it.second.get_requests();
        const auto size_in_bytes = static_cast<int>(it.second.get_requests_size_in_bytes());

        MPIWrapper::async_send(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

        ++mpi_requests_index;
    }

    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);

    return synapse_deletion_requests_incoming;
}

void Neurons::delete_synapses_process_requests(const MapSynapseDeletionRequests& synapse_deletion_requests_incoming, std::list<PendingSynapseDeletion>& pending_deletions) {
    // From smallest to largest rank that sent deletion request
    for (const auto& it : synapse_deletion_requests_incoming) {
        const SynapseDeletionRequests& requests = it.second;
        const int other_rank = it.first;
        const auto num_requests = requests.size();

        // All requests of a rank
        for (auto request_index = 0; request_index < num_requests; ++request_index) {
            const auto src_neuron_id = requests.get_source_neuron_id(request_index);
            const auto tgt_neuron_id = requests.get_target_neuron_id(request_index);
            const auto affected_neuron_id = requests.get_affected_neuron_id(request_index);
            const auto affected_element_type = requests.get_affected_element_type(request_index);
            const auto signal_type = requests.get_signal_type(request_index);
            const auto synapse_id = requests.get_synapse_id(request_index);

            RankNeuronId src_id(MPIWrapper::get_my_rank(), src_neuron_id);
            RankNeuronId tgt_id(other_rank, tgt_neuron_id);

            if (ElementType::DENDRITE == affected_element_type) {
                src_id = RankNeuronId(other_rank, src_neuron_id);
                tgt_id = RankNeuronId(MPIWrapper::get_my_rank(), tgt_neuron_id);
            }

            auto pending_deletion = std::find_if(pending_deletions.begin(), pending_deletions.end(), [&src_id, &tgt_id, synapse_id](auto param) {
                return param.check_light_equality(src_id, tgt_id, synapse_id);
            });

            if (pending_deletion == pending_deletions.end()) {
                pending_deletions.emplace_back(src_id, tgt_id, RankNeuronId(MPIWrapper::get_my_rank(), affected_neuron_id),
                    affected_element_type, signal_type, synapse_id);
            } else {
                pending_deletion->set_affected_element_already_deleted();
            }

        } // All requests of a rank
    } // All ranks that sent deletion requests
}

size_t Neurons::delete_synapses_commit_deletions(const std::list<PendingSynapseDeletion>& list, NetworkGraph& network_graph) {
    const int my_rank = MPIWrapper::get_my_rank();
    size_t num_synapses_deleted = 0;

    /* Execute pending synapse deletions */
    for (const auto& it : list) {
        // Pending synapse deletion is valid (not completely) if source or
        // target neuron belong to me. To be completely valid, things such as
        // the neuron id need to be validated as well.
        const auto& src_neuron = it.get_src_neuron_id();
        const auto src_neuron_rank = src_neuron.get_rank();
        const auto src_neuron_id = src_neuron.get_neuron_id();

        const auto& tgt_neuron = it.get_tgt_neuron_id();
        const auto tgt_neuron_rank = tgt_neuron.get_rank();
        const auto tgt_neuron_id = tgt_neuron.get_neuron_id();

        RelearnException::check(src_neuron_rank == my_rank || tgt_neuron_rank == my_rank, "Should delete a non-local synapse");

        const auto signal_type = it.get_signal_type();
        const auto element_type = it.get_affected_element_type();

        const auto& affected_neuron = it.get_affected_neuron_id();

        if (src_neuron_rank == my_rank && tgt_neuron_rank == my_rank) {
            /**
			* Count the deleted synapse once for each connected neuron.
			* The reason is that synapses where neurons are on different ranks are also
			* counted once on each rank
			*/
            num_synapses_deleted += 2;
        } else {
            num_synapses_deleted += 1;
        }

        /**
		*  Update network graph
		*/
        int weight_increment = 0;
        if (SignalType::EXCITATORY == signal_type) {
            // DendriteType::EXCITATORY synapses have positive count, so decrement
            weight_increment = -1;
        } else {
            // DendriteType::INHIBITORY synapses have negative count, so increment
            weight_increment = +1;
        }

        network_graph.add_edge_weight(tgt_neuron_id, tgt_neuron_rank, src_neuron_id, src_neuron_rank, weight_increment);

        /**
		* Set element of affected neuron vacant if necessary,
		* i.e., only if the affected neuron belongs to me and the
		* element of the affected neuron still exists.
		*
		* NOTE: Checking that the affected neuron belongs to me is important
		* because the list of pending deletion requests also contains requests whose
		* affected neuron belongs to a different rank.
		*/
        const auto affected_neuron_id = affected_neuron.get_neuron_id();

        if (affected_neuron.get_rank() == my_rank && !it.get_affected_element_already_deleted()) {
            if (ElementType::AXON == element_type) {
                axons.update_conn_cnt(affected_neuron_id, -1);
                continue;
            }

            if (SignalType::EXCITATORY == signal_type) {
                dendrites_exc.update_conn_cnt(affected_neuron_id, -1);
            } else {
                dendrites_inh.update_conn_cnt(affected_neuron_id, -1);
            }
        }
    }

    return num_synapses_deleted;
}

size_t Neurons::create_synapses(Octree& global_tree, NetworkGraph& network_graph) {
    /**
	* 2. Create Synapses
	*
	* - Update region trees (num dendrites in leaves and inner nodes) - postorder traversal (input: cnts, connected_cnts arrays)
	* - Determine target region for every axon
	* - Find target neuron for every axon (input: position, type; output: target neuron_id)
	* - Update synaptic elements (no connection when target neuron's dendrites have already been taken by previous axon)
	* - Update network
	*/

    /**
	* Update global tree bottom-up with current number
	* of vacant dendrites and resulting positions
	*/

    /**********************************************************************************/

    // Lock local RMA memory for local stores
    MPIWrapper::lock_window(MPIWrapper::get_my_rank(), MPI_Locktype::exclusive);

    // Update my local trees bottom-up
    GlobalTimers::timers.start(TimerRegion::UPDATE_LOCAL_TREES);
    global_tree.update_local_trees(dendrites_exc, dendrites_inh, num_neurons);
    GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_LOCAL_TREES);

    /**
	* Exchange branch nodes
	*/
    GlobalTimers::timers.start(TimerRegion::EXCHANGE_BRANCH_NODES);
    OctreeNode* rma_buffer_branch_nodes = MPIWrapper::get_buffer_octree_nodes();
    // Copy local trees' root nodes to correct positions in receive buffer

    const size_t num_local_trees = global_tree.get_num_local_trees();
    for (size_t i = 0; i < num_local_trees; i++) {
        const size_t global_subdomain_id = partition->get_my_subdomain_id_start() + i;
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
    const size_t num_rma_buffer_branch_nodes = MPIWrapper::get_num_buffer_octree_nodes();
    for (size_t i = 0; i < num_rma_buffer_branch_nodes; i++) {
        if (i < partition->get_my_subdomain_id_start() || i > partition->get_my_subdomain_id_end()) {
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
    MPIWrapper::unlock_window(MPIWrapper::get_my_rank());

    /**********************************************************************************/

    // Makes sure that all ranks finished their local access epoch
    // before a remote origin opens an access epoch
    MPIWrapper::barrier(MPIWrapper::Scope::global);

    /**
	* Find target neuron for every vacant axon
	*/
    GlobalTimers::timers.start(TimerRegion::FIND_TARGET_NEURONS);

    const std::vector<double>* dendrites_cnts = nullptr; // TODO(fabian) find a nicer solution
    const std::vector<unsigned int>* dendrites_connected_cnts = nullptr;

    int num_axons_connected_increment = 0;
    MapSynapseCreationRequests synapse_creation_requests_outgoing;

    const std::vector<double>& axons_cnts = axons.get_cnts();
    const std::vector<unsigned int>& axons_connected_cnts = axons.get_connected_cnts();
    const std::vector<SignalType>& axons_signal_types = axons.get_signal_types();

    // For my neurons
    for (size_t neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
        // Number of vacant axons
        const auto num_vacant_axons = static_cast<unsigned int>(axons_cnts[neuron_id]) - axons_connected_cnts[neuron_id];
        RelearnException::check(num_vacant_axons >= 0, "num vacant axons is negative");

        if (num_vacant_axons == 0) {
            continue;
        }

        // DendriteType::EXCITATORY axon
        SignalType dendrite_type_needed = SignalType::EXCITATORY;
        if (SignalType::INHIBITORY == axons_signal_types[neuron_id]) {
            // DendriteType::INHIBITORY axon
            dendrite_type_needed = SignalType::INHIBITORY;
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
            std::optional<RankNeuronId> rank_neuron_id = global_tree.find_target_neuron(neuron_id, axon_xyz_pos, dendrite_type_needed);

            if (rank_neuron_id.has_value()) {
                RankNeuronId val = rank_neuron_id.value();
                /*
				* Append request for synapse creation to rank "target_rank"
				* Note that "target_rank" could also be my own rank.
				*/
                synapse_creation_requests_outgoing[val.get_rank()].append(neuron_id, val.get_neuron_id(), dendrite_type_needed);
            }
        } /* all vacant axons of a neuron */
    } /* my neurons */

    GlobalTimers::timers.stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

    // Make cache empty for next connectivity update
    GlobalTimers::timers.start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree.empty_remote_nodes_cache();
    GlobalTimers::timers.stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    GlobalTimers::timers.start(TimerRegion::CREATE_SYNAPSES);

    size_t num_synapses_created = 0;

    {

        /**
		* At this point "synapse_creation_requests_outgoing" contains
		* all synapse creation requests from this rank
		*
		* The next step is to send the requests to the target ranks and
		* receive the requests from other ranks (including myself)
		*/

        /**
		* Send to every rank the number of requests it should prepare for from me.
		* Likewise, receive the number of requests that I should prepare for from every rank.
		*/
        std::vector<size_t> num_synapse_requests_for_ranks(MPIWrapper::get_num_ranks(), 0);
        // Fill vector with my number of synapse requests for every rank (including me)
        for (const auto& it : synapse_creation_requests_outgoing) {
            auto rank = it.first;
            auto num_requests = (it.second).size();

            num_synapse_requests_for_ranks[rank] = num_requests;
        }

        std::vector<size_t> num_synapse_requests_from_ranks(MPIWrapper::get_num_ranks(), Constants::uninitialized);
        // Send and receive the number of synapse requests
        MPIWrapper::all_to_all(num_synapse_requests_for_ranks, num_synapse_requests_from_ranks, MPIWrapper::Scope::global);

        MapSynapseCreationRequests synapse_creation_requests_incoming;
        // Now I know how many requests I will get from every rank.
        // Allocate memory for all incoming synapse requests.
        for (auto rank = 0; rank < MPIWrapper::get_num_ranks(); rank++) {
            auto num_requests = num_synapse_requests_from_ranks[rank];
            if (0 != num_requests) {
                synapse_creation_requests_incoming[rank].resize(num_requests);
            }
        }

        std::vector<MPIWrapper::AsyncToken>
            mpi_requests(synapse_creation_requests_outgoing.size() + synapse_creation_requests_incoming.size());

        /**
		* Send and receive actual synapse requests
		*/
        auto mpi_requests_index = 0;

        // Receive actual synapse requests
        for (auto& it : synapse_creation_requests_incoming) {
            const auto rank = it.first;
            auto* buffer = it.second.get_requests();
            const auto size_in_bytes = static_cast<int>(it.second.get_requests_size_in_bytes());

            MPIWrapper::async_receive(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

            mpi_requests_index++;
        }
        // Send actual synapse requests
        for (const auto& it : synapse_creation_requests_outgoing) {
            const auto rank = it.first;
            const auto* const buffer = it.second.get_requests();
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
        for (auto& it : synapse_creation_requests_incoming) {
            const auto source_rank = it.first;
            SynapseCreationRequests& requests = it.second;
            const auto num_requests = requests.size();

            // All requests of a rank
            for (auto request_index = 0; request_index < num_requests; request_index++) {
                size_t source_neuron_id{ Constants::uninitialized };
                size_t target_neuron_id{ Constants::uninitialized };
                size_t dendrite_type_needed{ Constants::uninitialized };
                std::tie(source_neuron_id, target_neuron_id, dendrite_type_needed) = requests.get_request(request_index);

                // Sanity check: if the request received is targeted for me
                if (target_neuron_id >= num_neurons) {
                    RelearnException::fail("Target_neuron_id exceeds my neurons");
                    exit(EXIT_FAILURE);
                }
                // DendriteType::INHIBITORY dendrite requested
                if (1 == dendrite_type_needed) {
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
                RelearnException::check((*dendrites_cnts)[target_neuron_id] - (*dendrites_connected_cnts)[target_neuron_id] >= 0, "Connectivity went downside");

                const auto diff = static_cast<unsigned int>((*dendrites_cnts)[target_neuron_id] - (*dendrites_connected_cnts)[target_neuron_id]);
                if (diff != 0) {
                    // Increment num of connected dendrites
                    if (1 == dendrite_type_needed) {
                        dendrites_inh.update_conn_cnt(target_neuron_id, 1);
                    } else {
                        dendrites_exc.update_conn_cnt(target_neuron_id, 1);
                    }

                    // Update network
                    network_graph.add_edge_weight(target_neuron_id, MPIWrapper::get_my_rank(), source_neuron_id, source_rank, num_axons_connected_increment);

                    // Set response to "connected" (success)
                    requests.set_response(request_index, 1);
                    num_synapses_created++;
                } else {
                    // Other axons were faster and came first
                    // Set response to "not connected" (not success)
                    requests.set_response(request_index, 0);
                }
            } // All requests of a rank
        } // Increasing order of ranks that sent requests

        /**
		  * Send and receive responses for synapse requests
		  */
        mpi_requests_index = 0;

        // Receive responses
        for (auto& it : synapse_creation_requests_outgoing) {
            const auto rank = it.first;
            auto* buffer = it.second.get_responses();
            const auto size_in_bytes = static_cast<int>(it.second.get_responses_size_in_bytes());

            MPIWrapper::async_receive(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

            mpi_requests_index++;
        }
        // Send responses
        for (const auto& it : synapse_creation_requests_incoming) {
            const auto rank = it.first;
            const auto* const buffer = it.second.get_responses();
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
        for (const auto& it : synapse_creation_requests_outgoing) {
            const auto target_rank = it.first;
            const SynapseCreationRequests& requests = it.second;
            const auto num_requests = requests.size();

            // All responses from a rank
            for (auto request_index = 0; request_index < num_requests; request_index++) {
                char connected = requests.get_response(request_index);
                size_t source_neuron_id{ Constants::uninitialized };
                size_t target_neuron_id{ Constants::uninitialized };
                size_t dendrite_type_needed{ Constants::uninitialized };
                std::tie(source_neuron_id, target_neuron_id, dendrite_type_needed) = requests.get_request(request_index);

                // Request to form synapse succeeded
                if (connected != 0) {
                    // Increment num of connected axons
                    axons.update_conn_cnt(source_neuron_id, 1);
                    //axons_connected_cnts[source_neuron_id]++;
                    num_synapses_created++;

                    const double delta = axons.get_cnt(source_neuron_id) - axons.get_connected_cnt(source_neuron_id);
                    RelearnException::check(delta >= 0, "%f", delta);

                    // I have already created the synapse in the network
                    // if the response comes from myself
                    if (target_rank != MPIWrapper::get_my_rank()) {
                        // Update network
                        num_axons_connected_increment = (1 == dendrite_type_needed) ? -1 : +1;
                        network_graph.add_edge_weight(target_neuron_id, target_rank, source_neuron_id, MPIWrapper::get_my_rank(), num_axons_connected_increment);
                    }
                } else {
                    // Other axons were faster and came first
                }
            } // All responses from a rank
        } // All outgoing requests
    }

    GlobalTimers::timers.stop_and_add(TimerRegion::CREATE_SYNAPSES);

    return num_synapses_created;
}

void Neurons::debug_check_counts(const NetworkGraph& network_graph) {
    if (!Config::do_debug_checks) {
        return;
    }

    const std::vector<double>& axs_count = axons.get_cnts();
    const std::vector<unsigned int>& axs_conn_count = axons.get_connected_cnts();
    const std::vector<double>& de_count = dendrites_exc.get_cnts();
    const std::vector<unsigned int>& de_conn_count = dendrites_exc.get_connected_cnts();
    const std::vector<double>& di_count = dendrites_inh.get_cnts();
    const std::vector<unsigned int>& di_conn_count = dendrites_inh.get_connected_cnts();

    for (size_t i = 0; i < num_neurons; i++) {
        const double diff_axs = axs_count[i] - axs_conn_count[i];
        const double diff_de = de_count[i] - de_conn_count[i];
        const double diff_di = di_count[i] - di_conn_count[i];

        RelearnException::check(diff_axs >= 0.0, "%f", diff_axs);
        RelearnException::check(diff_de >= 0.0, "%f", diff_de);
        RelearnException::check(diff_di >= 0.0, "%f", diff_di);
    }

    for (size_t i = 0; i < num_neurons; i++) {
        const double connected_axons = axs_conn_count[i];
        const double connected_dend_exc = de_conn_count[i];
        const double connected_dend_inh = di_conn_count[i];

        const size_t num_conn_axons = static_cast<size_t>(connected_axons);
        const size_t num_conn_dend_ex = static_cast<size_t>(connected_dend_exc);
        const size_t num_conn_dend_in = static_cast<size_t>(connected_dend_inh);

        const size_t num_out_ng = network_graph.get_num_out_edges(i);
        const size_t num_in_exc_ng = network_graph.get_num_in_edges_ex(i);
        const size_t num_in_inh_ng = network_graph.get_num_in_edges_in(i);

        RelearnException::check(num_conn_axons == num_out_ng, "In Neurons conn axons, %u vs. %u", num_conn_axons, num_out_ng);
        RelearnException::check(num_conn_dend_ex == num_in_exc_ng, "In Neurons conn dend ex, %u vs. %u", num_conn_dend_ex, num_in_exc_ng);
        RelearnException::check(num_conn_dend_in == num_in_inh_ng, "In Neurons conn dend in, %u vs. %u", num_conn_dend_in, num_in_inh_ng);
    }
}

void Neurons::print_sums_of_synapses_and_elements_to_log_file_on_rank_0(size_t step, size_t sum_synapses_deleted, size_t sum_synapses_created) {
    unsigned int sum_axons_exc_cnts = 0;
    unsigned int sum_axons_exc_connected_cnts = 0;
    unsigned int sum_axons_inh_cnts = 0;
    unsigned int sum_axons_inh_connected_cnts = 0;
    unsigned int sum_dends_exc_cnts = 0;
    unsigned int sum_dends_exc_connected_cnts = 0;
    unsigned int sum_dends_inh_cnts = 0;
    unsigned int sum_dends_inh_connected_cnts = 0;
    unsigned int sum_axons_exc_vacant = 0;
    unsigned int sum_axons_inh_vacant = 0;
    unsigned int sum_dends_exc_vacant = 0;
    unsigned int sum_dends_inh_vacant = 0;

    // My vacant axons (exc./inh.)
    sum_axons_exc_cnts = sum_axons_exc_connected_cnts = 0;
    sum_axons_inh_cnts = sum_axons_inh_connected_cnts = 0;

    const std::vector<double>& cnts_ax = axons.get_cnts();
    const std::vector<unsigned int>& connected_cnts_ax = axons.get_connected_cnts();
    const std::vector<SignalType>& signal_types = axons.get_signal_types();

    for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
        if (SignalType::EXCITATORY == signal_types[neuron_id]) {
            sum_axons_exc_cnts += static_cast<unsigned int>(cnts_ax[neuron_id]);
            sum_axons_exc_connected_cnts += static_cast<unsigned int>(connected_cnts_ax[neuron_id]);
        } else {
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
    const std::vector<unsigned int>& connected_cnts_den_ex = dendrites_exc.get_connected_cnts();
    for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
        sum_dends_exc_cnts += static_cast<unsigned int>(cnts_den_ex[neuron_id]);
        sum_dends_exc_connected_cnts += static_cast<unsigned int>(connected_cnts_den_ex[neuron_id]);
    }
    sum_dends_exc_vacant = sum_dends_exc_cnts - sum_dends_exc_connected_cnts;

    // Inh.
    sum_dends_inh_cnts = sum_dends_inh_connected_cnts = 0;
    const std::vector<double>& cnts_den_in = dendrites_inh.get_cnts();
    const std::vector<unsigned int>& connected_cnts_den_in = dendrites_inh.get_connected_cnts();
    for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
        sum_dends_inh_cnts += static_cast<unsigned int>(cnts_den_in[neuron_id]);
        sum_dends_inh_connected_cnts += static_cast<unsigned int>(connected_cnts_den_in[neuron_id]);
    }
    sum_dends_inh_vacant = sum_dends_inh_cnts - sum_dends_inh_connected_cnts;

    // Get global sums at rank 0
    std::array<unsigned int, Constants::num_items_per_request> sums_local = { sum_axons_exc_vacant,
        sum_axons_inh_vacant,
        sum_dends_exc_vacant,
        sum_dends_inh_vacant,
        static_cast<unsigned int>(sum_synapses_deleted),
        static_cast<unsigned int>(sum_synapses_created) };

    std::array<unsigned int, Constants::num_items_per_request> sums_global{ 0, 0, 0, 0, 0, 0 }; // Init all to zero

    MPIWrapper::reduce(sums_local, sums_global, MPIWrapper::ReduceFunction::sum, 0, MPIWrapper::Scope::global);

    // Output data
    if (0 == MPIWrapper::get_my_rank()) {
        std::stringstream ss;
        const int cwidth = 20; // Column width

        // Write headers to file if not already done so
        if (0 == step) {
            ss << "# SUMS OVER ALL NEURONS\n";
            ss << std::left
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
        ss << std::left
           << std::setw(cwidth) << step
           << std::setw(cwidth) << sums_global[0]
           << std::setw(cwidth) << sums_global[1]
           << std::setw(cwidth) << sums_global[2]
           << std::setw(cwidth) << sums_global[3]
           << std::setw(cwidth) << sums_global[4] / 2 // As counted on both of the neurons
           << std::setw(cwidth) << sums_global[5] / 2 // As counted on both of the neurons
           << "\n";

        LogFiles::write_to_file(LogFiles::EventType::Sums, ss.str(), false);
    }
}

void Neurons::print_neurons_overview_to_log_file_on_rank_0(size_t step) {
    const StatisticalMeasures<double> calcium_statistics = global_statistics(calcium, num_neurons, partition->get_total_num_neurons(), 0, MPIWrapper::Scope::global);

    const StatisticalMeasures<double> activity_statistics = global_statistics(neuron_model->get_x(), num_neurons, partition->get_total_num_neurons(), 0, MPIWrapper::Scope::global);

    // Output data
    if (0 == MPIWrapper::get_my_rank()) {
        std::stringstream ss;
        const int cwidth = 16; // Column width

        // Write headers to file if not already done so
        if (0 == step) {
            ss << "# ALL NEURONS\n";
            ss << std::left
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
        ss << std::left
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

        LogFiles::write_to_file(LogFiles::EventType::NeuronsOverview, ss.str(), false);
    }
}

void Neurons::print_network_graph_to_log_file(const NetworkGraph& network_graph, const NeuronIdMap& neuron_id_map) {
    std::stringstream ss;

    // Write output format to file
    ss << "# " << partition->get_total_num_neurons() << "\n"; // Total number of neurons
    ss << "# <target neuron id> <source neuron id> <weight>"
       << "\n";

    // Write network graph to file
    network_graph.print(ss, neuron_id_map);

    LogFiles::write_to_file(LogFiles::EventType::Network, ss.str(), false);
}

void Neurons::print_positions_to_log_file(const NeuronIdMap& neuron_id_map) {
    std::stringstream ss;

    // Write total number of neurons to log file
    ss << "# " << partition->get_total_num_neurons() << "\n";
    ss << "# "
       << "<global id> <pos x> <pos y> <pos z> <area> <type>"
       << "\n";

    const std::vector<double>& axons_x_dims = positions.get_x_dims();
    const std::vector<double>& axons_y_dims = positions.get_y_dims();
    const std::vector<double>& axons_z_dims = positions.get_z_dims();

    const std::vector<SignalType>& signal_types = axons.get_signal_types();

    // Print global ids, positions, and areas of local neurons
    bool ret = false;
    size_t glob_id = 0;

    const int my_rank = MPIWrapper::get_my_rank();
    ss << std::fixed << std::setprecision(6);

    for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
        RankNeuronId rank_neuron_id{ my_rank, neuron_id };
        std::tie(ret, glob_id) = neuron_id_map.rank_neuron_id2glob_id(rank_neuron_id);
        RelearnException::check(ret, "ret is false");

        const char* const signal_type_name = signal_types[neuron_id] == SignalType::EXCITATORY ? "ex" : "in";

        glob_id++;

        ss << glob_id << " "
           << axons_x_dims[neuron_id] << " "
           << axons_y_dims[neuron_id] << " "
           << axons_z_dims[neuron_id] << " "
           << area_names[neuron_id] << " "
           << signal_type_name << "\n";
    }

    ss << std::flush;
    ss << std::defaultfloat;

    LogFiles::write_to_file(LogFiles::EventType::Positions, ss.str(), false);
}

void Neurons::print() {
    // Column widths
    const int cwidth_left = 6;
    const int cwidth = 16;

    std::stringstream ss;

    // Heading
    ss << std::left << std::setw(cwidth_left) << "gid" << std::setw(cwidth) << "x" << std::setw(cwidth) << "AP";
    ss << std::setw(cwidth) << "refrac" << std::setw(cwidth) << "C" << std::setw(cwidth) << "A" << std::setw(cwidth) << "D_ex" << std::setw(cwidth) << "D_in"
       << "\n";

    // Values
    for (size_t i = 0; i < num_neurons; i++) {
        ss << std::left << std::setw(cwidth_left) << i << std::setw(cwidth) << neuron_model->get_x(i) << std::setw(cwidth) << neuron_model->get_fired(i);
        ss << std::setw(cwidth) << neuron_model->get_secondary_variable(i) << std::setw(cwidth) << calcium[i] << std::setw(cwidth) << axons.get_cnt(i);
        ss << std::setw(cwidth) << dendrites_exc.get_cnt(i) << std::setw(cwidth) << dendrites_inh.get_cnt(i) << "\n";
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, ss.str(), true);
}

void Neurons::print_info_for_barnes_hut() {
    const std::vector<double>& x_dims = positions.get_x_dims();
    const std::vector<double>& y_dims = positions.get_y_dims();
    const std::vector<double>& z_dims = positions.get_z_dims();

    const std::vector<double>& axons_cnts = axons.get_cnts();
    const std::vector<double>& dendrites_exc_cnts = dendrites_exc.get_cnts();
    const std::vector<double>& dendrites_inh_cnts = dendrites_inh.get_cnts();

    const std::vector<unsigned int>& axons_connected_cnts = axons.get_connected_cnts();
    const std::vector<unsigned int>& dendrites_exc_connected_cnts = dendrites_exc.get_connected_cnts();
    const std::vector<unsigned int>& dendrites_inh_connected_cnts = dendrites_inh.get_connected_cnts();

    // Column widths
    const int cwidth_small = 8;
    const int cwidth_medium = 16;
    const int cwidth_big = 27;

    std::stringstream ss;
    std::string my_string;

    // Heading
    ss << std::left << std::setw(cwidth_small) << "gid" << std::setw(cwidth_small) << "region" << std::setw(cwidth_medium) << "position";
    ss << std::setw(cwidth_big) << "axon (exist|connected)" << std::setw(cwidth_big) << "exc_den (exist|connected)";
    ss << std::setw(cwidth_big) << "inh_den (exist|connected)"
       << "\n";

    // Values
    for (size_t i = 0; i < num_neurons; i++) {
        ss << std::left << std::setw(cwidth_small) << i;

        const auto x = static_cast<unsigned int>(x_dims[i]);
        const auto y = static_cast<unsigned int>(y_dims[i]);
        const auto z = static_cast<unsigned int>(z_dims[i]);

        my_string = "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")";
        ss << std::setw(cwidth_medium) << my_string;

        my_string = std::to_string(axons_cnts[i]) + "|" + std::to_string(axons_connected_cnts[i]);
        ss << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_exc_cnts[i]) + "|" + std::to_string(dendrites_exc_connected_cnts[i]);
        ss << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_inh_cnts[i]) + "|" + std::to_string(dendrites_inh_connected_cnts[i]);
        ss << std::setw(cwidth_big) << my_string;

        ss << "\n";
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, ss.str(), true);
}

void Neurons::print_pending_synapse_deletions(const std::list<PendingSynapseDeletion>& list) {
    std::stringstream ss;

    for (const auto& it : list) {
        size_t affected_element_type_converted = it.get_affected_element_type() == ElementType::AXON ? 0 : 1;
        size_t signal_type_converted = it.get_signal_type() == SignalType::EXCITATORY ? 0 : 1;

        ss << "src_neuron_id: " << it.get_src_neuron_id() << "\n";
        ss << "tgt_neuron_id: " << it.get_tgt_neuron_id() << "\n";
        ss << "affected_neuron_id: " << it.get_affected_neuron_id() << "\n";
        ss << "affected_element_type: " << affected_element_type_converted << "\n";
        ss << "signal_type: " << signal_type_converted << "\n";
        ss << "synapse_id: " << it.get_synapse_id() << "\n";
        ss << "affected_element_already_deleted: " << it.get_affected_element_already_deleted() << "\n"
           << "\n";
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, ss.str(), true);
}
