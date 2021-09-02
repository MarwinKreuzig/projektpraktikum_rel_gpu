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

#include "../io/LogFiles.h"
#include "../mpi/MPIWrapper.h"
#include "../structure/Octree.h"
#include "../structure/Partition.h"
#include "../util/Random.h"
#include "../util/Timers.h"
#include "NetworkGraph.h"
#include "helper/RankNeuronId.h"
#include "models/NeuronModels.h"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <optional>
#include <sstream>

void Neurons::init(size_t number_neurons) {
    num_neurons = number_neurons;

    neuron_model->init(number_neurons);
    extra_info->init(number_neurons);

    axons->init(number_neurons);
    dendrites_exc->init(number_neurons);
    dendrites_inh->init(number_neurons);

    /**
	* Mark dendrites as exc./inh.
	*/
    for (size_t i = 0; i < number_neurons; i++) {
        dendrites_exc->set_signal_type(i, SignalType::EXCITATORY);
        dendrites_inh->set_signal_type(i, SignalType::INHIBITORY);
    }

    disable_flags.resize(number_neurons, 1);
    calcium.resize(number_neurons);

    // Init member variables
    for (size_t i = 0; i < number_neurons; i++) {
        // Set calcium concentration
        const auto fired = neuron_model->get_fired(i);
        calcium[i] = fired ? neuron_model->get_beta() : 0.0;
    }
}

void Neurons::init_synaptic_elements() {
    // Give unbound synaptic elements as well
    const double num_axons_offset = 0;
    const double num_dends_offset = 0;

    const std::vector<double>& axons_cnts = axons->get_cnts();
    const std::vector<double>& dendrites_inh_cnts = dendrites_inh->get_cnts();
    const std::vector<double>& dendrites_exc_cnts = dendrites_exc->get_cnts();

    for (auto i = 0; i < num_neurons; i++) {
        const size_t axon_connections = network_graph->get_num_out_edges(i);
        const size_t dendrites_ex_connections = network_graph->get_num_in_edges_ex(i);
        const size_t dendrites_in_connections = network_graph->get_num_in_edges_in(i);

        axons->update_cnt(i, static_cast<double>(axon_connections));
        dendrites_exc->update_cnt(i, static_cast<double>(dendrites_ex_connections));
        dendrites_inh->update_cnt(i, static_cast<double>(dendrites_in_connections));

        axons->update_conn_cnt(i, static_cast<int>(axon_connections));
        dendrites_exc->update_conn_cnt(i, static_cast<int>(dendrites_ex_connections));
        dendrites_inh->update_conn_cnt(i, static_cast<int>(dendrites_in_connections));

        RelearnException::check(axons_cnts[i] >= axons->get_connected_cnts()[i], "Error is with: %d", i);
        RelearnException::check(dendrites_inh_cnts[i] >= dendrites_inh->get_connected_cnts()[i], "Error is with: %d", i);
        RelearnException::check(dendrites_exc_cnts[i] >= dendrites_exc->get_connected_cnts()[i], "Error is with: %d", i);
    }
}

/**
 * Disables all neurons with specified ids
 */
void Neurons::disable_neurons(const std::vector<size_t>& neuron_ids) {
    neuron_model->disable_neurons(neuron_ids);

    const auto my_rank = MPIWrapper::get_my_rank();

    std::vector<int> deleted_axons_connections(num_neurons, 0);
    std::vector<int> deleted_dend_ex_connections(num_neurons, 0);
    std::vector<int> deleted_dend_in_connections(num_neurons, 0);

    for (const auto neuron_id : neuron_ids) {
        RelearnException::check(neuron_id < num_neurons, "In Neurons::disable_neurons, there was a too large id: %ull vs %ull", neuron_id, num_neurons);
        disable_flags[neuron_id] = 0;

        const auto in_edges = network_graph->get_in_edges(neuron_id); // Intended copy

        for (const auto& [edge_key, weight] : in_edges) {
            const auto& [rank, source_id] = edge_key;
            RelearnException::check(rank == my_rank, "Currently, disabling neurons is only supported without mpi");
            network_graph->add_edge_weight(neuron_id, my_rank, source_id, rank, -weight);

            deleted_axons_connections[source_id] += weight;
        }

        const auto out_edges = network_graph->get_out_edges(neuron_id); // Intended copy

        for (const auto& [edge_key, weight] : out_edges) {
            const auto& [rank, target_id] = edge_key;
            RelearnException::check(rank == my_rank, "Currently, disabling neurons is only supported without mpi");
            network_graph->add_edge_weight(target_id, rank, neuron_id, my_rank, -weight);

            if (weight > 0) {
                deleted_dend_ex_connections[target_id] += weight;
            } else {
                deleted_dend_in_connections[target_id] -= weight;
            }
        }
    }

    axons->update_after_deletion(deleted_axons_connections, neuron_ids);
    dendrites_exc->update_after_deletion(deleted_dend_ex_connections, neuron_ids);
    dendrites_inh->update_after_deletion(deleted_dend_in_connections, neuron_ids);
}

void Neurons::enable_neurons(const std::vector<size_t>& neuron_ids) {
    for (const auto neuron_id : neuron_ids) {
        RelearnException::check(neuron_id < num_neurons, "In Neurons::enable_neurons, there was a too large id: %ull vs %ull", neuron_id, num_neurons);
        disable_flags[neuron_id] = 1;
    }
}

void Neurons::create_neurons(size_t creation_count) {
    const auto current_size = num_neurons;
    const auto new_size = current_size + creation_count;

    neuron_model->create_neurons(creation_count);
    extra_info->create_neurons(creation_count);

    network_graph->create_neurons(creation_count);

    axons->create_neurons(creation_count);
    dendrites_exc->create_neurons(creation_count);
    dendrites_inh->create_neurons(creation_count);

    for (size_t i = current_size; i < new_size; i++) {
        dendrites_exc->set_signal_type(i, SignalType::EXCITATORY);
        dendrites_inh->set_signal_type(i, SignalType::INHIBITORY);
    }

    disable_flags.resize(new_size, 1);
    calcium.resize(new_size);

    for (size_t i = current_size; i < new_size; i++) {
        // Set calcium concentration
        const auto fired = neuron_model->get_fired(i);
        calcium[i] = fired ? neuron_model->get_beta() : 0.0;
    }

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& x_dims = extra_info->get_x_dims();
    const auto& y_dims = extra_info->get_y_dims();
    const auto& z_dims = extra_info->get_z_dims();

    for (size_t i = current_size; i < new_size; i++) {
        const auto* const node = global_tree->insert({ x_dims[i], y_dims[i], z_dims[i] }, i, my_rank);
        RelearnException::check(node != nullptr, "node is nullptr");
    }

    num_neurons = new_size;
}

void Neurons::update_electrical_activity() {
    neuron_model->update_electrical_activity(*network_graph, disable_flags);
    update_calcium();
}

void Neurons::update_calcium() {
    GlobalTimers::timers.start(TimerRegion::CALC_ACTIVITY);

    const auto h = neuron_model->get_h();
    const auto tau_C = neuron_model->get_tau_C();
    const auto beta = neuron_model->get_beta();
    const auto& fired = neuron_model->get_fired();

    // The following line is commented as compilers cannot make up their mind whether they want to have the constants shared or not
    //#pragma omp parallel for shared(fired, h, tau_C, beta) default(none)
    // NOLINTNEXTLINE
#pragma omp parallel for
    for (auto neuron_id = 0; neuron_id < calcium.size(); ++neuron_id) {
        if (disable_flags[neuron_id] == 0) {
            continue;
        }

        for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
            // Update calcium depending on the firing
            calcium[neuron_id] += (1 / static_cast<double>(h)) * (-calcium[neuron_id] / tau_C + beta * static_cast<double>(fired[neuron_id]));
        }
    }

    GlobalTimers::timers.stop_and_add(TimerRegion::CALC_ACTIVITY);
}

Neurons::StatisticalMeasures Neurons::global_statistics(const std::vector<double>& local_values, [[maybe_unused]] size_t num_local_values, size_t total_num_values, int root) const {
    const auto scope = MPIWrapper::Scope::global;

    const double my_avg = std::accumulate(local_values.begin(), local_values.end(), 0.0)
        / total_num_values;

    // Get global min and max at rank "root"
    const auto [d_my_min, d_my_max] = [&]() -> std::tuple<double, double> {
        const auto result = std::minmax_element(local_values.begin(), local_values.end());
        return { static_cast<double>(*result.first), static_cast<double>(*result.second) };
    }();

    const double d_min = MPIWrapper::reduce(d_my_min, MPIWrapper::ReduceFunction::min, root, scope);
    const double d_max = MPIWrapper::reduce(d_my_max, MPIWrapper::ReduceFunction::max, root, scope);

    // Get global avg at all ranks (needed for variance)
    const double avg = MPIWrapper::all_reduce(my_avg, MPIWrapper::ReduceFunction::sum, scope);

    /**
	* Calc variance
	*/
    double my_var = 0;
    for (size_t neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
        my_var += (local_values[neuron_id] - avg) * (local_values[neuron_id] - avg);
    }
    my_var /= total_num_values;

    // Get global variance at rank "root"
    const double var = MPIWrapper::reduce(my_var, MPIWrapper::ReduceFunction::sum, root, scope);

    // Calc standard deviation
    const double std = sqrt(var);

    return { d_min, d_max, avg, var, std };
}

size_t Neurons::delete_synapses() {
    /**
	* 1. Update number of synaptic elements and delete synapses if necessary
	*/

    GlobalTimers::timers.start(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

    /**
	* Create list with synapses to delete (pending synapse deletions)
	*/
    // Dendrite exc cannot delete a synapse that is connected to a dendrite inh. pending_deletions is used as an empty dummy
    const auto to_delete_axons = axons->commit_updates(disable_flags);
    const auto to_delete_dendrites_excitatory = dendrites_exc->commit_updates(disable_flags);
    const auto to_delete_dendrites_inhibitory = dendrites_inh->commit_updates(disable_flags);

    auto deletions_axons = delete_synapses_find_synapses(*axons, to_delete_axons, {});
    const auto deletions_dendrites_excitatory = delete_synapses_find_synapses(*dendrites_exc, to_delete_dendrites_excitatory, deletions_axons.first);
    const auto deletions_dendrites_inhibitory = delete_synapses_find_synapses(*dendrites_inh, to_delete_dendrites_inhibitory, deletions_axons.first);

    for (const auto index : deletions_dendrites_excitatory.second) {
        deletions_axons.first[index].set_affected_element_already_deleted();
    }

    for (const auto index : deletions_dendrites_inhibitory.second) {
        deletions_axons.first[index].set_affected_element_already_deleted();
    }

    PendingDeletionsV pending_deletions;
    pending_deletions.insert(pending_deletions.cend(), deletions_axons.first.begin(), deletions_axons.first.end());
    pending_deletions.insert(pending_deletions.cend(), deletions_dendrites_excitatory.first.begin(), deletions_dendrites_excitatory.first.end());
    pending_deletions.insert(pending_deletions.cend(), deletions_dendrites_inhibitory.first.begin(), deletions_dendrites_inhibitory.first.end());

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
    const auto num_synapses_deleted = delete_synapses_commit_deletions(pending_deletions);
    GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

    return num_synapses_deleted;
}

std::pair<Neurons::PendingDeletionsV, std::vector<size_t>> Neurons::delete_synapses_find_synapses(const SynapticElements& synaptic_elements, const std::pair<unsigned int, std::vector<unsigned int>>& to_delete,
    const PendingDeletionsV& other_pending_deletions) {

    const auto& sum_to_delete = to_delete.first;
    const auto& number_deletions = to_delete.second;

    PendingDeletionsV pending_deletions;
    pending_deletions.reserve(sum_to_delete);

    if (sum_to_delete == 0) {
        return std::make_pair(pending_deletions, std::vector<size_t>{});
    }

    const auto element_type = synaptic_elements.get_element_type();

    std::vector<size_t> total_vector_affected_indices;

    for (size_t neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == 0) {
            continue;
        }

        /**
		* Create and delete synaptic elements as required.
		* This function only deletes elements (bound and unbound), no synapses.
		*/
        const auto num_synapses_to_delete = number_deletions[neuron_id];
        if (num_synapses_to_delete == 0) {
            continue;
        }

        const auto signal_type = synaptic_elements.get_signal_type(neuron_id);
        const auto affected_indices = delete_synapses_find_synapses_on_neuron(neuron_id, element_type, signal_type, num_synapses_to_delete, pending_deletions, other_pending_deletions);

        total_vector_affected_indices.insert(total_vector_affected_indices.cend(), affected_indices.begin(), affected_indices.end());
    }

    return std::make_pair(pending_deletions, total_vector_affected_indices);
}

std::vector<size_t> Neurons::delete_synapses_find_synapses_on_neuron(size_t neuron_id,
    ElementType element_type,
    SignalType signal_type,
    unsigned int num_synapses_to_delete,
    PendingDeletionsV& pending_deletions,
    const PendingDeletionsV& other_pending_deletions) {

    // Only do something if necessary
    if (0 == num_synapses_to_delete) {
        return {};
    }

    const bool is_axon = element_type == ElementType::AXON;
    const ElementType other_element_type = is_axon ? ElementType::DENDRITE : ElementType::AXON;

    std::vector<Synapse> current_synapses;

    if (is_axon) {
        // Walk through outgoing edges
        const NetworkGraph::Edges& out_edges = network_graph->get_out_edges(neuron_id);
        current_synapses = delete_synapses_register_edges(out_edges);
    } else {
        // Walk through ingoing edges
        const NetworkGraph::Edges& in_edges = network_graph->get_in_edges(neuron_id, signal_type);
        current_synapses = delete_synapses_register_edges(in_edges);
    }

    RelearnException::check(num_synapses_to_delete <= current_synapses.size(), "num_synapses_to_delete > last_synapses.size()");

    /**
	* Select synapses for deletion
	*/
    std::vector<size_t> already_removed_indices;

    for (unsigned int num_synapses_selected = 0; num_synapses_selected < num_synapses_to_delete; ++num_synapses_selected) {
        // Randomly select synapse for deletion
        auto synapse_selected = current_synapses.cbegin();
        // Draw random number from [0,1)
        const double random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::Neurons, 0.0, 1.0);

        // Make iterator point to selected element
        std::advance(synapse_selected, static_cast<int>(current_synapses.size() * random_number));

        RelearnException::check(synapse_selected != current_synapses.cend(), "Didn't select a synapse to delete");

        RankNeuronId src_neuron_id = RankNeuronId(MPIWrapper::get_my_rank(), neuron_id);
        RankNeuronId tgt_neuron_id = synapse_selected->get_rank_neuron_id();
        auto synapse_id = synapse_selected->get_synapse_id();

        if (!is_axon) {
            src_neuron_id = synapse_selected->get_rank_neuron_id();
            tgt_neuron_id = RankNeuronId(MPIWrapper::get_my_rank(), neuron_id);
        }

        // Check if synapse is already in pending deletions, if not, add it.
        const auto pending_deletion = std::find_if(other_pending_deletions.begin(), other_pending_deletions.end(), [&](auto param) {
            return param.check_light_equality(src_neuron_id, tgt_neuron_id, synapse_id);
        });

        if (pending_deletion == other_pending_deletions.end()) {
            pending_deletions.emplace_back(src_neuron_id, tgt_neuron_id, synapse_selected->get_rank_neuron_id(),
                other_element_type, signal_type, synapse_selected->get_synapse_id());
        } else {
            const auto distance = std::distance(other_pending_deletions.begin(), pending_deletion);
            already_removed_indices.push_back(distance);
        }

        // Remove selected synapse from synapse list
        current_synapses.erase(synapse_selected);
    }

    return already_removed_indices;
}

std::vector<Neurons::Synapse> Neurons::delete_synapses_register_edges(const std::vector<std::pair<std::pair<int, size_t>, int>>& edges) {
    std::vector<Neurons::Synapse> current_synapses;

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
            current_synapses.emplace_back(rank_neuron_id, synapse_id);
        }
    }

    return current_synapses;
}

Neurons::MapSynapseDeletionRequests Neurons::delete_synapses_exchange_requests(const PendingDeletionsV& pending_deletions) {
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
        if (auto num_requests = num_synapse_deletion_requests_from_ranks[rank]; 0 != num_requests) {
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

void Neurons::delete_synapses_process_requests(const MapSynapseDeletionRequests& synapse_deletion_requests_incoming, PendingDeletionsV& pending_deletions) {
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

size_t Neurons::delete_synapses_commit_deletions(const PendingDeletionsV& list) {
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

        network_graph->add_edge_weight(tgt_neuron_id, tgt_neuron_rank, src_neuron_id, src_neuron_rank, weight_increment);

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
                axons->update_conn_cnt(affected_neuron_id, -1);
                continue;
            }

            if (SignalType::EXCITATORY == signal_type) {
                dendrites_exc->update_conn_cnt(affected_neuron_id, -1);
            } else {
                dendrites_inh->update_conn_cnt(affected_neuron_id, -1);
            }
        }
    }

    return num_synapses_deleted;
}

size_t Neurons::create_synapses() {
    /**
	* 2. Create Synapses
	*
	* - Update region trees (num dendrites in leaves and inner nodes) - postorder traversal (input: cnts, connected_cnts arrays)
	* - Determine target region for every axon
	* - Find target neuron for every axon (input: position, type; output: target neuron_id)
	* - Update synaptic elements (no connection when target neuron's dendrites have already been taken by previous axon)
	* - Update network
	*/

    create_synapses_update_octree();
    MapSynapseCreationRequests synapse_creation_requests_outgoing = create_synapses_find_targets();

    GlobalTimers::timers.start(TimerRegion::CREATE_SYNAPSES);

    MapSynapseCreationRequests synapse_creation_requests_incoming = create_synapses_exchange_requests(synapse_creation_requests_outgoing);
    const auto& local_results = create_synapses_process_requests(synapse_creation_requests_incoming);
    const auto synapses_created_locally = local_results.first;

    const auto& received_responses = create_synapses_exchange_responses(local_results.second, synapse_creation_requests_outgoing);
    const auto synapses_created_remotely = create_synapses_process_responses(synapse_creation_requests_outgoing, received_responses);

    GlobalTimers::timers.stop_and_add(TimerRegion::CREATE_SYNAPSES);

    const auto num_synapses_created = synapses_created_locally + synapses_created_remotely;

    return num_synapses_created;
}

void Neurons::create_synapses_update_octree() {
    // Lock local RMA memory for local stores
    MPIWrapper::lock_window(MPIWrapper::get_my_rank(), MPI_Locktype::exclusive);

    // Update my local trees bottom-up
    GlobalTimers::timers.start(TimerRegion::UPDATE_LOCAL_TREES);
    global_tree->update_local_trees(*dendrites_exc, *dendrites_inh, *axons, num_neurons);
    GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_LOCAL_TREES);

    global_tree->synchronize_local_trees();

    // Unlock local RMA memory and make local stores visible in public window copy
    MPIWrapper::unlock_window(MPIWrapper::get_my_rank());

    // Makes sure that all ranks finished their local access epoch
    // before a remote origin opens an access epoch
    MPIWrapper::barrier(MPIWrapper::Scope::global);
}

void Neurons::make_creation_request_for(
    SignalType needed,
    MapSynapseCreationRequests& request,
    std::stack<OctreeNode*>& nodes_with_ax,
    std::vector<OctreeNode*>& nodes_with_dend) {

    while (!nodes_with_ax.empty()) {
        OctreeNode* current_node = nodes_with_ax.top();
        nodes_with_ax.pop();

        /*
        - check if node is single neuron
        - set interaction list of current node when level 3
        - calculate atractiveness
        - do random experiment
        - set interaction list
        - push children to stack
        */

       // for nodes at level 1
        if (current_node->get_level() == 1) {
            for (size_t i = 0; i < nodes_with_dend.size(); i++) {
                //if the target and source nodes are different, the target node can simply be added to the interaction list 
                current_node->add_to_interactionlist(nodes_with_dend.at(i));
            }    
        }
        
        /*if (current_node->get_cell().get_number_axons_for(needed) > Constants::max_neurons_in_source) {
            current_node->print_calculations(needed, Octree::default_sigma);
        }*/
       
        //node is a leaf
        if (!current_node->is_parent()){
            size_t source_id = current_node->get_cell().get_neuron_id();
            size_t target_id = 0;
            int target_num = current_node->get_interactionlist_length();
            const OctreeNode* target_node;

            if (target_num == 1){
                target_node = current_node->get_from_interactionlist(0);
            }else {
                const std::vector<double> temp = global_tree->calc_attractiveness_to_connect_FMM(current_node, needed);
                target_node = global_tree->do_random_experiment(current_node, temp).value();
            }
            current_node->reset_interactionlist();
            if (target_node->is_parent()) {
                for (unsigned int i = 0; i < 8; i++) {
                    const OctreeNode* target_child = target_node->get_child(i);
                    if (target_child != nullptr){
                        if (target_child != current_node && target_child->get_cell().get_number_dendrites_for(needed) > 0) {
                            current_node->add_to_interactionlist(target_child);
                        }
                    }
                }
                    nodes_with_ax.push(current_node);
                //otherwise the target id is fetched     
            } else {
                target_id = target_node->get_cell().get_neuron_id();
                if (target_id != source_id) {
                    //printf("source id = %i | target_id = %i \n", source_id, target_id);
                    request[0].append(source_id, target_id, needed);
                }
            }
            continue;
        }   

        
        if (current_node->get_interactionlist_length() > 0) {
            //find target node for one source node
            const std::vector<double> temp = global_tree->calc_attractiveness_to_connect_FMM(current_node, needed);
            RelearnException::check(global_tree->do_random_experiment(current_node, temp).has_value(), "Random experiment gave no value back.");
            const OctreeNode* target_node = global_tree->do_random_experiment(current_node, temp).value();

            if (target_node->is_parent()){
                for (size_t i = 0; i < 8; i++) {
                    OctreeNode* child_node = current_node->get_child(i);
                    if (child_node != nullptr && child_node->get_cell().get_number_axons_for(needed) > 0) {
                        for (int j = 0; j < 8; j++) {
                            const OctreeNode* child_target_node = target_node->get_child(j);
                            if (child_target_node != nullptr && child_target_node->get_cell().get_number_dendrites_for(needed) > 0)
                                child_node->add_to_interactionlist(child_target_node);
                        }
                        //go one level down in the tree and put the children on the stack 
                        nodes_with_ax.push(child_node);
                    }
                }
            }
            else{
                std::vector<double> attractiveness;
                std::vector<double> index;
                for(unsigned int i = 0; i < Constants::number_oct; i++){
                    OctreeNode* child_node = current_node->get_child(i);
                    if(child_node != nullptr && child_node->get_cell().get_number_axons_for(needed) > 0){
                        child_node->add_to_interactionlist(target_node);
                        const std::vector<double> temp = global_tree->calc_attractiveness_to_connect_FMM(child_node, needed);
                        attractiveness.push_back(temp[0]);
                        index.push_back(i);
                    }
                }
                int x = Functions::choose_interval(attractiveness);
                nodes_with_ax.push(current_node->get_child(index[x]));
            }
        }
        // the interaction list must be reset for each node 
        current_node->reset_interactionlist();
    }
}

MapSynapseCreationRequests Neurons::create_synapses_find_targets() {
     MapSynapseCreationRequests synapse_creation_requests_outgoing;
    GlobalTimers::timers.start(TimerRegion::FIND_TARGET_NEURONS);

    //init
    OctreeNode* root = global_tree->get_local_root(0);

    std::stack<OctreeNode*> nodes_with_ax_in;
    std::stack<OctreeNode*> nodes_with_ax_ex;
    std::stack<OctreeNode*> init_stack;
    init_stack.push(root);

    std::vector<OctreeNode*> nodes_with_dend_in;
    std::vector<OctreeNode*> nodes_with_dend_ex;

    OctreeNode* current_node;
    OctreeNode* child_node;
    while (!init_stack.empty()) {

        current_node = init_stack.top();
        init_stack.pop();
        if (current_node->get_level() < 1) {
            for (size_t j = 0; j < 8; j++) {
                child_node = current_node->get_child(j);
                if (child_node != nullptr) {
                    init_stack.push(child_node);
                }
            }
        }
        //sort nodes in their corresponding list
        else {
            if (current_node->get_cell().get_number_excitatory_axons() > 0) {
                nodes_with_ax_ex.push(current_node);
            }
            if (current_node->get_cell().get_number_inhibitory_axons() > 0)
                nodes_with_ax_in.push(current_node);
            //prepare interaction lists
            if (current_node->get_cell().get_number_inhibitory_dendrites() > 0) {
                nodes_with_dend_in.push_back(current_node);
            }
            if (current_node->get_cell().get_number_excitatory_dendrites() > 0)
                nodes_with_dend_ex.push_back(current_node);
        }
    }
    if (!nodes_with_ax_ex.empty() && nodes_with_dend_ex.size() > 0) {
        make_creation_request_for(SignalType::EXCITATORY, synapse_creation_requests_outgoing, nodes_with_ax_ex, nodes_with_dend_ex);
    }
    if (!nodes_with_ax_in.empty() && nodes_with_dend_in.size() > 0) {
        make_creation_request_for(SignalType::INHIBITORY, synapse_creation_requests_outgoing, nodes_with_ax_in, nodes_with_dend_in);
    }

    GlobalTimers::timers.stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

    // Make cache empty for next connectivity update
    GlobalTimers::timers.start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree->empty_remote_nodes_cache();
    GlobalTimers::timers.stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}


MapSynapseCreationRequests Neurons::create_synapses_exchange_requests(const MapSynapseCreationRequests& synapse_creation_requests_outgoing) {
    MapSynapseCreationRequests synapse_creation_requests_incoming;

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
    // Now I know how many requests I will get from every rank.
    // Allocate memory for all incoming synapse requests.
    for (auto rank = 0; rank < MPIWrapper::get_num_ranks(); rank++) {
        if (auto num_requests = num_synapse_requests_from_ranks[rank]; 0 != num_requests) {
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

    return synapse_creation_requests_incoming;
}

std::pair<size_t, std::map<int, std::vector<char>>> Neurons::create_synapses_process_requests(const MapSynapseCreationRequests& synapse_creation_requests_incoming) {
    if (synapse_creation_requests_incoming.empty()) {
        return std::make_pair(0, std::map<int, std::vector<char>>{});
    }

    size_t num_synapses_created = 0;
    std::map<int, std::vector<char>> responses;

    for (const auto& it : synapse_creation_requests_incoming) {
        const auto source_rank = it.first;
        const SynapseCreationRequests& requests = it.second;
        const auto num_requests = requests.size();

        responses[source_rank].resize(num_requests);

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

            int num_axons_connected_increment = 0;

            const std::vector<double>* dendrites_cnts = nullptr; // TODO(fabian) find a nicer solution
            const std::vector<unsigned int>* dendrites_connected_cnts = nullptr;

            // DendriteType::INHIBITORY dendrite requested
            if (1 == dendrite_type_needed) {
                dendrites_cnts = &dendrites_inh->get_cnts();
                dendrites_connected_cnts = &dendrites_inh->get_connected_cnts();
                num_axons_connected_increment = -1;
            }
            // DendriteType::EXCITATORY dendrite requested
            else {
                dendrites_cnts = &dendrites_exc->get_cnts();
                dendrites_connected_cnts = &dendrites_exc->get_connected_cnts();
                num_axons_connected_increment = +1;
            }

            // Target neuron has still dendrite available, so connect
            RelearnException::check((*dendrites_cnts)[target_neuron_id] - (*dendrites_connected_cnts)[target_neuron_id] >= 0, "Connectivity went downside");

            const auto diff = static_cast<unsigned int>((*dendrites_cnts)[target_neuron_id] - (*dendrites_connected_cnts)[target_neuron_id]);
            if (diff != 0) {
                // Increment num of connected dendrites
                if (1 == dendrite_type_needed) {
                    dendrites_inh->update_conn_cnt(target_neuron_id, 1);
                } else {
                    dendrites_exc->update_conn_cnt(target_neuron_id, 1);
                }

                // Update network
                network_graph->add_edge_weight(target_neuron_id, MPIWrapper::get_my_rank(), source_neuron_id, source_rank, num_axons_connected_increment);

                // Set response to "connected" (success)
                responses[source_rank][request_index] = 1;
                num_synapses_created++;
            } else {
                // Other axons were faster and came first
                // Set response to "not connected" (not success)
                responses[source_rank][request_index] = 0;
            }
        } // All requests of a rank
    } // Increasing order of ranks that sent requests

    return std::make_pair(num_synapses_created, responses);
}

std::map<int, std::vector<char>> Neurons::create_synapses_exchange_responses(const std::map<int, std::vector<char>>& synapse_creation_responses, const MapSynapseCreationRequests& synapse_creation_requests_outgoing) {
    /**
    * Send and receive responses for synapse requests
    */
    int mpi_requests_index = 0;
    std::vector<MPIWrapper::AsyncToken>
        mpi_requests(synapse_creation_requests_outgoing.size() + synapse_creation_responses.size());

    std::map<int, std::vector<char>> received_responses;

    for (const auto& it : synapse_creation_requests_outgoing) {
        const auto rank = it.first;
        const auto size = it.second.size();

        received_responses[rank].resize(size);
    }

    // Receive responses
    for (auto& it : received_responses) {
        const auto rank = it.first;
        auto* buffer = it.second.data();
        const auto size_in_bytes = static_cast<int>(it.second.size());

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

        mpi_requests_index++;
    }
    // Send responses
    for (const auto& it : synapse_creation_responses) {
        const auto rank = it.first;
        const auto* const buffer = it.second.data();
        const auto size_in_bytes = static_cast<int>(it.second.size());

        MPIWrapper::async_send(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

        mpi_requests_index++;
    }
    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);

    return received_responses;
}

size_t Neurons::create_synapses_process_responses(const MapSynapseCreationRequests& synapse_creation_requests_outgoing, const std::map<int, std::vector<char>>& received_responses) {
    size_t num_synapses_created = 0;
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
            char connected = received_responses.at(target_rank)[request_index];
            size_t source_neuron_id{ Constants::uninitialized };
            size_t target_neuron_id{ Constants::uninitialized };
            size_t dendrite_type_needed{ Constants::uninitialized };
            std::tie(source_neuron_id, target_neuron_id, dendrite_type_needed) = requests.get_request(request_index);

            // Request to form synapse succeeded
            if (connected != 0) {
                // Increment num of connected axons
                axons->update_conn_cnt(source_neuron_id, 1);
                //axons_connected_cnts[source_neuron_id]++;
                num_synapses_created++;

                const double delta = axons->get_cnt(source_neuron_id) - axons->get_connected_cnt(source_neuron_id);
                RelearnException::check(delta >= 0, "%f", delta);

                // I have already created the synapse in the network
                // if the response comes from myself
                if (target_rank != MPIWrapper::get_my_rank()) {
                    // Update network
                    const auto num_axons_connected_increment_2 = (1 == dendrite_type_needed) ? -1 : +1;
                    network_graph->add_edge_weight(target_neuron_id, target_rank, source_neuron_id, MPIWrapper::get_my_rank(), num_axons_connected_increment_2);
                }
            } else {
                // Other axons were faster and came first
            }
        } // All responses from a rank
    } // All outgoing requests

    return num_synapses_created;
}

void Neurons::debug_check_counts() {
    if (!Config::do_debug_checks) {
        return;
    }

    RelearnException::check(network_graph != nullptr, "network_graph is nullptr");

    const std::vector<double>& axs_count = axons->get_cnts();
    const std::vector<unsigned int>& axs_conn_count = axons->get_connected_cnts();
    const std::vector<double>& de_count = dendrites_exc->get_cnts();
    const std::vector<unsigned int>& de_conn_count = dendrites_exc->get_connected_cnts();
    const std::vector<double>& di_count = dendrites_inh->get_cnts();
    const std::vector<unsigned int>& di_conn_count = dendrites_inh->get_connected_cnts();

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

        const auto num_conn_axons = static_cast<size_t>(connected_axons);
        const auto num_conn_dend_ex = static_cast<size_t>(connected_dend_exc);
        const auto num_conn_dend_in = static_cast<size_t>(connected_dend_inh);

        const size_t num_out_ng = network_graph->get_num_out_edges(i);
        const size_t num_in_exc_ng = network_graph->get_num_in_edges_ex(i);
        const size_t num_in_inh_ng = network_graph->get_num_in_edges_in(i);

        RelearnException::check(num_conn_axons == num_out_ng, "In Neurons conn axons, %u vs. %u", num_conn_axons, num_out_ng);
        RelearnException::check(num_conn_dend_ex == num_in_exc_ng, "In Neurons conn dend ex, %u vs. %u", num_conn_dend_ex, num_in_exc_ng);
        RelearnException::check(num_conn_dend_in == num_in_inh_ng, "In Neurons conn dend in, %u vs. %u", num_conn_dend_in, num_in_inh_ng);
    }
}

[[nodiscard]] std::tuple<size_t, size_t> Neurons::update_connectivity() {
    RelearnException::check(network_graph != nullptr, "Network graph is nullptr");
    RelearnException::check(global_tree != nullptr, "Global octree is nullptr");

    debug_check_counts();
    network_graph->debug_check();
    size_t num_synapses_deleted = delete_synapses();
    debug_check_counts();
    network_graph->debug_check();
    size_t num_synapses_created = create_synapses();
    debug_check_counts();
    network_graph->debug_check();

    return std::make_tuple(num_synapses_deleted, num_synapses_created);
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

    const std::vector<double>& cnts_ax = axons->get_cnts();
    const std::vector<unsigned int>& connected_cnts_ax = axons->get_connected_cnts();
    const std::vector<SignalType>& signal_types = axons->get_signal_types();

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
    const std::vector<double>& cnts_den_ex = dendrites_exc->get_cnts();
    const std::vector<unsigned int>& connected_cnts_den_ex = dendrites_exc->get_connected_cnts();
    for (size_t neuron_id = 0; neuron_id < this->num_neurons; ++neuron_id) {
        sum_dends_exc_cnts += static_cast<unsigned int>(cnts_den_ex[neuron_id]);
        sum_dends_exc_connected_cnts += static_cast<unsigned int>(connected_cnts_den_ex[neuron_id]);
    }
    sum_dends_exc_vacant = sum_dends_exc_cnts - sum_dends_exc_connected_cnts;

    // Inh.
    sum_dends_inh_cnts = sum_dends_inh_connected_cnts = 0;
    const std::vector<double>& cnts_den_in = dendrites_inh->get_cnts();
    const std::vector<unsigned int>& connected_cnts_den_in = dendrites_inh->get_connected_cnts();
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

    // NOLINTNEXTLINE
    RelearnException::check(Constants::num_items_per_request > 5, "In Neurons, number of items per request is smaller than 6");
    std::array<unsigned int, Constants::num_items_per_request> sums_global{ 0, 0, 0, 0, 0, 0 }; // Init all to zero

    MPIWrapper::reduce(sums_local, sums_global, MPIWrapper::ReduceFunction::sum, 0, MPIWrapper::Scope::global);

    //get total and connected sum of axons and dendrites
    unsigned int sum_axons_ex = 0;
    unsigned int sum_axons_in = 0;
    unsigned int sum_dendrites_ex = 0;
    unsigned int sum_dendrites_in = 0;
    unsigned int sum_con_axons_ex = 0;
    unsigned int sum_con_axons_in = 0;
    unsigned int sum_con_dendrites_ex = 0;
    unsigned int sum_con_dendrites_in = 0;

    for(unsigned int i =0; i<num_neurons;i++){
        if (axons->get_signal_type(i)==SignalType::EXCITATORY){
            sum_axons_ex += axons->get_cnt(i);
            sum_dendrites_ex += dendrites_exc->get_cnt(i);
            sum_con_axons_ex += axons->get_connected_cnt(i);
            sum_con_dendrites_ex +=  dendrites_exc->get_connected_cnt(i);
        }
        if (axons->get_signal_type(i)==SignalType::INHIBITORY){
            sum_axons_in += axons->get_cnt(i);
            sum_dendrites_in += dendrites_inh->get_cnt(i);
            sum_con_axons_in += axons->get_connected_cnt(i);
            sum_con_dendrites_in +=  dendrites_inh->get_connected_cnt(i);
        }
    }


    // Output data
    if (0 == MPIWrapper::get_my_rank()) {
        std::stringstream ss;
        const int cwidth = 20; // Column width

        // Write headers to file if not already done so
        if (0 == step) {
            ss << "# SUMS OVER ALL NEURONS\n";
            ss << std::left
               << std::setw(cwidth) << "# step,"
               << std::setw(cwidth) << "Axons exc. (vacant),"
               << std::setw(cwidth) << "Axons inh. (vacant),"
               << std::setw(cwidth) << "Dends exc. (vacant),"
               << std::setw(cwidth) << "Dends inh. (vacant),"
               << std::setw(cwidth) << "Axons total exc.,"
               << std::setw(cwidth) << "Axons total inh.,"
               << std::setw(cwidth) << "Dends total exc.,"
               << std::setw(cwidth) << "Dends total inh.,"
               << std::setw(cwidth) << "Axons con. exc.,"
               << std::setw(cwidth) << "Axons con. inh.,"
               << std::setw(cwidth) << "Dends con. exc.,"
               << std::setw(cwidth) << "Dends con. inh.,"
               << std::setw(cwidth) << "Synapses deleted,"
               << std::setw(cwidth) << "Synapses created"
               << "\n";
        }

        const auto last_idx = 5;

        // Write data at step "step"
        ss << std::left
           << std::setw(cwidth) << step <<","
           << std::setw(cwidth) << sums_global[0] <<","
           << std::setw(cwidth) << sums_global[1] <<","
           << std::setw(cwidth) << sums_global[2] <<","
           << std::setw(cwidth) << sums_global[3] <<","
           << std::setw(cwidth) << sum_axons_ex <<","
           << std::setw(cwidth) << sum_axons_in <<","
           << std::setw(cwidth) << sum_dendrites_ex <<","
           << std::setw(cwidth) << sum_dendrites_in <<","
           << std::setw(cwidth) << sum_con_axons_ex <<","
           << std::setw(cwidth) << sum_con_axons_in <<","
           << std::setw(cwidth) << sum_con_dendrites_ex <<","
           << std::setw(cwidth) << sum_con_dendrites_in <<","
           << std::setw(cwidth) << sums_global[4] / 2 <<","// As counted on both of the neurons
           << std::setw(cwidth) << sums_global[last_idx] / 2 <<","// As counted on both of the neurons
           << "\n";

        LogFiles::write_to_file(LogFiles::EventType::Sums, ss.str(), false);
    }
}

void Neurons::print_neurons_overview_to_log_file_on_rank_0(size_t step) {
    const auto total_num_neurons = partition->get_total_num_neurons();

    const StatisticalMeasures& calcium_statistics = global_statistics(calcium, num_neurons, total_num_neurons, 0);
    const StatisticalMeasures& activity_statistics = global_statistics(neuron_model->get_x(), num_neurons, total_num_neurons, 0);
    const StatisticalMeasures& axons_statistics = global_statistics(axons->get_cnts(), num_neurons, total_num_neurons, 0);
    //const StatisticalMeasures& axons_conn_statistics = global_statistics_integral(axons->get_connected_cnts(), num_neurons, total_num_neurons, 0);
    const StatisticalMeasures& axons_free_statistics = global_statistics(axons->get_vacant_cnts(), num_neurons, total_num_neurons, 0);
    const StatisticalMeasures& dendrites_ex_statistics = global_statistics(dendrites_exc->get_cnts(), num_neurons, total_num_neurons, 0);
    const StatisticalMeasures& dendrites_ex_free_statistics = global_statistics(dendrites_exc->get_vacant_cnts(), num_neurons, total_num_neurons, 0);
    const StatisticalMeasures& dendrites_in_statistics = global_statistics(dendrites_inh->get_cnts(), num_neurons, total_num_neurons, 0);
    const StatisticalMeasures& dendrites_in_free_statistics = global_statistics(dendrites_inh->get_vacant_cnts(), num_neurons, total_num_neurons, 0);

    if (0 != MPIWrapper::get_my_rank()) {
        // All ranks must compute the statistics, but only one should print them
        return;
    }

    std::stringstream ss;
    ss.precision(Constants::print_precision);

    constexpr int cwidth = 31 + 5 * Constants::print_precision;

    auto print_statistics = [&ss](const StatisticalMeasures& statistics) {
        ss << std::setw(4 + Constants::print_precision) << std::right << std::fixed << statistics.avg
           << ", " << std::setw(4 + Constants::print_precision) << std::right << std::fixed << statistics.min
           << ", " << std::setw(4 + Constants::print_precision) << std::right << std::fixed << statistics.max
           << ", " << std::setw(4 + Constants::print_precision) << std::right << std::fixed << statistics.var
           << ", " << std::setw(4 + Constants::print_precision) << std::right << std::fixed << statistics.std;
        ;
    };

    // Write headers to file if not already done so
    if (0 == step) {
        ss << "# ALL NEURONS\n";
        ss << std::left
           << std::setw(12) << "#,"
           << std::setw(cwidth) << " Calcium,"
           << std::setw(cwidth) << "\t Activity,"
           << std::setw(cwidth) << "\t Axons,"
           << std::setw(cwidth) << "\t Axons free,"
           << std::setw(cwidth) << "\t Dendrites_ex,"
           << std::setw(cwidth) << "\t Dendrites_ex free,"
           << std::setw(cwidth) << "\t Dendrites_in,"
           << std::setw(cwidth) << "\t Dendrites_in free,"
           << "\n";
        ss << std::left
           << std::setw(12) << "# step,"
           << std::setw(5 + Constants::print_precision) << std::right << "C-avg,"
           << std::setw(6 + Constants::print_precision) << std::right << "C-min,"
           << std::setw(6 + Constants::print_precision) << std::right << "C-max,"
           << std::setw(6 + Constants::print_precision) << std::right << "C-var,"
           << std::setw(5 + Constants::print_precision) << std::right << "C-std_dev," << '\t'
           << std::setw(5 + Constants::print_precision) << std::right << "Act-avg,"
           << std::setw(6 + Constants::print_precision) << std::right << "Act-min,"
           << std::setw(6 + Constants::print_precision) << std::right << "Act-max,"
           << std::setw(6 + Constants::print_precision) << std::right << "Act-var,"
           << std::setw(5 + Constants::print_precision) << std::right << "Act-std_dev," << '\t'
           << std::setw(5 + Constants::print_precision) << std::right << "Ax-avg,"
           << std::setw(6 + Constants::print_precision) << std::right << "Ax-min,"
           << std::setw(6 + Constants::print_precision) << std::right << "Ax-max,"
           << std::setw(6 + Constants::print_precision) << std::right << "Ax-var,"
           << std::setw(5 + Constants::print_precision) << std::right << "Ax-std_dev," << '\t'
           << std::setw(5 + Constants::print_precision) << std::right << "f-Ax-avg,"
           << std::setw(6 + Constants::print_precision) << std::right << "f-Ax-min,"
           << std::setw(6 + Constants::print_precision) << std::right << "f-Ax-max,"
           << std::setw(6 + Constants::print_precision) << std::right << "f-Ax-var,"
           << std::setw(5 + Constants::print_precision) << std::right << "f-Ax-std_dev," << '\t'
           << std::setw(5 + Constants::print_precision) << std::right << "DendEx-avg,"
           << std::setw(6 + Constants::print_precision) << std::right << "DendEx-min,"
           << std::setw(6 + Constants::print_precision) << std::right << "DendEx-max,"
           << std::setw(6 + Constants::print_precision) << std::right << "DendEx-var,"
           << std::setw(5 + Constants::print_precision) << std::right << "DendEx-std_dev," << '\t'
           << std::setw(5 + Constants::print_precision) << std::right << "f-DendEx-avg,"
           << std::setw(6 + Constants::print_precision) << std::right << "f-DendEx-min,"
           << std::setw(6 + Constants::print_precision) << std::right << "f-DendEx-max,"
           << std::setw(6 + Constants::print_precision) << std::right << "f-DendEx-var,"
           << std::setw(5 + Constants::print_precision) << std::right << "f-DendEx-std_dev," << '\t'
           << std::setw(5 + Constants::print_precision) << std::right << "DendIn-avg,"
           << std::setw(6 + Constants::print_precision) << std::right << "DendIn-min,"
           << std::setw(6 + Constants::print_precision) << std::right << "DendIn-max,"
           << std::setw(6 + Constants::print_precision) << std::right << "DendIn-var,"
           << std::setw(5 + Constants::print_precision) << std::right << "DendIn-std_dev," << '\t'
           << std::setw(5 + Constants::print_precision) << std::right << "f-DendIn-avg,"
           << std::setw(6 + Constants::print_precision) << std::right << "f-DendIn-min,"
           << std::setw(6 + Constants::print_precision) << std::right << "f-DendIn-max,"
           << std::setw(6 + Constants::print_precision) << std::right << "f-DendIn-var,"
           << std::setw(5 + Constants::print_precision) << std::right << "f-DendIn-std_dev" << '\t'
           << "\n";
    }

    // Write data at step "step"
    ss << std::left << std::setw(12) << step << ",";
    print_statistics(calcium_statistics);
    ss << "," << '\t';
    print_statistics(activity_statistics);
    ss << "," << '\t';
    print_statistics(axons_statistics);
    ss << "," << '\t';
    print_statistics(axons_free_statistics);
    ss << "," << ',\t';
    print_statistics(dendrites_ex_statistics);
    ss << "," << '\t';
    print_statistics(dendrites_ex_free_statistics);
    ss << "," << '\t';
    print_statistics(dendrites_in_statistics);
    ss << "," << '\t';
    print_statistics(dendrites_in_free_statistics);
    ss << "\n";

    LogFiles::write_to_file(LogFiles::EventType::NeuronsOverview, ss.str(), false);
}

void Neurons::print_network_graph_to_log_file() {
    std::stringstream ss;

    // Write output format to file
    ss << "# " << partition->get_total_num_neurons() << "\n"; // Total number of neurons
    ss << "# <target neuron id> <source neuron id> <weight>"
       << "\n";

    // Write network graph to file
    network_graph->print(ss, extra_info);

    LogFiles::write_to_file(LogFiles::EventType::Network, ss.str(), false);
}

void Neurons::print_positions_to_log_file() {
    std::stringstream ss;

    // Write total number of neurons to log file
    ss << "# " << partition->get_total_num_neurons() << "\n";
    ss << "# "
       << "<global id> <pos x> <pos y> <pos z> <area> <type>"
       << "\n";

    const std::vector<double>& axons_x_dims = extra_info->get_x_dims();
    const std::vector<double>& axons_y_dims = extra_info->get_y_dims();
    const std::vector<double>& axons_z_dims = extra_info->get_z_dims();
    const std::vector<std::string>& area_names = extra_info->get_area_names();

    const std::vector<SignalType>& signal_types = axons->get_signal_types();

    // Print global ids, positions, and areas of local neurons
    const int my_rank = MPIWrapper::get_my_rank();
    ss << std::fixed << std::setprecision(Constants::print_precision);

    for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
        RankNeuronId rank_neuron_id{ my_rank, neuron_id };

        const auto possible_global_id = extra_info->rank_neuron_id2glob_id(rank_neuron_id);
        RelearnException::check(possible_global_id.has_value(), "ret is false");
        const auto global_id = possible_global_id.value();
        const auto& signal_type_name = (signal_types[neuron_id] == SignalType::EXCITATORY) ? std::string("ex") : std::string("int");

        ss << (global_id + 1) << " "
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
        ss << std::setw(cwidth) << neuron_model->get_secondary_variable(i) << std::setw(cwidth) << calcium[i] << std::setw(cwidth) << axons->get_cnt(i);
        ss << std::setw(cwidth) << dendrites_exc->get_cnt(i) << std::setw(cwidth) << dendrites_inh->get_cnt(i) << "\n";
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, ss.str(), true);
}

void Neurons::print_info_for_barnes_hut() {
    const std::vector<double>& x_dims = extra_info->get_x_dims();
    const std::vector<double>& y_dims = extra_info->get_y_dims();
    const std::vector<double>& z_dims = extra_info->get_z_dims();

    const std::vector<double>& axons_cnts = axons->get_cnts();
    const std::vector<double>& dendrites_exc_cnts = dendrites_exc->get_cnts();
    const std::vector<double>& dendrites_inh_cnts = dendrites_inh->get_cnts();

    const std::vector<unsigned int>& axons_connected_cnts = axons->get_connected_cnts();
    const std::vector<unsigned int>& dendrites_exc_connected_cnts = dendrites_exc->get_connected_cnts();
    const std::vector<unsigned int>& dendrites_inh_connected_cnts = dendrites_inh->get_connected_cnts();

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

void Neurons::print_pending_synapse_deletions(const PendingDeletionsV& list) {
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
