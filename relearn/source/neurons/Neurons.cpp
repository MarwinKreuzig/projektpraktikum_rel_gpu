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
#include "../structure/NeuronIdTranslator.h"
#include "../structure/NodeCache.h"
#include "../structure/Octree.h"
#include "../structure/Partition.h"
#include "../util/Random.h"
#include "../util/Timers.h"
#include "../util/Utility.h"
#include "NetworkGraph.h"
#include "helper/RankNeuronId.h"
#include "models/NeuronModels.h"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <optional>
#include <sstream>

void Neurons::init(const size_t num_neurons, std::vector<double> target_calcium_values, std::vector<double> initial_calcium_values) {
    RelearnException::check(num_neurons > 0, "Neurons::init: num_neurons was 0");
    RelearnException::check(num_neurons == target_calcium_values.size(), "Neurons::init: num_neurons was different than target_calcium_values.size()");
    RelearnException::check(num_neurons == initial_calcium_values.size(), "Neurons::init: num_neurons was different than initial_calcium_values.size()");

    number_neurons = num_neurons;

    neuron_model->init(number_neurons);
    extra_info->init(number_neurons);

    axons->init(number_neurons);
    dendrites_exc->init(number_neurons);
    dendrites_inh->init(number_neurons);

    /**
     * Mark dendrites as exc./inh.
     */
    for (size_t i = 0; i < number_neurons; i++) {
        const auto id = NeuronID{ i };
        dendrites_exc->set_signal_type(id, SignalType::EXCITATORY);
        dendrites_inh->set_signal_type(id, SignalType::INHIBITORY);
    }

    disable_flags.resize(number_neurons, UpdateStatus::ENABLED);
    calcium = std::move(initial_calcium_values);
    target_calcium = std::move(target_calcium_values);

    // Init member variables
    for (size_t i = 0; i < number_neurons; i++) {
        // Set calcium concentration
        const auto fired = neuron_model->get_fired(NeuronID{ i });
        if (fired) {
            calcium[i] += neuron_model->get_beta();
        }
    }
}

void Neurons::init_synaptic_elements() {
    const std::vector<double>& axons_cnts = axons->get_grown_elements();
    const std::vector<double>& dendrites_inh_cnts = dendrites_inh->get_grown_elements();
    const std::vector<double>& dendrites_exc_cnts = dendrites_exc->get_grown_elements();

    for (auto i = 0; i < number_neurons; i++) {
        const auto id = NeuronID{ i };
        const size_t axon_connections = network_graph->get_number_out_edges(id);
        const size_t dendrites_ex_connections = network_graph->get_number_excitatory_in_edges(id);
        const size_t dendrites_in_connections = network_graph->get_number_inhibitory_in_edges(id);

        axons->update_grown_elements(id, static_cast<double>(axon_connections));
        dendrites_exc->update_grown_elements(id, static_cast<double>(dendrites_ex_connections));
        dendrites_inh->update_grown_elements(id, static_cast<double>(dendrites_in_connections));

        axons->update_connected_elements(id, static_cast<int>(axon_connections));
        dendrites_exc->update_connected_elements(id, static_cast<int>(dendrites_ex_connections));
        dendrites_inh->update_connected_elements(id, static_cast<int>(dendrites_in_connections));

        RelearnException::check(axons_cnts[i] >= axons->get_connected_elements()[i], "Error is with: %d", i);
        RelearnException::check(dendrites_inh_cnts[i] >= dendrites_inh->get_connected_elements()[i], "Error is with: %d", i);
        RelearnException::check(dendrites_exc_cnts[i] >= dendrites_exc->get_connected_elements()[i], "Error is with: %d", i);
    }
}

size_t Neurons::disable_neurons(const std::vector<NeuronID>& neuron_ids) {
    neuron_model->disable_neurons(neuron_ids);

    const auto my_rank = MPIWrapper::get_my_rank();

    std::vector<unsigned int> deleted_axon_connections(number_neurons, 0);
    std::vector<unsigned int> deleted_dend_ex_connections(number_neurons, 0);
    std::vector<unsigned int> deleted_dend_in_connections(number_neurons, 0);

    size_t number_deleted_out_inh_edges_within = 0;
    size_t number_deleted_out_exc_edges_within = 0;

    size_t weight_deleted_out_exc_edges_within = 0;
    size_t weight_deleted_out_inh_edges_within = 0;

    size_t number_deleted_out_inh_edges_to_outside = 0;
    size_t number_deleted_out_exc_edges_to_outside = 0;

    size_t weight_deleted_out_exc_edges_to_outside = 0;
    size_t weight_deleted_out_inh_edges_to_outside = 0;

    for (const auto neuron_id : neuron_ids) {
        const auto local_out_edges = network_graph->get_local_out_edges(neuron_id);
        const auto distant_out_edges = network_graph->get_distant_out_edges(neuron_id);

        RelearnException::check(distant_out_edges.empty(), "Neurons::disable_neurons:: Currently, disabling neurons is only supported without mpi");

        for (const auto& [target_neuron_id, weight] : local_out_edges) {
            const RankNeuronId target_id{ my_rank, target_neuron_id };
            const RankNeuronId source_id{ my_rank, neuron_id };

            network_graph->add_edge_weight(target_id, source_id, -weight);

            bool is_within = std::binary_search(neuron_ids.begin(), neuron_ids.end(), target_neuron_id);

            if (is_within) {
                if (weight > 0) {
                    deleted_dend_ex_connections[target_neuron_id.id] += weight;
                    number_deleted_out_exc_edges_within++;
                    weight_deleted_out_exc_edges_within += weight;
                } else {
                    deleted_dend_in_connections[target_neuron_id.id] -= weight;
                    number_deleted_out_inh_edges_within++;
                    weight_deleted_out_inh_edges_within += std::abs(weight);
                }
            } else {
                if (weight > 0) {
                    deleted_dend_ex_connections[target_neuron_id.id] += weight;
                    number_deleted_out_exc_edges_to_outside++;
                    weight_deleted_out_exc_edges_to_outside += weight;
                } else {
                    deleted_dend_in_connections[target_neuron_id.id] -= weight;
                    number_deleted_out_inh_edges_to_outside++;
                    weight_deleted_out_inh_edges_to_outside += std::abs(weight);
                }
            }
        }
    }

    size_t number_deleted_in_edges_from_outside = 0;
    size_t weight_deleted_in_edges_from_outside = 0;

    for (const auto neuron_id : neuron_ids) {
        RelearnException::check(neuron_id.id < number_neurons, "Neurons::disable_neurons: There was a too large id: {} vs {}", neuron_id, number_neurons);
        disable_flags[neuron_id.id] = UpdateStatus::DISABLED;

        const auto local_in_edges = network_graph->get_local_in_edges(neuron_id);
        const auto distant_in_edges = network_graph->get_distant_in_edges(neuron_id);
        RelearnException::check(distant_in_edges.empty(), "Neurons::disable_neurons:: Currently, disabling neurons is only supported without mpi");

        for (const auto& [source_neuron_id, weight] : local_in_edges) {
            const RankNeuronId target_id{ my_rank, neuron_id };
            const RankNeuronId source_id{ my_rank, source_neuron_id };

            network_graph->add_edge_weight(target_id, source_id, -weight);

            deleted_axon_connections[source_neuron_id.id] += std::abs(weight);

            bool is_within = std::binary_search(neuron_ids.begin(), neuron_ids.end(), source_neuron_id);

            if (is_within) {
                RelearnException::fail("Neurons::disable_neurons: While disabling neurons, found a within-in-edge that has not been deleted");
            } else {
                weight_deleted_in_edges_from_outside += std::abs(weight);
                number_deleted_in_edges_from_outside++;
            }
        }
    }

    const auto number_deleted_edges_within = number_deleted_out_inh_edges_within + number_deleted_out_exc_edges_within;
    const auto weight_deleted_edges_within = weight_deleted_out_inh_edges_within + weight_deleted_out_exc_edges_within;

    axons->update_after_deletion(deleted_axon_connections, neuron_ids);
    dendrites_exc->update_after_deletion(deleted_dend_ex_connections, neuron_ids);
    dendrites_inh->update_after_deletion(deleted_dend_in_connections, neuron_ids);

    LogFiles::print_message_rank(0, "Deleted {} in-edges with weight {} and ({}, {}) out-edges with weight ({}, {}) (exc., inh.) within the deleted portion",
        number_deleted_edges_within, weight_deleted_edges_within, number_deleted_out_exc_edges_within,
        number_deleted_out_inh_edges_within, weight_deleted_out_exc_edges_within, weight_deleted_out_inh_edges_within);

    LogFiles::print_message_rank(0, "Deleted {} in-edges with weight {} and ({}, {}) out-edges with weight ({}, {}) (exc., inh.) connecting to the outside",
        number_deleted_in_edges_from_outside, weight_deleted_in_edges_from_outside, number_deleted_out_exc_edges_to_outside,
        number_deleted_out_inh_edges_to_outside, weight_deleted_out_exc_edges_to_outside, weight_deleted_out_inh_edges_to_outside);

    LogFiles::print_message_rank(0, "Deleted {} in-edges with weight {} and ({}, {}) out-edges with weight ({}, {}) (exc., inh.) altogether",
        number_deleted_edges_within + number_deleted_in_edges_from_outside,
        weight_deleted_edges_within + weight_deleted_in_edges_from_outside,
        number_deleted_out_exc_edges_within + number_deleted_out_exc_edges_to_outside,
        number_deleted_out_inh_edges_within + number_deleted_out_inh_edges_to_outside,
        weight_deleted_out_exc_edges_within + weight_deleted_out_exc_edges_to_outside,
        weight_deleted_out_inh_edges_within + weight_deleted_out_inh_edges_to_outside);

    const auto deleted_connections_to_outer_world = weight_deleted_in_edges_from_outside + weight_deleted_out_exc_edges_to_outside + weight_deleted_out_inh_edges_to_outside;

    return deleted_connections_to_outer_world + weight_deleted_edges_within;
}

void Neurons::enable_neurons(const std::vector<NeuronID>& neuron_ids) {
    for (const auto neuron_id : neuron_ids) {
        RelearnException::check(neuron_id.id < number_neurons, "Neurons::enable_neurons: There was a too large id: {} vs {}", neuron_id, number_neurons);
        disable_flags[neuron_id.id] = UpdateStatus::ENABLED;
    }
}

void Neurons::create_neurons(const size_t creation_count, const std::vector<double>& new_target_calcium_values, const std::vector<double>& new_initial_calcium_values) {
    RelearnException::check(creation_count == new_target_calcium_values.size(), "Neurons::create_neurons: creation_count was unequal to new_target_calcium_values.size()");
    RelearnException::check(creation_count == new_initial_calcium_values.size(), "Neurons::create_neurons: creation_count was unequal to new_initial_calcium_values.size()");

    const auto current_size = number_neurons;
    const auto new_size = current_size + creation_count;

    neuron_model->create_neurons(creation_count);
    extra_info->create_neurons(creation_count);

    network_graph->create_neurons(creation_count);

    axons->create_neurons(creation_count);
    dendrites_exc->create_neurons(creation_count);
    dendrites_inh->create_neurons(creation_count);

    for (size_t i = current_size; i < new_size; i++) {
        const auto id = NeuronID{ i };
        dendrites_exc->set_signal_type(id, SignalType::EXCITATORY);
        dendrites_inh->set_signal_type(id, SignalType::INHIBITORY);
    }

    disable_flags.resize(new_size, UpdateStatus::ENABLED);

    calcium.insert(calcium.cend(), new_initial_calcium_values.begin(), new_initial_calcium_values.end());
    target_calcium.insert(target_calcium.cend(), new_target_calcium_values.begin(), new_target_calcium_values.end());

    for (size_t i = current_size; i < new_size; i++) {
        // Set calcium concentration
        const auto fired = neuron_model->get_fired(NeuronID{ i });
        if (fired) {
            calcium[i] += neuron_model->get_beta();
        }
    }

    const auto my_rank = MPIWrapper::get_my_rank();

    for (size_t i = current_size; i < new_size; i++) {
        auto id = NeuronID{ i };
        const auto& pos = extra_info->get_position(id);
        global_tree->insert(pos, id, my_rank);
    }

    global_tree->initializes_leaf_nodes(new_size);

    number_neurons = new_size;
}

void Neurons::update_electrical_activity() {
    neuron_model->update_electrical_activity(*network_graph, disable_flags);
    update_calcium();
}

void Neurons::update_calcium() {
    Timers::start(TimerRegion::CALC_ACTIVITY);

    const auto h = neuron_model->get_h();
    const auto tau_C = neuron_model->get_tau_C();
    const auto beta = neuron_model->get_beta();
    const auto& fired = neuron_model->get_fired();

    // The following line is commented as compilers cannot make up their mind whether they want to have the constants shared or not
    //#pragma omp parallel for shared(fired, h, tau_C, beta) default(none)
    // NOLINTNEXTLINE

    const auto val = (1.0 / static_cast<double>(h));

#pragma omp parallel for default(none) shared(fired, h, val, tau_C, beta)
    for (auto neuron_id = 0; neuron_id < calcium.size(); ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::DISABLED) {
            continue;
        }

        // Update calcium depending on the firing
        auto c = calcium[neuron_id];
        if (fired[neuron_id] == 1) {
            for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
                c += val * (-c / tau_C + beta);
            }
        } else {
            for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
                c += val * (-c / tau_C);
            }
        }
        calcium[neuron_id] = c;
    }

    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);
}

StatisticalMeasures Neurons::global_statistics(const std::vector<double>& local_values, const int root, const std::vector<UpdateStatus>& disable_flags) const {
    const auto [d_my_min, d_my_max, d_my_acc, d_num_values] = Util::min_max_acc(local_values, disable_flags);
    const double my_avg = d_my_acc / d_num_values;

    const double d_min = MPIWrapper::reduce(d_my_min, MPIWrapper::ReduceFunction::min, root);
    const double d_max = MPIWrapper::reduce(d_my_max, MPIWrapper::ReduceFunction::max, root);

    const auto num_values = static_cast<double>(MPIWrapper::all_reduce_uint64(d_num_values, MPIWrapper::ReduceFunction::sum));

    // Get global avg at all ranks (needed for variance)
    const double avg = MPIWrapper::all_reduce_double(my_avg, MPIWrapper::ReduceFunction::sum);

    /**
     * Calc variance
     */
    double my_var = 0;
    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::DISABLED) {
            continue;
        }

        my_var += (local_values[neuron_id] - avg) * (local_values[neuron_id] - avg);
    }
    my_var /= num_values;

    // Get global variance at rank "root"
    const double var = MPIWrapper::reduce(my_var, MPIWrapper::ReduceFunction::sum, root);

    // Calc standard deviation
    const double std = sqrt(var);

    return { d_min, d_max, avg, var, std };
}

size_t Neurons::delete_synapses() {
    /**
     * 1. Update number of synaptic elements and delete synapses if necessary
     */

    Timers::start(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

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
    Timers::stop_and_add(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

    return num_synapses_deleted;
}

std::pair<Neurons::PendingDeletionsV, std::vector<size_t>> Neurons::delete_synapses_find_synapses(
    const SynapticElements& synaptic_elements,
    const std::pair<unsigned int, std::vector<unsigned int>>& to_delete,
    const PendingDeletionsV& other_pending_deletions) {

    const auto& sum_to_delete = to_delete.first;
    const auto& number_deletions = to_delete.second;

    PendingDeletionsV pending_deletions{};
    pending_deletions.reserve(sum_to_delete);

    if (sum_to_delete == 0) {
        return std::make_pair(pending_deletions, std::vector<size_t>{});
    }

    const auto element_type = synaptic_elements.get_element_type();

    std::vector<size_t> total_vector_affected_indices{};

    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::DISABLED) {
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

        const auto id = NeuronID{ neuron_id };
        const auto signal_type = synaptic_elements.get_signal_type(id);
        const auto affected_indices = delete_synapses_find_synapses_on_neuron(id, element_type, signal_type, num_synapses_to_delete, pending_deletions, other_pending_deletions);

        total_vector_affected_indices.insert(total_vector_affected_indices.cend(), affected_indices.begin(), affected_indices.end());
    }

    return std::make_pair(pending_deletions, total_vector_affected_indices);
}

std::vector<size_t> Neurons::delete_synapses_find_synapses_on_neuron(
    const NeuronID& neuron_id,
    const ElementType element_type,
    const SignalType signal_type,
    const unsigned int num_synapses_to_delete,
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
        NetworkGraph::DistantEdges out_edges = network_graph->get_all_out_edges(neuron_id);
        current_synapses = delete_synapses_register_edges(out_edges);
    } else {
        // Walk through ingoing edges
        NetworkGraph::DistantEdges in_edges = network_graph->get_all_in_edges(neuron_id, signal_type);
        current_synapses = delete_synapses_register_edges(in_edges);
    }

    RelearnException::check(num_synapses_to_delete <= current_synapses.size(), "Neurons::delete_synapses_find_synapses_on_neuron:: num_synapses_to_delete > last_synapses.size()");

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

        RelearnException::check(synapse_selected != current_synapses.cend(), "Neurons::delete_synapses_find_synapses_on_neuron: Didn't select a synapse to delete");

        RankNeuronId src_neuron_id = RankNeuronId(MPIWrapper::get_my_rank(), neuron_id);
        RankNeuronId tgt_neuron_id = synapse_selected->get_neuron_id();
        auto synapse_id = synapse_selected->get_synapse_id();

        if (!is_axon) {
            src_neuron_id = synapse_selected->get_neuron_id();
            tgt_neuron_id = RankNeuronId(MPIWrapper::get_my_rank(), neuron_id);
        }

        // Check if synapse is already in pending deletions, if not, add it.
        const auto pending_deletion = std::find_if(other_pending_deletions.begin(), other_pending_deletions.end(), [&](auto param) {
            return param.check_light_equality(src_neuron_id, tgt_neuron_id, synapse_id);
        });

        if (pending_deletion == other_pending_deletions.end()) {
            pending_deletions.emplace_back(src_neuron_id, tgt_neuron_id, synapse_selected->get_neuron_id(),
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

std::vector<Neurons::Synapse> Neurons::delete_synapses_register_edges(const std::vector<std::pair<RankNeuronId, int>>& edges) {
    std::vector<Neurons::Synapse> current_synapses;

    for (const auto& it : edges) {
        /**
         * Create "edge weight" number of synapses and add them to the synapse list
         * NOTE: We take abs(it->second) here as DendriteType::INHIBITORY synapses have count < 0
         */
        const auto rank = it.first.get_rank();
        const auto id = it.first.get_neuron_id();

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
    MPIWrapper::all_to_all(num_synapse_deletion_requests_for_ranks, num_synapse_deletion_requests_from_ranks);
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

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

        ++mpi_requests_index;
    }

    // Send actual synapse deletion requests
    for (const auto& it : synapse_deletion_requests_outgoing) {
        const auto rank = it.first;
        const auto* const buffer = it.second.get_requests();
        const auto size_in_bytes = static_cast<int>(it.second.get_requests_size_in_bytes());

        MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

        ++mpi_requests_index;
    }

    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);

    return synapse_deletion_requests_incoming;
}

void Neurons::delete_synapses_process_requests(const MapSynapseDeletionRequests& synapse_deletion_requests_incoming, PendingDeletionsV& pending_deletions) {
    const auto my_rank = MPIWrapper::get_my_rank();

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

            RankNeuronId src_id(my_rank, src_neuron_id);
            RankNeuronId tgt_id(other_rank, tgt_neuron_id);

            if (ElementType::DENDRITE == affected_element_type) {
                src_id = RankNeuronId(other_rank, src_neuron_id);
                tgt_id = RankNeuronId(my_rank, tgt_neuron_id);
            }

            auto pending_deletion = std::find_if(pending_deletions.begin(), pending_deletions.end(), [&src_id, &tgt_id, synapse_id](auto param) {
                return param.check_light_equality(src_id, tgt_id, synapse_id);
            });

            if (pending_deletion == pending_deletions.end()) {
                pending_deletions.emplace_back(src_id, tgt_id, RankNeuronId(my_rank, affected_neuron_id),
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
        const auto& src_neuron = it.get_source_neuron_id();
        const auto src_neuron_rank = src_neuron.get_rank();

        const auto& tgt_neuron = it.get_target_neuron_id();
        const auto tgt_neuron_rank = tgt_neuron.get_rank();

        RelearnException::check(src_neuron_rank == my_rank || tgt_neuron_rank == my_rank, "Neurons::delete_synapses_commit_deletions: Should delete a non-local synapse");

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

        network_graph->add_edge_weight(tgt_neuron, src_neuron, weight_increment);

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
                axons->update_connected_elements(affected_neuron_id, -1);
                continue;
            }

            if (SignalType::EXCITATORY == signal_type) {
                dendrites_exc->update_connected_elements(affected_neuron_id, -1);
            } else {
                dendrites_inh->update_connected_elements(affected_neuron_id, -1);
            }
        }
    }

    return num_synapses_deleted;
}

size_t Neurons::create_synapses() {
    /**
     * 2. Create Synapses
     *
     * - Update region trees (num dendrites in leaves and inner nodes) - postorder traversal (input: grown_elements, connected_elements arrays)
     * - Determine target region for every axon
     * - Find target neuron for every axon (input: position, type; output: target neuron_id)
     * - Update synaptic elements (no connection when target neuron's dendrites have already been taken by previous axon)
     * - Update network
     */

    create_synapses_update_octree();
    // MapSynapseCreationRequests synapse_creation_requests_outgoing = create_synapses_find_targets();
    MapSynapseCreationRequests synapse_creation_requests_outgoing = algorithm->find_target_neurons(number_neurons, disable_flags, extra_info, axons);

    Timers::start(TimerRegion::CREATE_SYNAPSES);

    MapSynapseCreationRequests synapse_creation_requests_incoming = create_synapses_exchange_requests(synapse_creation_requests_outgoing);
    const auto& local_results = create_synapses_process_requests(synapse_creation_requests_incoming);
    const auto synapses_created_locally = local_results.first;

    const auto& received_responses = create_synapses_exchange_responses(local_results.second, synapse_creation_requests_outgoing);
    const auto synapses_created_remotely = create_synapses_process_responses(synapse_creation_requests_outgoing, received_responses);

    Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES);

    const auto num_synapses_created = synapses_created_locally + synapses_created_remotely;

    return num_synapses_created;
}

void Neurons::create_synapses_update_octree() {
    // Lock local RMA memory for local stores
    MPIWrapper::lock_window(MPIWrapper::get_my_rank(), MPI_Locktype::exclusive);

    // Update my local trees bottom-up
    Timers::start(TimerRegion::UPDATE_LEAF_NODES);
    algorithm->update_leaf_nodes(disable_flags, axons, dendrites_exc, dendrites_inh);
    Timers::stop_and_add(TimerRegion::UPDATE_LEAF_NODES);

    // Update my local trees bottom-up
    Timers::start(TimerRegion::UPDATE_LOCAL_TREES);
    global_tree->update_local_trees();
    Timers::stop_and_add(TimerRegion::UPDATE_LOCAL_TREES);

    global_tree->synchronize_local_trees();

    // Unlock local RMA memory and make local stores visible in public window copy
    MPIWrapper::unlock_window(MPIWrapper::get_my_rank());

    // Makes sure that all ranks finished their local access epoch
    // before a remote origin opens an access epoch
    MPIWrapper::barrier();
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
    MPIWrapper::all_to_all(num_synapse_requests_for_ranks, num_synapse_requests_from_ranks);
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

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

        mpi_requests_index++;
    }
    // Send actual synapse requests
    for (const auto& it : synapse_creation_requests_outgoing) {
        const auto rank = it.first;
        const auto* const buffer = it.second.get_requests();
        const auto size_in_bytes = static_cast<int>(it.second.get_requests_size_in_bytes());

        MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

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

    const auto my_rank = MPIWrapper::get_my_rank();

    for (const auto& it : synapse_creation_requests_incoming) {
        const auto source_rank = it.first;
        const SynapseCreationRequests& requests = it.second;
        const auto num_requests = requests.size();

        responses[source_rank].resize(num_requests);

        // All requests of a rank
        for (auto request_index = 0; request_index < num_requests; request_index++) {
            const auto [source_neuron_id, target_neuron_id, dendrite_type_needed] = requests.get_request(request_index);

            // Sanity check: if the request received is targeted for me
            if (target_neuron_id.id >= number_neurons) {
                RelearnException::fail("Neurons::create_synapses_process_requests: Target_neuron_id exceeds my neurons");
            }

            int num_axons_connected_increment = 0;

            const std::vector<double>* dendrites_cnts = nullptr; // TODO(fabian) find a nicer solution
            const std::vector<unsigned int>* dendrites_connected_cnts = nullptr;

            // DendriteType::INHIBITORY dendrite requested
            if (SignalType::INHIBITORY == dendrite_type_needed) {
                dendrites_cnts = &dendrites_inh->get_grown_elements();
                dendrites_connected_cnts = &dendrites_inh->get_connected_elements();
                num_axons_connected_increment = -1;
            }
            // DendriteType::EXCITATORY dendrite requested
            else {
                dendrites_cnts = &dendrites_exc->get_grown_elements();
                dendrites_connected_cnts = &dendrites_exc->get_connected_elements();
                num_axons_connected_increment = +1;
            }

            // Target neuron has still dendrite available, so connect
            RelearnException::check((*dendrites_cnts)[target_neuron_id.id] - (*dendrites_connected_cnts)[target_neuron_id.id] >= 0, "Neurons::create_synapses_process_requests: Connectivity went downside");

            const auto diff = static_cast<unsigned int>((*dendrites_cnts)[target_neuron_id.id] - (*dendrites_connected_cnts)[target_neuron_id.id]);
            if (diff != 0) {
                // Increment num of connected dendrites
                if (SignalType::INHIBITORY == dendrite_type_needed) {
                    dendrites_inh->update_connected_elements(target_neuron_id, 1);
                } else {
                    dendrites_exc->update_connected_elements(target_neuron_id, 1);
                }

                const RankNeuronId target_id{ my_rank, target_neuron_id };
                const RankNeuronId source_id{ source_rank, source_neuron_id };

                // Update network
                network_graph->add_edge_weight(target_id, source_id, num_axons_connected_increment);

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

std::map<int, std::vector<char>> Neurons::create_synapses_exchange_responses(
    const std::map<int, std::vector<char>>& synapse_creation_responses,
    const MapSynapseCreationRequests& synapse_creation_requests_outgoing) {
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

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

        mpi_requests_index++;
    }
    // Send responses
    for (const auto& it : synapse_creation_responses) {
        const auto rank = it.first;
        const auto* const buffer = it.second.data();
        const auto size_in_bytes = static_cast<int>(it.second.size());

        MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

        mpi_requests_index++;
    }
    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);

    return received_responses;
}

size_t Neurons::create_synapses_process_responses(
    const MapSynapseCreationRequests& synapse_creation_requests_outgoing,
    const std::map<int, std::vector<char>>& received_responses) {
    size_t num_synapses_created = 0;

    const auto my_rank = MPIWrapper::get_my_rank();
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
            const auto [source_neuron_id, target_neuron_id, dendrite_type_needed] = requests.get_request(request_index);

            // Request to form synapse succeeded
            if (connected != 0) {
                // Increment num of connected axons
                axons->update_connected_elements(source_neuron_id, 1);
                // axons_connected_cnts[source_neuron_id]++;
                num_synapses_created++;

                const double delta = axons->get_grown_elements(source_neuron_id) - axons->get_connected_elements(source_neuron_id);
                RelearnException::check(delta >= 0, "Neurons::create_synapses_process_responses: delta is negative: {}", delta);

                // I have already created the synapse in the network
                // if the response comes from myself
                if (target_rank != my_rank) {
                    // Update network
                    const auto num_axons_connected_increment_2 = (SignalType::INHIBITORY == dendrite_type_needed) ? -1 : +1;

                    const RankNeuronId target_id{ target_rank, target_neuron_id };
                    const RankNeuronId source_id{ my_rank, source_neuron_id };

                    network_graph->add_edge_weight(target_id, source_id, num_axons_connected_increment_2);
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

    RelearnException::check(network_graph != nullptr, "Neurons::debug_check_counts: network_graph is nullptr");

    const std::vector<double>& axs_count = axons->get_grown_elements();
    const std::vector<unsigned int>& axs_conn_count = axons->get_connected_elements();
    const std::vector<double>& de_count = dendrites_exc->get_grown_elements();
    const std::vector<unsigned int>& de_conn_count = dendrites_exc->get_connected_elements();
    const std::vector<double>& di_count = dendrites_inh->get_grown_elements();
    const std::vector<unsigned int>& di_conn_count = dendrites_inh->get_connected_elements();

    for (size_t i = 0; i < number_neurons; i++) {
        const double diff_axs = axs_count[i] - axs_conn_count[i];
        const double diff_de = de_count[i] - de_conn_count[i];
        const double diff_di = di_count[i] - di_conn_count[i];

        RelearnException::check(diff_axs >= 0.0, "Neurons::debug_check_counts: {}", diff_axs);
        RelearnException::check(diff_de >= 0.0, "Neurons::debug_check_counts: {}", diff_de);
        RelearnException::check(diff_di >= 0.0, "Neurons::debug_check_counts: {}", diff_di);
    }

    for (size_t i = 0; i < number_neurons; i++) {
        const double connected_axons = axs_conn_count[i];
        const double connected_dend_exc = de_conn_count[i];
        const double connected_dend_inh = di_conn_count[i];

        const auto num_conn_axons = static_cast<size_t>(connected_axons);
        const auto num_conn_dend_ex = static_cast<size_t>(connected_dend_exc);
        const auto num_conn_dend_in = static_cast<size_t>(connected_dend_inh);

        const auto id = NeuronID{ i };
        const size_t num_out_ng = network_graph->get_number_out_edges(id);
        const size_t num_in_exc_ng = network_graph->get_number_excitatory_in_edges(id);
        const size_t num_in_inh_ng = network_graph->get_number_inhibitory_in_edges(id);

        RelearnException::check(num_conn_axons == num_out_ng, "Neurons::debug_check_counts: In Neurons conn axons, {} vs. {}", num_conn_axons, num_out_ng);
        RelearnException::check(num_conn_dend_ex == num_in_exc_ng, "Neurons::debug_check_counts: In Neurons conn dend ex, {} vs. {}", num_conn_dend_ex, num_in_exc_ng);
        RelearnException::check(num_conn_dend_in == num_in_inh_ng, "Neurons::debug_check_counts: In Neurons conn dend in, {} vs. {}", num_conn_dend_in, num_in_inh_ng);
    }
}

StatisticalMeasures Neurons::get_statistics(NeuronAttribute attribute) const {
    switch (attribute) {
    case NeuronAttribute::Calcium:
        return global_statistics(calcium, 0, disable_flags);

    case NeuronAttribute::X:
        return global_statistics(neuron_model->get_x(), 0, disable_flags);

    case NeuronAttribute::Fired:
        return global_statistics_integral(neuron_model->get_fired(), 0, disable_flags);

    case NeuronAttribute::I_sync:
        return global_statistics(neuron_model->get_I_syn(), 0, disable_flags);

    case NeuronAttribute::Axons:
        return global_statistics(axons->get_grown_elements(), 0, disable_flags);

    case NeuronAttribute::AxonsConnected:
        return global_statistics_integral(axons->get_connected_elements(), 0, disable_flags);

    case NeuronAttribute::DendritesExcitatory:
        return global_statistics(dendrites_exc->get_grown_elements(), 0, disable_flags);

    case NeuronAttribute::DendritesExcitatoryConnected:
        return global_statistics_integral(dendrites_exc->get_connected_elements(), 0, disable_flags);

    case NeuronAttribute::DendritesInhibitory:
        return global_statistics(dendrites_inh->get_grown_elements(), 0, disable_flags);

    case NeuronAttribute::DendritesInhibitoryConnected:
        return global_statistics_integral(dendrites_inh->get_connected_elements(), 0, disable_flags);
    }

    RelearnException::fail("Neurons::get_statistics: Got an unsupported attribute: {}", attribute);

    return {};
}

[[nodiscard]] std::tuple<size_t, size_t> Neurons::update_connectivity() {
    RelearnException::check(network_graph != nullptr, "Network graph is nullptr");
    RelearnException::check(global_tree != nullptr, "Global octree is nullptr");
    RelearnException::check(algorithm != nullptr, "Algorithm is nullptr");

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

void Neurons::print_sums_of_synapses_and_elements_to_log_file_on_rank_0(const size_t step, const size_t sum_synapses_deleted, const size_t sum_synapses_created) {
    int64_t sum_axons_excitatory_counts = 0;
    int64_t sum_axons_excitatory_connected_counts = 0;
    int64_t sum_axons_inhibitory_counts = 0;
    int64_t sum_axons_inhibitory_connected_counts = 0;

    const auto& axon_counts = axons->get_grown_elements();
    const auto& axons_connected_counts = axons->get_connected_elements();
    const auto& axons_signal_types = axons->get_signal_types();

    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (SignalType::EXCITATORY == axons_signal_types[neuron_id]) {
            sum_axons_excitatory_counts += static_cast<int64_t>(axon_counts[neuron_id]);
            sum_axons_excitatory_connected_counts += static_cast<int64_t>(axons_connected_counts[neuron_id]);
        } else {
            sum_axons_inhibitory_counts += static_cast<int64_t>(axon_counts[neuron_id]);
            sum_axons_inhibitory_connected_counts += static_cast<int64_t>(axons_connected_counts[neuron_id]);
        }
    }

    int64_t sum_dendrites_excitatory_counts = 0;
    int64_t sum_dendrites_excitatory_connected_counts = 0;
    const auto& excitatory_dendrites_counts = dendrites_exc->get_grown_elements();
    const auto& excitatory_dendrites_connected_counts = dendrites_exc->get_connected_elements();
    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        sum_dendrites_excitatory_counts += static_cast<int64_t>(excitatory_dendrites_counts[neuron_id]);
        sum_dendrites_excitatory_connected_counts += static_cast<int64_t>(excitatory_dendrites_connected_counts[neuron_id]);
    }

    int64_t sum_dendrites_inhibitory_counts = 0;
    int64_t sum_dendrites_inhibitory_connected_counts = 0;
    const auto& inhibitory_dendrites_counts = dendrites_inh->get_grown_elements();
    const auto& inhibitory_dendrites_connected_counts = dendrites_inh->get_connected_elements();
    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        sum_dendrites_inhibitory_counts += static_cast<int64_t>(inhibitory_dendrites_counts[neuron_id]);
        sum_dendrites_inhibitory_connected_counts += static_cast<int64_t>(inhibitory_dendrites_connected_counts[neuron_id]);
    }

    int64_t sum_dends_exc_vacant = sum_dendrites_excitatory_counts - sum_dendrites_excitatory_connected_counts;
    int64_t sum_dends_inh_vacant = sum_dendrites_inhibitory_counts - sum_dendrites_inhibitory_connected_counts;

    int64_t sum_axons_exc_vacant = sum_axons_excitatory_counts - sum_axons_excitatory_connected_counts;
    int64_t sum_axons_inh_vacant = sum_axons_inhibitory_counts - sum_axons_inhibitory_connected_counts;

    // Get global sums at rank 0
    std::array<int64_t, 6> sums_local = { sum_axons_exc_vacant,
        sum_axons_inh_vacant,
        sum_dends_exc_vacant,
        sum_dends_inh_vacant,
        static_cast<int64_t>(sum_synapses_deleted),
        static_cast<int64_t>(sum_synapses_created) };

    std::array<int64_t, 6> sums_global = MPIWrapper::reduce(sums_local, MPIWrapper::ReduceFunction::sum, 0);

    // Output data
    if (0 == MPIWrapper::get_my_rank()) {
        const int cwidth = 20; // Column width

        // Write headers to file if not already done so
        if (0 == step) {
            LogFiles::write_to_file(LogFiles::EventType::Sums, false,
                "# SUMS OVER ALL NEURONS\n{1:{0}}{2:{0}}{3:{0}}{4:{0}}{5:{0}}{6:{0}}{7:{0}}",
                cwidth,
                "# step",
                "Axons exc. (vacant)",
                "Axons inh. (vacant)",
                "Dends exc. (vacant)",
                "Dends inh. (vacant)",
                "Synapses deleted",
                "Synapses created");
        }

        LogFiles::write_to_file(LogFiles::EventType::Sums, false,
            "{2:<{0}}{3:<{0}}{4:<{0}}{5:<{0}}{6:<{0}}{7:<{0}}{8:<{0}}",
            cwidth,
            Constants::print_precision,
            step,
            sums_global[0],
            sums_global[1],
            sums_global[2],
            sums_global[3],
            sums_global[4] / 2,
            sums_global[5] / 2);
    }
}

void Neurons::print_neurons_overview_to_log_file_on_rank_0(const size_t step) {
    const StatisticalMeasures& calcium_statistics = get_statistics(NeuronAttribute::Calcium);
    const StatisticalMeasures& axons_statistics = get_statistics(NeuronAttribute::Axons);
    const StatisticalMeasures& axons_connected_statistics = get_statistics(NeuronAttribute::AxonsConnected);
    const StatisticalMeasures& dendrites_excitatory_statistics = get_statistics(NeuronAttribute::DendritesExcitatory);
    const StatisticalMeasures& dendrites_excitatory_connected_statistics = get_statistics(NeuronAttribute::DendritesExcitatoryConnected);

    if (0 != MPIWrapper::get_my_rank()) {
        // All ranks must compute the statistics, but only one should print them
        return;
    }

    const int cwidth = 20; // Column width

    // Write headers to file if not already done so
    if (0 == step) {
        LogFiles::write_to_file(LogFiles::EventType::NeuronsOverview, false,
            "# ALL NEURONS\n{1:{0}}"
            "{2:{0}}{3:{0}}{4:{0}}{5:{0}}{6:{0}}"
            "{7:{0}}{8:{0}}{9:{0}}{10:{0}}{11:{0}}"
            "{12:{0}}{13:{0}}{14:{0}}{15:{0}}{16:{0}}"
            "{17:{0}}{18:{0}}{19:{0}}{20:{0}}{21:{0}}"
            "{22:{0}}{23:{0}}{24:{0}}{25:{0}}{26:{0}}",
            cwidth,
            "# step",
            "C (avg)",
            "C (min)",
            "C (max)",
            "C (var)",
            "C (std_dev)",
            "axons (avg)",
            "axons (min)",
            "axons (max)",
            "axons (var)",
            "axons (std_dev)",
            "axons.c (avg)",
            "axons.c (min)",
            "axons.c (max)",
            "axons.c (var)",
            "axons.c (std_dev)",
            "den.ex (avg)",
            "den.ex (min)",
            "den.ex (max)",
            "den.ex (var)",
            "den.ex (std_dev)",
            "den.ex.c (avg)",
            "den.ex.c (min)",
            "den.ex.c (max)",
            "den.ex.c (var)",
            "den.ex.c (std_dev)");

        LogFiles::write_to_file(LogFiles::EventType::NeuronsOverviewCSV, false,
            "# step",
            "C (avg)",
            "C (min)",
            "C (max)",
            "C (var)",
            "C (std_dev)",
            "axons (avg)",
            "axons (min)",
            "axons (max)",
            "axons (var)",
            "axons (std_dev)",
            "axons.c (avg)",
            "axons.c (min)",
            "axons.c (max)",
            "axons.c (var)",
            "axons.c (std_dev)",
            "den.ex (avg)",
            "den.ex (min)",
            "den.ex (max)",
            "den.ex (var)",
            "den.ex (std_dev)",
            "den.ex.c (avg)",
            "den.ex.c (min)",
            "den.ex.c (max)",
            "den.ex.c (var)",
            "den.ex.c (std_dev)");
    }

    // Write data at step "step"
    LogFiles::write_to_file(LogFiles::EventType::NeuronsOverview, false,
        "{2:<{0}}"
        "{3:<{0}.{1}f}{4:<{0}.{1}f}{5:<{0}.{1}f}{6:<{0}.{1}f}{7:<{0}.{1}f}"
        "{8:<{0}.{1}f}{9:<{0}.{1}f}{10:<{0}.{1}f}{11:<{0}.{1}f}{12:<{0}.{1}f}"
        "{13:<{0}.{1}f}{14:<{0}.{1}f}{15:<{0}.{1}f}{16:<{0}.{1}f}{17:<{0}.{1}f}"
        "{18:<{0}.{1}f}{19:<{0}.{1}f}{20:<{0}.{1}f}{21:<{0}.{1}f}{22:<{0}.{1}f}"
        "{23:<{0}.{1}f}{24:<{0}.{1}f}{25:<{0}.{1}f}{26:<{0}.{1}f}{27:<{0}.{1}f}",
        cwidth,
        Constants::print_precision,
        step,
        calcium_statistics.avg,
        calcium_statistics.min,
        calcium_statistics.max,
        calcium_statistics.var,
        calcium_statistics.std,
        axons_statistics.avg,
        axons_statistics.min,
        axons_statistics.max,
        axons_statistics.var,
        axons_statistics.std,
        axons_connected_statistics.avg,
        axons_connected_statistics.min,
        axons_connected_statistics.max,
        axons_connected_statistics.var,
        axons_connected_statistics.std,
        dendrites_excitatory_statistics.avg,
        dendrites_excitatory_statistics.min,
        dendrites_excitatory_statistics.max,
        dendrites_excitatory_statistics.var,
        dendrites_excitatory_statistics.std,
        dendrites_excitatory_connected_statistics.avg,
        dendrites_excitatory_connected_statistics.min,
        dendrites_excitatory_connected_statistics.max,
        dendrites_excitatory_connected_statistics.var,
        dendrites_excitatory_connected_statistics.std);

    LogFiles::write_to_file(LogFiles::EventType::NeuronsOverviewCSV, false,
        "{};"
        "{};{};{};{};{};"
        "{};{};{};{};{};"
        "{};{};{};{};{};"
        "{};{};{};{};{};"
        "{};{};{};{};{}",
        step,
        calcium_statistics.avg,
        calcium_statistics.min,
        calcium_statistics.max,
        calcium_statistics.var,
        calcium_statistics.std,
        axons_statistics.avg,
        axons_statistics.min,
        axons_statistics.max,
        axons_statistics.var,
        axons_statistics.std,
        axons_connected_statistics.avg,
        axons_connected_statistics.min,
        axons_connected_statistics.max,
        axons_connected_statistics.var,
        axons_connected_statistics.std,
        dendrites_excitatory_statistics.avg,
        dendrites_excitatory_statistics.min,
        dendrites_excitatory_statistics.max,
        dendrites_excitatory_statistics.var,
        dendrites_excitatory_statistics.std,
        dendrites_excitatory_connected_statistics.avg,
        dendrites_excitatory_connected_statistics.min,
        dendrites_excitatory_connected_statistics.max,
        dendrites_excitatory_connected_statistics.var,
        dendrites_excitatory_connected_statistics.std);
}

void Neurons::print_calcium_statistics_to_essentials() {
    const StatisticalMeasures& calcium_statistics = global_statistics(calcium, 0, disable_flags);

    if (0 != MPIWrapper::get_my_rank()) {
        // All ranks must compute the statistics, but only one should print them
        return;
    }

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false,
        "Average calcium: {}\n"
        "Minimum calcium: {}\n"
        "Maximum calcium: {}",
        calcium_statistics.avg,
        calcium_statistics.min,
        calcium_statistics.max);
}

void Neurons::print_network_graph_to_log_file() {
    std::stringstream ss{};

    // Write output format to file
    ss << "# " << partition->get_total_number_neurons() << "\n"; // Total number of neurons
    ss << "# <target neuron id> <source neuron id> <weight>\n";

    if (translator != nullptr) {
        // Write network graph to file
        network_graph->print(ss, translator);
    }

    LogFiles::write_to_file(LogFiles::EventType::Network, false, ss.str());
}

void Neurons::print_positions_to_log_file() {
    // Write total number of neurons to log file
    LogFiles::write_to_file(LogFiles::EventType::Positions, false, "# {}\n#<global id> <pos x> <pos y> <pos z> <area> <type>", partition->get_total_number_neurons());

    const std::vector<std::string>& area_names = extra_info->get_area_names();
    const std::vector<SignalType>& signal_types = axons->get_signal_types();

    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        const auto id = NeuronID{ neuron_id };

        const auto global_id = translator->get_global_id(id);
        const auto& signal_type_name = (signal_types[neuron_id] == SignalType::EXCITATORY) ? std::string("ex") : std::string("in");

        const auto& pos = extra_info->get_position(NeuronID{ neuron_id });

        const auto x = pos.get_x();
        const auto y = pos.get_y();
        const auto z = pos.get_z();

        LogFiles::write_to_file(LogFiles::EventType::Positions, false,
            "{1:<} {2:<.{0}} {3:<.{0}} {4:<.{0}} {5:<} {6:<}",
            Constants::print_precision, (global_id + 1), x, y, z, area_names[neuron_id], signal_type_name);
    }
}

void Neurons::print() {
    // Column widths
    const int cwidth_left = 6;
    const int cwidth = 20;

    std::stringstream ss;

    // Heading
    LogFiles::write_to_file(LogFiles::EventType::Cout, true, "{2:<{1}}{3:<{0}}{4:<{0}}{5:<{0}}{6:<{0}}{7:<{0}}{8:<{0}}{9:<{0}}", cwidth, cwidth_left, "gid", "x", "AP", "refrac", "C", "A", "D_ex", "D_in");

    // Values
    for (size_t i = 0; i < number_neurons; i++) {
        const auto id = NeuronID{ i };
        LogFiles::write_to_file(LogFiles::EventType::Cout, true, "{3:<{1}}{4:<{0}.{2}f}{5:<{0}}{6:<{0}.{2}f}{7:<{0}.{2}f}{8:<{0}.{2}f}{9:<{0}.{2}f}{10:<{0}.{2}f}", cwidth, cwidth_left, Constants::print_precision, i, neuron_model->get_x(id), neuron_model->get_fired(id),
            neuron_model->get_secondary_variable(id), calcium[i], axons->get_grown_elements(id), dendrites_exc->get_grown_elements(id), dendrites_inh->get_grown_elements(id));
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, true, ss.str());
}

void Neurons::print_info_for_algorithm() {
    const std::vector<double>& axons_cnts = axons->get_grown_elements();
    const std::vector<double>& dendrites_exc_cnts = dendrites_exc->get_grown_elements();
    const std::vector<double>& dendrites_inh_cnts = dendrites_inh->get_grown_elements();

    const std::vector<unsigned int>& axons_connected_cnts = axons->get_connected_elements();
    const std::vector<unsigned int>& dendrites_exc_connected_cnts = dendrites_exc->get_connected_elements();
    const std::vector<unsigned int>& dendrites_inh_connected_cnts = dendrites_inh->get_connected_elements();

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
    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        const auto id = NeuronID{ neuron_id };
        ss << std::left << std::setw(cwidth_small) << id;

        const auto [x, y, z] = extra_info->get_position(id);

        my_string = "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")";
        ss << std::setw(cwidth_medium) << my_string;

        my_string = std::to_string(axons_cnts[neuron_id]) + "|" + std::to_string(axons_connected_cnts[neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_exc_cnts[neuron_id]) + "|" + std::to_string(dendrites_exc_connected_cnts[neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_inh_cnts[neuron_id]) + "|" + std::to_string(dendrites_inh_connected_cnts[neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        ss << "\n";
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, true, ss.str());
}

void Neurons::print_local_network_histogram(const size_t current_step) {
    const auto& in_histogram = network_graph->get_edges_histogram(NetworkGraph::EdgeDirection::In);
    const auto& out_histogram = network_graph->get_edges_histogram(NetworkGraph::EdgeDirection::Out);

    const auto conv_hist = [current_step](const std::vector<unsigned int>& hist) -> std::string {
        std::stringstream ss{};
        ss << '#' << current_step;
        for (auto val : hist) {
            ss << ';' << val;
        }

        return ss.str();
    };

    LogFiles::write_to_file(LogFiles::EventType::NetworkInHistogramLocal, false, conv_hist(in_histogram));
    LogFiles::write_to_file(LogFiles::EventType::NetworkOutHistogramLocal, false, conv_hist(out_histogram));
}

void Neurons::print_calcium_values_to_file(const size_t current_step) {
    std::stringstream ss{};

    ss << '#' << current_step;
    for (auto val : calcium) {
        ss << ';' << val;
    }

    LogFiles::write_to_file(LogFiles::EventType::CalciumValues, false, ss.str());
}

void Neurons::print_pending_synapse_deletions(const PendingDeletionsV& list) {
    for (const auto& it : list) {
        size_t affected_element_type_converted = it.get_affected_element_type() == ElementType::AXON ? 0 : 1;
        size_t signal_type_converted = it.get_signal_type() == SignalType::EXCITATORY ? 0 : 1;

        LogFiles::write_to_file(LogFiles::EventType::Cout, true,
            "src_neuron_id: {}\ntgt_neuron_id: {}\naffected_neuron_id: {}\naffected_element_type: {}\nsignal_type: {}\nsynapse_id: {}\naffected_element_already_deleted: {}\n",
            it.get_source_neuron_id(),
            it.get_target_neuron_id(),
            it.get_affected_neuron_id(),
            affected_element_type_converted,
            signal_type_converted,
            it.get_synapse_id(),
            it.get_affected_element_already_deleted());
    }
}
