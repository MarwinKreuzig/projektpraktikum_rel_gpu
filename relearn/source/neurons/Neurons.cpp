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

#include "helper/RankNeuronId.h"
#include "io/LogFiles.h"
#include "models/NeuronModels.h"
#include "mpi/MPIWrapper.h"
#include "neurons/LocalAreaTranslator.h"
#include "neurons/NetworkGraph.h"
#include "io/NeuronIO.h"
#include "sim/Essentials.h"
#include "structure/Octree.h"
#include "structure/Partition.h"
#include "util/Random.h"
#include "util/Timers.h"
#include "util/Utility.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <ranges>
#include <sstream>

void Neurons::init(const number_neurons_type number_neurons) {
    RelearnException::check(number_neurons > 0, "Neurons::init: number_neurons was 0");

    this->number_neurons = number_neurons;

    neuron_model->init(number_neurons);
    extra_info->init(number_neurons);

    axons->init(number_neurons);
    dendrites_exc->init(number_neurons);
    dendrites_inh->init(number_neurons);

    /**
     * Mark dendrites as exc./inh.
     */
    for (const auto& id : NeuronID::range(number_neurons)) {
        dendrites_exc->set_signal_type(id, SignalType::Excitatory);
        dendrites_inh->set_signal_type(id, SignalType::Inhibitory);
    }

    deletions_log.resize(number_neurons, {});

    calcium_calculator->init(number_neurons);

    neuron_model->set_extra_infos(extra_info);

    axons->set_extra_infos(extra_info);
    dendrites_exc->set_extra_infos(extra_info);
    dendrites_inh->set_extra_infos(extra_info);

    calcium_calculator->set_extra_infos(extra_info);
}

void Neurons::init_synaptic_elements(const LocalSynapses & local_synapses_plastic, const DistantInSynapses & in_synapses_plastic, const DistantOutSynapses & out_synapses_plastic) {
    last_created_local_synapses= local_synapses_plastic;
    last_created_in_synapses = in_synapses_plastic;
    last_created_out_synapses = out_synapses_plastic;

    const auto &axons_counts = axons->get_grown_elements();
    const auto &dendrites_inh_counts = dendrites_inh->get_grown_elements();
    const auto &dendrites_exc_counts = dendrites_exc->get_grown_elements();

    for (const auto& id : NeuronID::range(number_neurons)) {
        const auto axon_connections = network_graph_plastic->get_number_out_edges(id);
        const auto dendrites_ex_connections = network_graph_plastic->get_number_excitatory_in_edges(id);
        const auto dendrites_in_connections = network_graph_plastic->get_number_inhibitory_in_edges(id);

        axons->update_grown_elements(id, static_cast<double>(axon_connections));
        dendrites_exc->update_grown_elements(id, static_cast<double>(dendrites_ex_connections));
        dendrites_inh->update_grown_elements(id, static_cast<double>(dendrites_in_connections));

        axons->update_connected_elements(id, static_cast<int>(axon_connections));
        dendrites_exc->update_connected_elements(id, static_cast<int>(dendrites_ex_connections));
        dendrites_inh->update_connected_elements(id, static_cast<int>(dendrites_in_connections));

        const auto local_id = id.get_neuron_id();

        RelearnException::check(axons_counts[local_id] >= axons->get_connected_elements()[local_id],
            "Error is with: %d", local_id);
        RelearnException::check(dendrites_inh_counts[local_id] >= dendrites_inh->get_connected_elements()[local_id],
            "Error is with: %d", local_id);
        RelearnException::check(dendrites_exc_counts[local_id] >= dendrites_exc->get_connected_elements()[local_id],
            "Error is with: %d", local_id);
    }

    check_signal_types(network_graph_plastic, axons->get_signal_types(), MPIWrapper::get_my_rank());
}

void Neurons::check_signal_types(const std::shared_ptr<NetworkGraph> network_graph,
    const std::span<const SignalType> signal_types, const MPIRank my_rank) {
    for (const auto& neuron_id : NeuronID::range(signal_types.size())) {
        const auto& signal_type = signal_types[neuron_id.get_neuron_id()];
        const auto& out_edges = network_graph->get_all_out_edges(neuron_id);
        for (const auto& [tgt_rni, weight] : out_edges) {
            RelearnException::check(SignalType::Excitatory == signal_type && weight > 0 || SignalType::Inhibitory == signal_type && weight < 0,
                "Neuron has outgoing connections not matching its signal type. {} {} -> {} {} {}",
                my_rank, neuron_id, tgt_rni, signal_type, weight);
        }
    }
}

std::pair<size_t, CommunicationMap<SynapseDeletionRequest>> Neurons::disable_neurons(const std::span<const NeuronID> local_neuron_ids, const int num_ranks) {
    extra_info->set_disabled_neurons(local_neuron_ids);

    neuron_model->disable_neurons(local_neuron_ids);

    std::vector<unsigned int> deleted_axon_connections(number_neurons, 0);
    std::vector<unsigned int> deleted_dend_ex_connections(number_neurons, 0);
    std::vector<unsigned int> deleted_dend_in_connections(number_neurons, 0);

    size_t number_deleted_out_inh_edges_within = 0;
    size_t number_deleted_out_exc_edges_within = 0;

    size_t number_deleted_out_inh_edges_to_outside = 0;
    size_t number_deleted_out_exc_edges_to_outside = 0;

    size_t number_deleted_distant_out_axons = 0;
    size_t number_deleted_distant_in_exc = 0;
    size_t number_deleted_distant_in_inh = 0;

    const auto size_hint = std::min(number_neurons, number_neurons_type(num_ranks));
    CommunicationMap<SynapseDeletionRequest> synapse_deletion_requests_outgoing(num_ranks, size_hint);

    for (const auto& neuron_id : local_neuron_ids) {
        RelearnException::check(neuron_id.get_neuron_id() < number_neurons,
            "Neurons::disable_neurons: There was a too large id: {} vs {}", neuron_id,
            number_neurons);

        const auto local_out_edges = network_graph_plastic->get_local_out_edges(neuron_id);
        const auto distant_out_edges = network_graph_plastic->get_distant_out_edges(neuron_id);

        for (const auto& [target_neuron_id, weight] : local_out_edges) {
            network_graph_plastic->add_synapse(LocalSynapse(target_neuron_id, neuron_id, -weight));

            // Shall target_neuron_id also be disabled? Important: Do not remove synapse twice in this case
            const bool is_within = std::ranges::binary_search(local_neuron_ids, target_neuron_id);
            const auto local_target_neuron_id = target_neuron_id.get_neuron_id();

            if (weight > 0) {
                if (is_within) {
                    number_deleted_out_exc_edges_within++;
                    deleted_axon_connections[neuron_id.get_neuron_id()]++;
                    deleted_dend_ex_connections[local_target_neuron_id]++;

                } else {
                    deleted_dend_ex_connections[local_target_neuron_id]++;
                    number_deleted_out_exc_edges_to_outside++;
                }
            } else {
                if (is_within) {
                    number_deleted_out_inh_edges_within++;
                    deleted_axon_connections[neuron_id.get_neuron_id()]++;
                    deleted_dend_in_connections[local_target_neuron_id]++;

                } else {
                    deleted_dend_in_connections[local_target_neuron_id]++;
                    number_deleted_out_inh_edges_to_outside++;
                }
            }
        }

        for (const auto& [target_neuron_id, weight] : distant_out_edges) {
            network_graph_plastic->add_synapse(DistantOutSynapse(target_neuron_id, neuron_id, -weight));
            deleted_axon_connections[neuron_id.get_neuron_id()]++;
            number_deleted_distant_out_axons++;
            const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
            synapse_deletion_requests_outgoing.append(target_neuron_id.get_rank(), { neuron_id, target_neuron_id.get_neuron_id(), ElementType::Axon, signal_type });
        }
    }

    size_t number_deleted_in_edges_from_outside = 0;

    for (const auto& neuron_id : local_neuron_ids) {
        const auto local_in_edges = network_graph_plastic->get_local_in_edges(neuron_id);
        const auto distant_in_edges = network_graph_plastic->get_distant_in_edges(neuron_id);

        for (const auto& [source_neuron_id, weight] : local_in_edges) {
            network_graph_plastic->add_synapse(LocalSynapse(neuron_id, source_neuron_id, -weight));

            deleted_axon_connections[source_neuron_id.get_neuron_id()]++;

            const bool is_within = std::ranges::binary_search(local_neuron_ids, source_neuron_id);

            if (is_within) {
                RelearnException::fail(
                    "Neurons::disable_neurons: While disabling neurons, found a within-in-edge that has not been deleted");
            } else {
                number_deleted_in_edges_from_outside++;
            }
        }

        for (const auto& [source_neuron_id, weight] : distant_in_edges) {
            network_graph_plastic->add_synapse(DistantInSynapse(neuron_id, source_neuron_id, -weight));

            const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
            synapse_deletion_requests_outgoing.append(source_neuron_id.get_rank(), { neuron_id, source_neuron_id.get_neuron_id(), ElementType::Dendrite, signal_type });

            if (weight > 0) {
                deleted_dend_ex_connections[neuron_id.get_neuron_id()]++;
                number_deleted_distant_in_exc++;
            } else {
                deleted_dend_in_connections[neuron_id.get_neuron_id()]++;
                number_deleted_distant_in_inh++;
            }
        }
    }

    const auto number_deleted_edges_within = number_deleted_out_inh_edges_within + number_deleted_out_exc_edges_within;

    axons->update_after_deletion(deleted_axon_connections, local_neuron_ids);
    dendrites_exc->update_after_deletion(deleted_dend_ex_connections, local_neuron_ids);
    dendrites_inh->update_after_deletion(deleted_dend_in_connections, local_neuron_ids);

    LogFiles::print_message_rank(0,
        "Deleted {} in-edges with and ({}, {}) out-edges (exc., inh.) within the deleted portion",
        number_deleted_edges_within,
        number_deleted_out_exc_edges_within,
        number_deleted_out_inh_edges_within);

    LogFiles::print_message_rank(0,
        "Deleted {} in-edges and ({}, {}) out-edges  (exc., inh.) connecting to the outside",
        number_deleted_in_edges_from_outside,
        number_deleted_out_exc_edges_to_outside,
        number_deleted_out_inh_edges_to_outside);

    LogFiles::print_message_rank(0,
        "Deleted ({},{}) in-edges (exc., inh.) and {} out-edges connecting to the other ranks",
        number_deleted_distant_in_exc, number_deleted_distant_in_inh,
        number_deleted_distant_out_axons);

    LogFiles::print_message_rank(0,
        "Deleted {} in-edges and ({}, {}) out-edges (exc., inh.) altogether",
        number_deleted_edges_within + number_deleted_in_edges_from_outside,
        number_deleted_out_exc_edges_within + number_deleted_out_exc_edges_to_outside,
        number_deleted_out_inh_edges_within + number_deleted_out_inh_edges_to_outside);

    const auto deleted_connections = number_deleted_distant_out_axons + number_deleted_distant_in_inh + number_deleted_distant_in_exc
        + number_deleted_in_edges_from_outside + number_deleted_out_inh_edges_to_outside + number_deleted_out_exc_edges_to_outside
        + number_deleted_out_exc_edges_within + number_deleted_out_inh_edges_within;

    return std::make_pair(deleted_connections, synapse_deletion_requests_outgoing);
}

void Neurons::create_neurons(const number_neurons_type creation_count) {
    const auto current_size = number_neurons;
    const auto new_size = current_size + creation_count;

    local_area_translator->create_neurons(creation_count);
    neuron_model->create_neurons(creation_count);
    calcium_calculator->create_neurons(creation_count);
    extra_info->create_neurons(creation_count);

    network_graph_plastic->create_neurons(creation_count);
    network_graph_static->create_neurons(creation_count);

    axons->create_neurons(creation_count);
    dendrites_exc->create_neurons(creation_count);
    dendrites_inh->create_neurons(creation_count);

    deletions_log.resize(new_size, {});

    for (const auto& neuron_id : NeuronID::range(current_size, new_size)) {
        dendrites_exc->set_signal_type(neuron_id, SignalType::Excitatory);
        dendrites_inh->set_signal_type(neuron_id, SignalType::Inhibitory);

        const auto& pos = extra_info->get_position(neuron_id);
        global_tree->insert(pos, neuron_id);
    }

    global_tree->initializes_leaf_nodes(new_size);

    number_neurons = new_size;
}

void Neurons::update_electrical_activity(const step_type step) {
    neuron_model->update_electrical_activity(step, *network_graph_static, *network_graph_plastic);

    const auto& fired = neuron_model->get_fired();
    calcium_calculator->update_calcium(step, fired);

    const auto& calcium_values = calcium_calculator->get_calcium();
    const auto& current_min_id = calcium_calculator->get_current_minimum().get_neuron_id();
    const auto& current_max_id = calcium_calculator->get_current_maximum().get_neuron_id();

    LogFiles::write_to_file(LogFiles::EventType::ExtremeCalciumValues, false, "{};{:.6f};{};{:.6f}",
        current_min_id, calcium_values[current_min_id], current_max_id, calcium_values[current_max_id]);
}

void Neurons::update_number_synaptic_elements_delta() {
    const auto& calcium = calcium_calculator->get_calcium();
    const auto& target_calcium = calcium_calculator->get_target_calcium();

    axons->update_number_elements_delta(calcium, target_calcium);
    dendrites_exc->update_number_elements_delta(calcium, target_calcium);
    dendrites_inh->update_number_elements_delta(calcium, target_calcium);
}

StatisticalMeasures Neurons::global_statistics(const std::span<const double> local_values, const MPIRank root) const {
    const auto disable_flags = extra_info->get_disable_flags();
    const auto [d_my_min, d_my_max, d_my_acc, d_num_values] = Util::min_max_acc(local_values, disable_flags);
    const double my_avg = d_my_acc / static_cast<double>(d_num_values);

    const double d_min = MPIWrapper::reduce(d_my_min, MPIWrapper::ReduceFunction::Min, root);
    const double d_max = MPIWrapper::reduce(d_my_max, MPIWrapper::ReduceFunction::Max, root);

    const auto num_values = static_cast<double>(MPIWrapper::all_reduce_uint64(d_num_values,
        MPIWrapper::ReduceFunction::Sum));

    // Get global avg at all ranks (needed for variance)
    const double avg = MPIWrapper::all_reduce_double(my_avg, MPIWrapper::ReduceFunction::Sum) / MPIWrapper::get_num_ranks();

    /**
     * Calc variance
     */
    double my_var = 0.0;
    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        my_var += (local_values[neuron_id] - avg) * (local_values[neuron_id] - avg);
    }
    my_var /= num_values;

    // Get global variance at rank "root"
    const double var = MPIWrapper::reduce(my_var, MPIWrapper::ReduceFunction::Sum, root);

    // Calc standard deviation
    const double std = std::sqrt(var);

    return { d_min, d_max, avg, var, std };
}

std::pair<uint64_t, uint64_t> Neurons::delete_synapses() {
    auto deletion_helper = [this](const std::shared_ptr<SynapticElements>& synaptic_elements) {
        Timers::start(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

        Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
        const auto to_delete = synaptic_elements->commit_updates();
        Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

        Timers::start(TimerRegion::FIND_SYNAPSES_TO_DELETE);
        const auto outgoing_deletion_requests = delete_synapses_find_synapses(*synaptic_elements, to_delete);
        Timers::stop_and_add(TimerRegion::FIND_SYNAPSES_TO_DELETE);

        Timers::start(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);
        const auto incoming_deletion_requests = MPIWrapper::exchange_requests(outgoing_deletion_requests);
        Timers::stop_and_add(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);

        Timers::start(TimerRegion::PROCESS_DELETE_REQUESTS);
        const auto newly_freed_dendrites = delete_synapses_commit_deletions(incoming_deletion_requests, MPIWrapper::get_my_rank());
        Timers::stop_and_add(TimerRegion::PROCESS_DELETE_REQUESTS);

        Timers::stop_and_add(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

        return newly_freed_dendrites;
    };

    const auto axons_deleted = deletion_helper(axons);
    const auto excitatory_dendrites_deleted = deletion_helper(dendrites_exc);
    const auto inhibitory_dendrites_deleted = deletion_helper(dendrites_inh);

    return { axons_deleted, excitatory_dendrites_deleted + inhibitory_dendrites_deleted };
}

CommunicationMap<SynapseDeletionRequest>
Neurons::delete_synapses_find_synapses(const SynapticElements& synaptic_elements,
    const std::pair<unsigned int, std::vector<unsigned int>>& to_delete) {
    const auto& [sum_to_delete, number_deletions] = to_delete;

    const auto number_ranks = MPIWrapper::get_num_ranks();
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto size_hint = std::min(size_t(number_ranks), synaptic_elements.get_size());
    CommunicationMap<SynapseDeletionRequest> deletion_requests(number_ranks, size_hint);

    if (sum_to_delete == 0) {
        return deletion_requests;
    }

    const auto element_type = synaptic_elements.get_element_type();

    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        if (!extra_info->does_update_plasticity(neuron_id)) {
            continue;
        }

        /**
         * Create and delete synaptic elements as required.
         * This function only deletes elements (bound and unbound), no synapses.
         */
        const auto num_synapses_to_delete = number_deletions[local_neuron_id];
        if (num_synapses_to_delete == 0) {
            continue;
        }

        const auto signal_type = synaptic_elements.get_signal_type(neuron_id);
        const auto affected_neuron_ids = synapse_deletion_finder->find_for_neuron(neuron_id, element_type, signal_type,
                                                                                 num_synapses_to_delete, network_graph_plastic, neuron_model);

        for (const auto& [rank, other_neuron_id] : affected_neuron_ids) {
            SynapseDeletionRequest psd(neuron_id, other_neuron_id, element_type, signal_type);
            deletion_requests.append(rank, psd);

            if (my_rank == rank) {
                continue;
            }

            const auto weight = (SignalType::Excitatory == signal_type) ? -1 : 1;
            if (ElementType::Axon == element_type) {
                network_graph_plastic->add_synapse(
                    DistantOutSynapse(RankNeuronId(rank, other_neuron_id), neuron_id, weight));
            } else {
                network_graph_plastic->add_synapse(
                    DistantInSynapse(neuron_id, RankNeuronId(rank, other_neuron_id), weight));
            }
        }
    }

    return deletion_requests;
}

size_t Neurons::delete_synapses_commit_deletions(const CommunicationMap<SynapseDeletionRequest> &list, const MPIRank& my_rank) {

    size_t num_synapses_deleted = 0;

    for (const auto& [other_rank, requests] : list) {
        num_synapses_deleted += requests.size();

        for (const auto& [other_neuron_id, my_neuron_id, element_type, signal_type] : requests) {
            const auto weight = (SignalType::Excitatory == signal_type) ? -1 : 1;

            deletions_log[my_neuron_id.get_neuron_id()].emplace_back(RankNeuronId(other_rank,other_neuron_id), -weight);

            /**
             *  Update network graph
             */
            if (my_rank == other_rank) {
                if (ElementType::Dendrite == element_type) {
                    network_graph_plastic->add_synapse(LocalSynapse(other_neuron_id, my_neuron_id, weight));
                } else {
                    network_graph_plastic->add_synapse(LocalSynapse(my_neuron_id, other_neuron_id, weight));
                }
            } else {
                if (ElementType::Dendrite == element_type) {
                    network_graph_plastic->add_synapse(
                        DistantOutSynapse(RankNeuronId(other_rank, other_neuron_id), my_neuron_id, weight));
                } else {
                    network_graph_plastic->add_synapse(
                        DistantInSynapse(my_neuron_id, RankNeuronId(other_rank, other_neuron_id), weight));
                }
            }

            if (ElementType::Dendrite == element_type) {
                axons->update_connected_elements(my_neuron_id, -1);
                continue;
            }

            if (SignalType::Excitatory == signal_type) {
                dendrites_exc->update_connected_elements(my_neuron_id, -1);
            } else {
                dendrites_inh->update_connected_elements(my_neuron_id, -1);
            }
        }
    }

    return num_synapses_deleted;
}

size_t Neurons::delete_disabled_distant_synapses(const CommunicationMap<SynapseDeletionRequest>& list, const MPIRank& my_rank) {

    size_t num_synapses_deleted = 0;

    const auto& disable_flags = extra_info->get_disable_flags();

    for (const auto& [other_rank, requests] : list) {
        num_synapses_deleted += requests.size();

        for (const auto& [other_neuron_id, my_neuron_id, element_type, signal_type] : requests) {
            if (disable_flags[my_neuron_id.get_neuron_id()] != UpdateStatus::Enabled) {
                continue;
            }

            /**
             *  Update network graph
             */
            if (my_rank == other_rank) {
                RelearnException::fail("Local synapse deletion is not allowed via mpi");
            }

            if (ElementType::Dendrite == element_type) {
                const auto& out_edges = network_graph_plastic->get_distant_out_edges(my_neuron_id);
                RelearnTypes::synapse_weight weight = 0;
                for (const auto& [target, edge_weight] : out_edges) {
                    if (target.get_rank() == other_rank && target.get_neuron_id() == other_neuron_id) {
                        weight = edge_weight;
                        break;
                    }
                }
                RelearnException::check(weight != 0, "Couldnot find the weight of the connection");
                network_graph_plastic->add_synapse(
                    DistantOutSynapse(RankNeuronId(other_rank, other_neuron_id), my_neuron_id, -weight));
            } else {
                const auto& in_edges = network_graph_plastic->get_distant_in_edges(my_neuron_id);
                RelearnTypes::synapse_weight weight = 0;
                for (const auto& [source, edge_weight] : in_edges) {
                    if (source.get_rank() == other_rank && source.get_neuron_id() == other_neuron_id) {
                        weight = edge_weight;
                        break;
                    }
                }
                network_graph_plastic->add_synapse(
                    DistantInSynapse(my_neuron_id, RankNeuronId(other_rank, other_neuron_id), -weight));
            }

            if (ElementType::Dendrite == element_type) {
                axons->update_connected_elements(my_neuron_id, -1);
                continue;
            }

            if (SignalType::Excitatory == signal_type) {
                dendrites_exc->update_connected_elements(my_neuron_id, -1);
            } else {
                dendrites_inh->update_connected_elements(my_neuron_id, -1);
            }
        }
    }

    return num_synapses_deleted;
}

size_t Neurons::create_synapses() {
    const auto my_rank = MPIWrapper::get_my_rank();

    // Lock local RMA memory for local stores and make them visible afterwards
    MPIWrapper::lock_window(MPIWindow::Window::Octree,my_rank, MPI_Locktype::Exclusive);
    algorithm->update_octree();
    MPIWrapper::unlock_window(MPIWindow::Window::Octree, my_rank);

    // Makes sure that all ranks finished their local access epoch
    // before a remote origin opens an access epoch
    MPIWrapper::barrier();

    // Delegate the creation of new synapses to the algorithm
    const auto& [local_synapses, distant_in_synapses, distant_out_synapses]
        = algorithm->update_connectivity(number_neurons);

    // Update the network graph all at once
    Timers::start(TimerRegion::ADD_SYNAPSES_TO_NETWORK_GRAPH);
    network_graph_plastic->add_edges(local_synapses, distant_in_synapses, distant_out_synapses);
    Timers::stop_and_add(TimerRegion::ADD_SYNAPSES_TO_NETWORK_GRAPH);

    // The distant_out_synapses are counted on the ranks where they are in
    const auto num_synapses_created = local_synapses.size() + distant_in_synapses.size();

    last_created_local_synapses = std::move(local_synapses);
    last_created_in_synapses = std::move(distant_in_synapses);
    last_created_out_synapses = std::move(distant_out_synapses);

    return num_synapses_created;
}

void Neurons::debug_check_counts() {
    if (!Config::do_debug_checks) {
        return;
    }

    RelearnException::check(network_graph_plastic != nullptr,
        "Neurons::debug_check_counts: network_graph_plastic is nullptr");

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& grown_axons = axons->get_grown_elements();
    const auto& connected_axons = axons->get_connected_elements();
    const auto& grown_excitatory_dendrites = dendrites_exc->get_grown_elements();
    const auto& connected_excitatory_dendrites = dendrites_exc->get_connected_elements();
    const auto& grown_inhibitory_dendrites = dendrites_inh->get_grown_elements();
    const auto& connected_inhibitory_dendrites = dendrites_inh->get_connected_elements();

    for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < number_neurons; neuron_id++) {
        const auto vacant_axons = grown_axons[neuron_id] - connected_axons[neuron_id];
        const auto vacant_excitatory_dendrites = grown_excitatory_dendrites[neuron_id] - connected_excitatory_dendrites[neuron_id];
        const auto vacant_inhibitory_dendrites = grown_inhibitory_dendrites[neuron_id] - connected_inhibitory_dendrites[neuron_id];

        RelearnException::check(vacant_axons >= 0.0,
            "Neurons::debug_check_counts: {} has a weird number of vacant axons: {}", neuron_id,
            vacant_axons);
        RelearnException::check(vacant_excitatory_dendrites >= 0.0,
            "Neurons::debug_check_counts: {} has a weird number of vacant excitatory dendrites: {}",
            neuron_id, vacant_excitatory_dendrites);
        RelearnException::check(vacant_inhibitory_dendrites >= 0.0,
            "Neurons::debug_check_counts: {} has a weird number of vacant inhibitory dendrites: {}",
            neuron_id, vacant_inhibitory_dendrites);
    }

    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        const auto connected_axons_neuron = connected_axons[local_neuron_id];
        const auto connected_excitatory_dendrites_neuron = connected_excitatory_dendrites[local_neuron_id];
        const auto connected_inhibitory_dendrites_neuron = connected_inhibitory_dendrites[local_neuron_id];

        const auto number_out_edges = network_graph_plastic->get_number_out_edges(neuron_id);
        const auto number_excitatory_in_edges = network_graph_plastic->get_number_excitatory_in_edges(neuron_id);
        const auto number_inhibitory_in_edges = network_graph_plastic->get_number_inhibitory_in_edges(neuron_id);

        RelearnException::check(connected_axons_neuron == number_out_edges,
            "Neurons::debug_check_counts: Neuron {} has {} axons but {} out edges (rank {})",
            neuron_id, connected_axons_neuron, number_out_edges, my_rank);

        RelearnException::check(connected_excitatory_dendrites_neuron == number_excitatory_in_edges,
            "Neurons::debug_check_counts: Neuron {} has {} excitatory dendrites but {} excitatory in edges (rank {})",
            neuron_id, connected_excitatory_dendrites_neuron, number_excitatory_in_edges, my_rank);

        RelearnException::check(connected_inhibitory_dendrites_neuron == number_inhibitory_in_edges,
            "Neurons::debug_check_counts: Neuron {} has {} inhibitory dendrites but {} inhibitory in edges (rank {})",
            neuron_id, connected_inhibitory_dendrites_neuron, number_inhibitory_in_edges, my_rank);
    }
}

StatisticalMeasures Neurons::get_statistics(const NeuronAttribute attribute) const {
    switch (attribute) {
    case NeuronAttribute::Calcium:
        return global_statistics(calcium_calculator->get_calcium(), MPIRank::root_rank());

    case NeuronAttribute::X:
        return global_statistics(neuron_model->get_x(), MPIRank::root_rank());

    case NeuronAttribute::Fired:
        return global_statistics_integral(neuron_model->get_fired(), MPIRank::root_rank());

    case NeuronAttribute::SynapticInput:
        return global_statistics(neuron_model->get_synaptic_input(), MPIRank::root_rank());

    case NeuronAttribute::BackgroundActivity:
        return global_statistics(neuron_model->get_background_activity(), MPIRank::root_rank());

    case NeuronAttribute::Axons:
        return global_statistics(axons->get_grown_elements(), MPIRank::root_rank());

    case NeuronAttribute::AxonsConnected:
        return global_statistics_integral(axons->get_connected_elements(), MPIRank::root_rank());

    case NeuronAttribute::DendritesExcitatory:
        return global_statistics(dendrites_exc->get_grown_elements(), MPIRank::root_rank());

    case NeuronAttribute::DendritesExcitatoryConnected:
        return global_statistics_integral(dendrites_exc->get_connected_elements(), MPIRank::root_rank());

    case NeuronAttribute::DendritesInhibitory:
        return global_statistics(dendrites_inh->get_grown_elements(), MPIRank::root_rank());

    case NeuronAttribute::DendritesInhibitoryConnected:
        return global_statistics_integral(dendrites_inh->get_connected_elements(), MPIRank::root_rank());
    }

    RelearnException::fail("Neurons::get_statistics: Got an unsupported attribute: {}", static_cast<int>(attribute));

    return {};
}

std::tuple<uint64_t, uint64_t, uint64_t> Neurons::update_connectivity() {
    RelearnException::check(network_graph_plastic != nullptr, "Network graph is nullptr");
    RelearnException::check(global_tree != nullptr, "Global octree is nullptr");
    RelearnException::check(algorithm != nullptr, "Algorithm is nullptr");

    neuron_model->publish_fire_history();

    debug_check_counts();
    network_graph_plastic->debug_check();
    const auto& [num_axons_deleted, num_dendrites_deleted] = delete_synapses();
    debug_check_counts();
    network_graph_plastic->debug_check();
    size_t num_synapses_created = create_synapses();
    debug_check_counts();
    network_graph_plastic->debug_check();

    return { num_axons_deleted, num_dendrites_deleted, num_synapses_created };
}

void Neurons::print_sums_of_synapses_and_elements_to_log_file_on_rank_0(const step_type step,
    const uint64_t sum_axon_deleted,
    const uint64_t sum_dendrites_deleted,
    const uint64_t sum_synapses_created) {
    int64_t sum_axons_excitatory_counts = 0;
    int64_t sum_axons_excitatory_connected_counts = 0;
    int64_t sum_axons_inhibitory_counts = 0;
    int64_t sum_axons_inhibitory_connected_counts = 0;

    const auto& axon_counts = axons->get_grown_elements();
    const auto& axons_connected_counts = axons->get_connected_elements();
    const auto& axons_signal_types = axons->get_signal_types();

    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (SignalType::Excitatory == axons_signal_types[neuron_id]) {
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
    std::array<int64_t, 7> sums_local = { sum_axons_exc_vacant,
        sum_axons_inh_vacant,
        sum_dends_exc_vacant,
        sum_dends_inh_vacant,
        static_cast<int64_t>(sum_axon_deleted),
        static_cast<int64_t>(sum_dendrites_deleted),
        static_cast<int64_t>(sum_synapses_created) };

    std::array<int64_t, 7> sums_global = MPIWrapper::reduce(sums_local, MPIWrapper::ReduceFunction::Sum,
        MPIRank::root_rank());

    // Output data
    if (MPIRank::root_rank() == MPIWrapper::get_my_rank()) {
        const int cwidth = 20; // Column width

        // Write headers to file if not already done so
        if (0 == step) {
            LogFiles::write_to_file(LogFiles::EventType::Sums, false,
                "# SUMS OVER ALL NEURONS\n{1:{0}}{2:{0}}{3:{0}}{4:{0}}{5:{0}}{6:{0}}{7:{0}}{8:{0}}",
                cwidth,
                "# step",
                "Axons exc. (vacant)",
                "Axons inh. (vacant)",
                "Dends exc. (vacant)",
                "Dends inh. (vacant)",
                "Synapses (axons) deleted",
                "Synapses (dendrites) deleted",
                "Synapses created");
        }

        LogFiles::write_to_file(LogFiles::EventType::Sums, false,
            "{2:<{0}}{3:<{0}}{4:<{0}}{5:<{0}}{6:<{0}}{7:<{0}}{8:<{0}}{9:<{0}}",
            cwidth,
            Constants::print_precision,
            step,
            sums_global[0],
            sums_global[1],
            sums_global[2],
            sums_global[3],
            sums_global[4] / 2,
            sums_global[5] / 2,
            sums_global[6] / 2);
    }
}

void Neurons::print_neurons_overview_to_log_file_on_rank_0(const step_type step) const {
    const StatisticalMeasures& calcium_statistics = get_statistics(NeuronAttribute::Calcium);
    const StatisticalMeasures& axons_statistics = get_statistics(NeuronAttribute::Axons);
    const StatisticalMeasures& axons_connected_statistics = get_statistics(NeuronAttribute::AxonsConnected);
    const StatisticalMeasures& dendrites_excitatory_statistics = get_statistics(NeuronAttribute::DendritesExcitatory);
    const StatisticalMeasures& dendrites_excitatory_connected_statistics = get_statistics(
        NeuronAttribute::DendritesExcitatoryConnected);

    if (MPIRank::root_rank() != MPIWrapper::get_my_rank()) {
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

void Neurons::print_calcium_statistics_to_essentials(const std::unique_ptr<Essentials>& essentials) {
    const auto& calcium = calcium_calculator->get_calcium();
    const StatisticalMeasures& calcium_statistics = global_statistics(calcium, MPIRank::root_rank());

    if (MPIRank::root_rank() != MPIWrapper::get_my_rank()) {
        // All ranks must compute the statistics, but only one should print them
        return;
    }

    essentials->insert("Calcium-Minimum", calcium_statistics.min);
    essentials->insert("Calcium-Average", calcium_statistics.avg);
    essentials->insert("Calcium-Maximum", calcium_statistics.max);
}

void Neurons::print_synaptic_changes_to_essentials(const std::unique_ptr<Essentials>& essentials) {
    auto helper = [this, &essentials](const auto& synaptic_elements, std::string message) {
        const auto local_adds = synaptic_elements.get_total_additions();
        const auto local_dels = synaptic_elements.get_total_deletions();

        const auto global_adds = MPIWrapper::reduce(local_adds, MPIWrapper::ReduceFunction::Sum, MPIRank::root_rank());
        const auto global_dels = MPIWrapper::reduce(local_dels, MPIWrapper::ReduceFunction::Sum, MPIRank::root_rank());

        if (MPIRank::root_rank() == MPIWrapper::get_my_rank()) {
            essentials->insert(message + "Additions", global_adds);
            essentials->insert(message + "Deletions", global_dels);
        }
    };

    helper(*axons, "Axons-");
    helper(*dendrites_exc, "Dendrites-Excitatory-");
    helper(*dendrites_inh, "Dendrites-Inhibitory-");
}

void Neurons::print_network_graph_to_log_file(const step_type step, bool with_prefix) const {
    std::string prefix = "";
    if (with_prefix) {
        prefix = "step_" + std::to_string(step) + "_";
    }
    LogFiles::save_and_open_new(LogFiles::EventType::InNetwork, prefix + "in_network", "network/");
    LogFiles::save_and_open_new(LogFiles::EventType::OutNetwork, prefix + "out_network", "network/");

    std::stringstream ss_in_network{};
    std::stringstream ss_out_network{};

    NeuronIO::write_out_synapses(network_graph_static->get_all_local_out_edges(),
        network_graph_static->get_all_distant_out_edges(),
        network_graph_plastic->get_all_local_out_edges(),
        network_graph_plastic->get_all_distant_out_edges(), MPIWrapper::get_my_rank(),
        partition->get_number_mpi_ranks(), partition->get_number_local_neurons(),
        partition->get_total_number_neurons(), ss_out_network, step);

    NeuronIO::write_in_synapses(network_graph_static->get_all_local_in_edges(),
        network_graph_static->get_all_distant_in_edges(),
        network_graph_plastic->get_all_local_in_edges(),
        network_graph_plastic->get_all_distant_in_edges(), MPIWrapper::get_my_rank(),
        partition->get_number_mpi_ranks(), partition->get_number_local_neurons(),
        partition->get_total_number_neurons(), ss_in_network, step);

    LogFiles::write_to_file(LogFiles::EventType::InNetwork, false, ss_in_network.str());
    LogFiles::write_to_file(LogFiles::EventType::OutNetwork, false, ss_out_network.str());
}

void Neurons::print_positions_to_log_file() {
    std::stringstream ss;
    NeuronIO::write_neurons_componentwise(NeuronID::range(number_neurons), extra_info->get_positions(),
        local_area_translator,
        axons->get_signal_types(), ss, partition->get_total_number_neurons(),
        partition->get_simulation_box_size(),
        partition->get_all_local_subdomain_boundaries());
    LogFiles::write_to_file(LogFiles::EventType::Positions, false, ss.str());
}

void Neurons::print_area_mapping_to_log_file() {
    std::stringstream ss;
    NeuronIO::write_area_names(ss, local_area_translator);
    LogFiles::write_to_file(LogFiles::EventType::AreaMapping, false, ss.str());
}

void Neurons::print() {
    const auto& calcium = calcium_calculator->get_calcium();

    // Column widths
    constexpr int cwidth_left = 6;
    constexpr int cwidth = 20;

    std::stringstream ss{};

    // Heading
    LogFiles::write_to_file(LogFiles::EventType::Cout, true,
        "{2:<{1}}{3:<{0}}{4:<{0}}{5:<{0}}{6:<{0}}{7:<{0}}{8:<{0}}{9:<{0}}", cwidth, cwidth_left,
        "gid", "x", "AP", "refractory_time", "C", "A", "D_ex", "D_in");

    // Values
    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        LogFiles::write_to_file(LogFiles::EventType::Cout, true,
            "{3:<{1}}{4:<{0}.{2}f}{5:<{0}}{6:<{0}.{2}f}{7:<{0}.{2}f}{8:<{0}.{2}f}{9:<{0}.{2}f}{10:<{0}.{2}f}",
            cwidth, cwidth_left, Constants::print_precision, local_neuron_id,
            neuron_model->get_x(neuron_id), neuron_model->get_fired(neuron_id),
            neuron_model->get_secondary_variable(neuron_id), calcium[local_neuron_id],
            axons->get_grown_elements(neuron_id),
            dendrites_exc->get_grown_elements(neuron_id),
            dendrites_inh->get_grown_elements(neuron_id));
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, true, ss.str());
}

void Neurons::print_info_for_algorithm() {
    const auto& axons_counts = axons->get_grown_elements();
    const auto& dendrites_exc_counts = dendrites_exc->get_grown_elements();
    const auto& dendrites_inh_counts = dendrites_inh->get_grown_elements();

    const auto& axons_connected_counts = axons->get_connected_elements();
    const auto& dendrites_exc_connected_counts = dendrites_exc->get_connected_elements();
    const auto& dendrites_inh_connected_counts = dendrites_inh->get_connected_elements();

    // Column widths
    const int cwidth_small = 8;
    const int cwidth_medium = 16;
    const int cwidth_big = 27;

    std::stringstream ss{};
    std::string my_string{};

    // Heading
    ss << std::left << std::setw(cwidth_small) << "gid" << std::setw(cwidth_small) << "region"
       << std::setw(cwidth_medium) << "position";
    ss << std::setw(cwidth_big) << "axon (exist|connected)" << std::setw(cwidth_big) << "exc_den (exist|connected)";
    ss << std::setw(cwidth_big) << "inh_den (exist|connected)\n";

    // Values
    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        ss << std::left << std::setw(cwidth_small) << neuron_id;

        const auto [x, y, z] = extra_info->get_position(neuron_id);

        my_string = "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")";
        ss << std::setw(cwidth_medium) << my_string;

        my_string = std::to_string(axons_counts[local_neuron_id]) + "|" + std::to_string(axons_connected_counts[local_neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_exc_counts[local_neuron_id]) + "|" + std::to_string(dendrites_exc_connected_counts[local_neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_inh_counts[local_neuron_id]) + "|" + std::to_string(dendrites_inh_connected_counts[local_neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        ss << '\n';
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, true, ss.str());
}

void Neurons::print_local_network_histogram(const step_type current_step) {
    const auto& out_histogram = axons->get_histogram();
    const auto& in_inhibitory_histogram = dendrites_inh->get_histogram();
    const auto& in_excitatory_histogram = dendrites_exc->get_histogram();

    const auto print_histogram = [current_step](
                                     const std::map<std::pair<unsigned int, unsigned int>, uint64_t>& hist) -> std::string {
        std::stringstream ss{};
        ss << '#' << current_step;
        for (const auto& [val, occurrences] : hist) {
            const auto& [connected, grown] = val;
            ss << ";(" << connected << ',' << grown << "):" << occurrences;
        }

        return ss.str();
    };

    LogFiles::write_to_file(LogFiles::EventType::NetworkOutHistogramLocal, false, print_histogram(out_histogram));
    LogFiles::write_to_file(LogFiles::EventType::NetworkInInhibitoryHistogramLocal, false,
        print_histogram(in_inhibitory_histogram));
    LogFiles::write_to_file(LogFiles::EventType::NetworkInExcitatoryHistogramLocal, false,
        print_histogram(in_excitatory_histogram));
}

void Neurons::print_calcium_values_to_file(const step_type current_step) {
    const auto& calcium = calcium_calculator->get_calcium();

    std::stringstream ss{};

    ss << '#' << current_step;
    for (const auto val : calcium) {
        ss << ';' << val;
    }

    LogFiles::write_to_file(LogFiles::EventType::CalciumValues, false, ss.str());
}

void Neurons::print_synaptic_inputs_to_file(const step_type current_step) {
    const auto& synaptic_input = neuron_model->get_synaptic_input();

    std::stringstream ss{};

    ss << '#' << current_step;
    for (const auto val : synaptic_input) {
        ss << ';' << val;
    }

    LogFiles::write_to_file(LogFiles::EventType::SynapticInput, false, ss.str());
}
