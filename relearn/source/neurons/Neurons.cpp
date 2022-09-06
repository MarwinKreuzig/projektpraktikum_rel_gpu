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

#include "NetworkGraph.h"
#include "helper/RankNeuronId.h"
#include "io/LogFiles.h"
#include "models/NeuronModels.h"
#include "mpi/MPIWrapper.h"
#include "structure/NodeCache.h"
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

void Neurons::init(const size_t number_neurons, std::vector<double> target_calcium_values, std::vector<double> initial_calcium_values) {
    RelearnException::check(number_neurons > 0, "Neurons::init: number_neurons was 0");
    RelearnException::check(number_neurons == target_calcium_values.size(), "Neurons::init: number_neurons was different than target_calcium_values.size()");
    RelearnException::check(number_neurons == initial_calcium_values.size(), "Neurons::init: number_neurons was different than initial_calcium_values.size()");

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

    disable_flags.resize(number_neurons, UpdateStatus::Enabled);
    calcium = std::move(initial_calcium_values);
    target_calcium = std::move(target_calcium_values);

    // Init member variables
    for (const auto& id : NeuronID::range(number_neurons)) {
        // Set calcium concentration
        const auto fired = neuron_model->get_fired(id);
        if (fired) {
            calcium[id.get_neuron_id()] += neuron_model->get_beta();
        }
    }
}

void Neurons::init_synaptic_elements() {
    const std::vector<double>& axons_cnts = axons->get_grown_elements();
    const std::vector<double>& dendrites_inh_cnts = dendrites_inh->get_grown_elements();
    const std::vector<double>& dendrites_exc_cnts = dendrites_exc->get_grown_elements();

    for (const auto& id : NeuronID::range(number_neurons)) {
        const size_t axon_connections = network_graph->get_number_out_edges(id);
        const size_t dendrites_ex_connections = network_graph->get_number_excitatory_in_edges(id);
        const size_t dendrites_in_connections = network_graph->get_number_inhibitory_in_edges(id);

        axons->update_grown_elements(id, static_cast<double>(axon_connections));
        dendrites_exc->update_grown_elements(id, static_cast<double>(dendrites_ex_connections));
        dendrites_inh->update_grown_elements(id, static_cast<double>(dendrites_in_connections));

        axons->update_connected_elements(id, static_cast<int>(axon_connections));
        dendrites_exc->update_connected_elements(id, static_cast<int>(dendrites_ex_connections));
        dendrites_inh->update_connected_elements(id, static_cast<int>(dendrites_in_connections));

        const auto local_id = id.get_neuron_id();

        RelearnException::check(axons_cnts[local_id] >= axons->get_connected_elements()[local_id], "Error is with: %d", local_id);
        RelearnException::check(dendrites_inh_cnts[local_id] >= dendrites_inh->get_connected_elements()[local_id], "Error is with: %d", local_id);
        RelearnException::check(dendrites_exc_cnts[local_id] >= dendrites_exc->get_connected_elements()[local_id], "Error is with: %d", local_id);
    }
}

size_t Neurons::disable_neurons(const std::vector<NeuronID>& neuron_ids) {
    neuron_model->disable_neurons(neuron_ids);

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
            network_graph->add_synapse(LocalSynapse(target_neuron_id, neuron_id, -weight));

            bool is_within = std::ranges::binary_search(neuron_ids, target_neuron_id);
            const auto local_target_neuron_id = target_neuron_id.get_neuron_id();

            if (is_within) {
                if (weight > 0) {
                    deleted_dend_ex_connections[local_target_neuron_id] += weight;
                    number_deleted_out_exc_edges_within++;
                    weight_deleted_out_exc_edges_within += weight;
                } else {
                    deleted_dend_in_connections[local_target_neuron_id] -= weight;
                    number_deleted_out_inh_edges_within++;
                    weight_deleted_out_inh_edges_within += std::abs(weight);
                }
            } else {
                if (weight > 0) {
                    deleted_dend_ex_connections[local_target_neuron_id] += weight;
                    number_deleted_out_exc_edges_to_outside++;
                    weight_deleted_out_exc_edges_to_outside += weight;
                } else {
                    deleted_dend_in_connections[local_target_neuron_id] -= weight;
                    number_deleted_out_inh_edges_to_outside++;
                    weight_deleted_out_inh_edges_to_outside += std::abs(weight);
                }
            }
        }
    }

    size_t number_deleted_in_edges_from_outside = 0;
    size_t weight_deleted_in_edges_from_outside = 0;

    for (const auto neuron_id : neuron_ids) {
        RelearnException::check(neuron_id.get_neuron_id() < number_neurons, "Neurons::disable_neurons: There was a too large id: {} vs {}", neuron_id, number_neurons);
        disable_flags[neuron_id.get_neuron_id()] = UpdateStatus::Disabled;

        const auto local_in_edges = network_graph->get_local_in_edges(neuron_id);
        const auto distant_in_edges = network_graph->get_distant_in_edges(neuron_id);
        RelearnException::check(distant_in_edges.empty(), "Neurons::disable_neurons:: Currently, disabling neurons is only supported without mpi");

        for (const auto& [source_neuron_id, weight] : local_in_edges) {
            network_graph->add_synapse(LocalSynapse(neuron_id, source_neuron_id, -weight));

            deleted_axon_connections[source_neuron_id.get_neuron_id()] += std::abs(weight);

            bool is_within = std::ranges::binary_search(neuron_ids, source_neuron_id);

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
    for (const auto& neuron_id : neuron_ids) {
        RelearnException::check(neuron_id.get_neuron_id() < number_neurons, "Neurons::enable_neurons: There was a too large id: {} vs {}", neuron_id, number_neurons);
        disable_flags[neuron_id.get_neuron_id()] = UpdateStatus::Enabled;
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
        dendrites_exc->set_signal_type(id, SignalType::Excitatory);
        dendrites_inh->set_signal_type(id, SignalType::Inhibitory);
    }

    disable_flags.resize(new_size, UpdateStatus::Enabled);

    calcium.insert(calcium.cend(), new_initial_calcium_values.begin(), new_initial_calcium_values.end());
    target_calcium.insert(target_calcium.cend(), new_target_calcium_values.begin(), new_target_calcium_values.end());

    for (size_t i = current_size; i < new_size; i++) {
        // Set calcium concentration
        const auto fired = neuron_model->get_fired(NeuronID{ i });
        if (fired) {
            calcium[i] += neuron_model->get_beta();
        }
    }

    for (size_t i = current_size; i < new_size; i++) {
        auto id = NeuronID{ i };
        const auto& pos = extra_info->get_position(id);
        global_tree->insert(pos, id);
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
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        // Update calcium depending on the firing
        auto c = calcium[neuron_id];
        if (fired[neuron_id] == FiredStatus::Inactive) {
            for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
                c += val * (-c / tau_C);
            }
        } else {
            for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
                c += val * (-c / tau_C + beta);
            }
        }
        calcium[neuron_id] = c;
    }

    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);
}

StatisticalMeasures Neurons::global_statistics(const std::vector<double>& local_values, const int root, const std::vector<UpdateStatus>& disable_flags) const {
    const auto [d_my_min, d_my_max, d_my_acc, d_num_values] = Util::min_max_acc(local_values, disable_flags);
    const double my_avg = d_my_acc / d_num_values;

    const double d_min = MPIWrapper::reduce(d_my_min, MPIWrapper::ReduceFunction::Min, root);
    const double d_max = MPIWrapper::reduce(d_my_max, MPIWrapper::ReduceFunction::Max, root);

    const auto num_values = static_cast<double>(MPIWrapper::all_reduce_uint64(d_num_values, MPIWrapper::ReduceFunction::Sum));

    // Get global avg at all ranks (needed for variance)
    const double avg = MPIWrapper::all_reduce_double(my_avg, MPIWrapper::ReduceFunction::Sum) / MPIWrapper::get_num_ranks();

    /**
     * Calc variance
     */
    double my_var = 0;
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
    const double std = sqrt(var);

    return { d_min, d_max, avg, var, std };
}

std::pair<uint64_t, uint64_t> Neurons::delete_synapses() {
    auto deletion_helper = [this](const std::shared_ptr<SynapticElements>& synaptic_elements) {
        Timers::start(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

        Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
        const auto to_delete = synaptic_elements->commit_updates(disable_flags);
        Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

        Timers::start(TimerRegion::FIND_SYNAPSES_TO_DELETE);
        const auto outgoing_deletion_requests = delete_synapses_find_synapses(*synaptic_elements, to_delete);
        Timers::stop_and_add(TimerRegion::FIND_SYNAPSES_TO_DELETE);

        Timers::start(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);
        const auto incoming_deletion_requests = MPIWrapper::exchange_requests(outgoing_deletion_requests);
        Timers::stop_and_add(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);

        Timers::start(TimerRegion::PROCESS_DELETE_REQUESTS);
        const auto newly_freed_dendrites = delete_synapses_commit_deletions(incoming_deletion_requests);
        Timers::stop_and_add(TimerRegion::PROCESS_DELETE_REQUESTS);

        Timers::stop_and_add(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

        return newly_freed_dendrites;
    };

    const auto axons_deleted = deletion_helper(axons);
    const auto excitatory_dendrites_deleted = deletion_helper(dendrites_exc);
    const auto inhibitory_dendrites_deleted = deletion_helper(dendrites_inh);

    return { axons_deleted, excitatory_dendrites_deleted + inhibitory_dendrites_deleted };
}

CommunicationMap<SynapseDeletionRequest> Neurons::delete_synapses_find_synapses(const SynapticElements& synaptic_elements, const std::pair<unsigned int, std::vector<unsigned int>>& to_delete) {
    const auto& [sum_to_delete, number_deletions] = to_delete;

    const auto number_ranks = MPIWrapper::get_num_ranks();
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto size_hint = std::min(size_t(number_ranks), synaptic_elements.get_size());
    CommunicationMap<SynapseDeletionRequest> deletion_requests(number_ranks, size_hint);

    if (sum_to_delete == 0) {
        return deletion_requests;
    }

    const auto element_type = synaptic_elements.get_element_type();

    for (size_t neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
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
        const auto affected_neuron_ids = delete_synapses_find_synapses_on_neuron(id, element_type, signal_type, num_synapses_to_delete);

        for (const auto& [rank, other_neuron_id] : affected_neuron_ids) {
            SynapseDeletionRequest psd(id, other_neuron_id, element_type, signal_type);
            deletion_requests.append(rank, psd);

            if (my_rank == rank) {
                continue;
            }

            const auto weight = (SignalType::Excitatory == signal_type) ? -1 : 1;
            if (ElementType::Axon == element_type) {
                network_graph->add_synapse(DistantOutSynapse(RankNeuronId(rank, other_neuron_id), id, weight));
            } else {
                network_graph->add_synapse(DistantInSynapse(id, RankNeuronId(rank, other_neuron_id), weight));
            }
        }
    }

    return deletion_requests;
}

std::vector<RankNeuronId> Neurons::delete_synapses_find_synapses_on_neuron(
    NeuronID neuron_id,
    ElementType element_type,
    SignalType signal_type,
    unsigned int num_synapses_to_delete) {

    // Only do something if necessary
    if (0 == num_synapses_to_delete) {
        return {};
    }

    auto register_edges = [](const std::vector<std::pair<RankNeuronId, int>>& edges) {
        std::vector<RankNeuronId> neuron_ids{};
        neuron_ids.reserve(edges.size());

        for (const auto& [rni, weight] : edges) {
            /**
             * Create "edge weight" number of synapses and add them to the synapse list
             * NOTE: We take abs(it->second) here as DendriteType::Inhibitory synapses have count < 0
             */

            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0, "Neurons::delete_synapses_find_synapses_on_neuron::delete_synapses_register_edges: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(rni);
            }
        }

        return neuron_ids;
    };

    std::vector<RankNeuronId> current_synapses{};
    if (element_type == ElementType::Axon) {
        // Walk through outgoing edges
        NetworkGraph::DistantEdges out_edges = network_graph->get_all_out_edges(neuron_id);
        current_synapses = register_edges(out_edges);
    } else {
        // Walk through ingoing edges
        NetworkGraph::DistantEdges in_edges = network_graph->get_all_in_edges(neuron_id, signal_type);
        current_synapses = register_edges(in_edges);
    }

    const auto number_synapses = current_synapses.size();

    RelearnException::check(num_synapses_to_delete <= number_synapses, "Neurons::delete_synapses_find_synapses_on_neuron:: num_synapses_to_delete > current_synapses.size()");

    std::vector<size_t> drawn_indices{};
    drawn_indices.reserve(num_synapses_to_delete);

    uniform_int_distribution<unsigned int> uid{};

    for (unsigned int i = 0; i < num_synapses_to_delete; i++) {
        auto random_number = RandomHolder::get_random_uniform_integer(RandomHolderKey::Neurons, size_t(0), number_synapses - 1);
        while (std::ranges::find(drawn_indices, random_number) != drawn_indices.end()) {
            random_number = RandomHolder::get_random_uniform_integer(RandomHolderKey::Neurons, size_t(0), number_synapses - 1);
        }

        drawn_indices.emplace_back(random_number);
    }

    std::vector<RankNeuronId> affected_neurons{};
    affected_neurons.reserve(num_synapses_to_delete);

    for (const auto index : drawn_indices) {
        affected_neurons.emplace_back(current_synapses[index]);
    }

    return affected_neurons;
}

size_t Neurons::delete_synapses_commit_deletions(const CommunicationMap<SynapseDeletionRequest>& list) {
    const int my_rank = MPIWrapper::get_my_rank();
    size_t num_synapses_deleted = 0;

    for (const auto& [other_rank, requests] : list) {
        num_synapses_deleted += requests.size();

        for (const auto& [other_neuron_id, my_neuron_id, element_type, signal_type] : requests) {
            const auto weight = (SignalType::Excitatory == signal_type) ? -1 : 1;

            /**
             *  Update network graph
             */
            if (my_rank == other_rank) {
                if (ElementType::Dendrite == element_type) {
                    network_graph->add_synapse(LocalSynapse(other_neuron_id, my_neuron_id, weight));
                } else {
                    network_graph->add_synapse(LocalSynapse(my_neuron_id, other_neuron_id, weight));
                }
            } else {
                if (ElementType::Dendrite == element_type) {
                    network_graph->add_synapse(DistantOutSynapse(RankNeuronId(other_rank, other_neuron_id), my_neuron_id, weight));
                    // network_graph->add_edge_weight(RankNeuronId(other_rank, other_neuron_id), RankNeuronId(my_rank, my_neuron_id), weight);
                } else {
                    network_graph->add_synapse(DistantInSynapse(my_neuron_id, RankNeuronId(other_rank, other_neuron_id), weight));
                    // network_graph->add_edge_weight(RankNeuronId(my_rank, my_neuron_id), RankNeuronId(other_rank, other_neuron_id), weight);
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

size_t Neurons::create_synapses() {
    const auto my_rank = MPIWrapper::get_my_rank();

    // Lock local RMA memory for local stores
    MPIWrapper::lock_window(my_rank, MPI_Locktype::Exclusive);

    // Update my leaf nodes
    Timers::start(TimerRegion::UPDATE_LEAF_NODES);
    algorithm->update_leaf_nodes(disable_flags);
    Timers::stop_and_add(TimerRegion::UPDATE_LEAF_NODES);

    // Update the octree
    global_tree->synchronize_tree();

    // Unlock local RMA memory and make local stores visible in public window copy
    MPIWrapper::unlock_window(my_rank);

    // Makes sure that all ranks finished their local access epoch
    // before a remote origin opens an access epoch
    MPIWrapper::barrier();

    // Delegate the creation of new synapses to the algorithm
    const auto& [local_synapses, distant_in_synapses, distant_out_synapses]
        = algorithm->update_connectivity(number_neurons, disable_flags, extra_info);

    // Update the network graph all at once
    Timers::start(TimerRegion::ADD_SYNAPSES_TO_NETWORKGRAPH);
    network_graph->add_edges(local_synapses, distant_in_synapses, distant_out_synapses);
    Timers::stop_and_add(TimerRegion::ADD_SYNAPSES_TO_NETWORKGRAPH);

    // The distant_out_synapses are counted on the ranks where they are in
    const auto num_synapses_created = local_synapses.size() + distant_in_synapses.size();

    return num_synapses_created;
}

void Neurons::debug_check_counts() {
    if (!Config::do_debug_checks) {
        return;
    }

    RelearnException::check(network_graph != nullptr, "Neurons::debug_check_counts: network_graph is nullptr");

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& grown_axons = axons->get_grown_elements();
    const auto& connected_axons = axons->get_connected_elements();
    const auto& grown_excitatory_dendrites = dendrites_exc->get_grown_elements();
    const auto& connected_excitatory_dendrites = dendrites_exc->get_connected_elements();
    const auto& grown_inhibitory_dendrites = dendrites_inh->get_grown_elements();
    const auto& connected_inhibitory_dendrites = dendrites_inh->get_connected_elements();

    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        const auto vacant_axons = grown_axons[neuron_id] - connected_axons[neuron_id];
        const auto vacant_excitatory_dendrites = grown_excitatory_dendrites[neuron_id] - connected_excitatory_dendrites[neuron_id];
        const auto vacant_inhibitory_dendrites = grown_inhibitory_dendrites[neuron_id] - connected_inhibitory_dendrites[neuron_id];

        RelearnException::check(vacant_axons >= 0.0, "Neurons::debug_check_counts: {} has a weird number of vacant axons: {}", neuron_id, vacant_axons);
        RelearnException::check(vacant_excitatory_dendrites >= 0.0, "Neurons::debug_check_counts: {} has a weird number of vacant excitatory dendrites: {}", neuron_id, vacant_excitatory_dendrites);
        RelearnException::check(vacant_inhibitory_dendrites >= 0.0, "Neurons::debug_check_counts: {} has a weird number of vacant inhibitory dendrites: {}", neuron_id, vacant_inhibitory_dendrites);
    }

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        const double connected_axons_neuron = connected_axons[local_neuron_id];
        const double connected_excitatory_dendrites_neuron = connected_excitatory_dendrites[local_neuron_id];
        const double connected_inhibitory_dendrites_neuron = connected_inhibitory_dendrites[local_neuron_id];

        const auto number_connected_axons = static_cast<size_t>(connected_axons_neuron);
        const auto number_connected_excitatory_dendrites = static_cast<size_t>(connected_excitatory_dendrites_neuron);
        const auto number_connected_inhibitory_dendrites = static_cast<size_t>(connected_inhibitory_dendrites_neuron);

        const size_t number_out_edges = network_graph->get_number_out_edges(neuron_id);
        const size_t number_excitatory_in_edges = network_graph->get_number_excitatory_in_edges(neuron_id);
        const size_t number_inhibitory_in_edges = network_graph->get_number_inhibitory_in_edges(neuron_id);

        RelearnException::check(number_connected_axons == number_out_edges,
            "Neurons::debug_check_counts: Neuron {} has {} axons but {} out edges (rank {})", neuron_id, number_connected_axons, number_out_edges, my_rank);

        RelearnException::check(number_connected_excitatory_dendrites == number_excitatory_in_edges,
            "Neurons::debug_check_counts: Neuron {} has {} excitatory dendrites but {} excitatory in edges (rank {})", neuron_id, number_connected_excitatory_dendrites, number_excitatory_in_edges, my_rank);

        RelearnException::check(number_connected_inhibitory_dendrites == number_inhibitory_in_edges,
            "Neurons::debug_check_counts: Neuron {} has {} inhibitory dendrites but {} inhibitory in edges (rank {})", neuron_id, number_connected_inhibitory_dendrites, number_inhibitory_in_edges, my_rank);
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

    case NeuronAttribute::SynapticInput:
        return global_statistics(neuron_model->get_synaptic_input(), 0, disable_flags);

    case NeuronAttribute::Background:
        return global_statistics(neuron_model->get_background_activity(), 0, disable_flags);

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

    RelearnException::fail("Neurons::get_statistics: Got an unsupported attribute: {}", static_cast<int>(attribute));

    return {};
}

std::tuple<uint64_t, uint64_t, uint64_t> Neurons::update_connectivity() {
    RelearnException::check(network_graph != nullptr, "Network graph is nullptr");
    RelearnException::check(global_tree != nullptr, "Global octree is nullptr");
    RelearnException::check(algorithm != nullptr, "Algorithm is nullptr");

    debug_check_counts();
    network_graph->debug_check();
    const auto& [num_axons_deleted, num_dendrites_deleted] = delete_synapses();
    debug_check_counts();
    network_graph->debug_check();
    size_t num_synapses_created = create_synapses();
    debug_check_counts();
    network_graph->debug_check();

    return { num_axons_deleted, num_dendrites_deleted, num_synapses_created };
}

void Neurons::print_sums_of_synapses_and_elements_to_log_file_on_rank_0(size_t step, uint64_t sum_axon_deleted, uint64_t sum_dendrites_deleted, uint64_t sum_synapses_created) {
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

    std::array<int64_t, 7> sums_global = MPIWrapper::reduce(sums_local, MPIWrapper::ReduceFunction::Sum, 0);

    // Output data
    if (0 == MPIWrapper::get_my_rank()) {
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

void Neurons::print_neurons_overview_to_log_file_on_rank_0(const size_t step) const {
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
    std::stringstream ss_in_network{};

    ss_in_network << "# Total number neurons: " << partition->get_total_number_neurons() << "\n";
    ss_in_network << "# Local number neurons: " << partition->get_number_local_neurons() << "\n";
    ss_in_network << "# Number MPI ranks: " << partition->get_number_mpi_ranks() << "\n";
    ss_in_network << "# <target_rank> <target_id>\t<source_rank> <source_id>\t<weight> \n";

    std::stringstream ss_out_network{};

    ss_out_network << "# Total number neurons: " << partition->get_total_number_neurons() << "\n";
    ss_out_network << "# Local number neurons: " << partition->get_number_local_neurons() << "\n";
    ss_out_network << "# Number MPI ranks: " << partition->get_number_mpi_ranks() << "\n";
    ss_out_network << "# <target_rank> <target_id>\t<source_rank> <source_id>\t<weight> \n";

    network_graph->print_with_ranks(ss_out_network, ss_in_network);

    LogFiles::write_to_file(LogFiles::EventType::InNetwork, false, ss_in_network.str());
    LogFiles::write_to_file(LogFiles::EventType::OutNetwork, false, ss_out_network.str());
}

void Neurons::print_positions_to_log_file() {
    const auto& total_number_neurons = partition->get_total_number_neurons();

    const auto& [simulation_box_min, simulation_box_max] = partition->get_simulation_box_size();
    const auto& [min_x, min_y, min_z] = simulation_box_min;
    const auto& [max_x, max_y, max_z] = simulation_box_max;

    // Write total number of neurons to log file
    LogFiles::write_to_file(LogFiles::EventType::Positions, false, "# {} of {}", number_neurons, total_number_neurons);
    LogFiles::write_to_file(LogFiles::EventType::Positions, false, "# Minimum x: {}", min_x);
    LogFiles::write_to_file(LogFiles::EventType::Positions, false, "# Minimum y: {}", min_y);
    LogFiles::write_to_file(LogFiles::EventType::Positions, false, "# Minimum z: {}", min_z);
    LogFiles::write_to_file(LogFiles::EventType::Positions, false, "# Maximum x: {}", max_x);
    LogFiles::write_to_file(LogFiles::EventType::Positions, false, "# Maximum y: {}", max_y);
    LogFiles::write_to_file(LogFiles::EventType::Positions, false, "# Maximum z: {}", max_z);
    LogFiles::write_to_file(LogFiles::EventType::Positions, false, "# <local id> <pos x> <pos y> <pos z> <area> <type>");

    const auto& positions = extra_info->get_positions();
    const auto& area_names = extra_info->get_area_names();
    const auto& signal_types = axons->get_signal_types();

    RelearnException::check(positions.size() == number_neurons,
        "Neurons::print_positions_to_log_file: positions had size {}, but there were {} local neurons.", positions.size(), number_neurons);
    RelearnException::check(area_names.size() == number_neurons,
        "Neurons::print_positions_to_log_file: area_names had size {}, but there were {} local neurons.", area_names.size(), number_neurons);
    RelearnException::check(signal_types.size() == number_neurons,
        "Neurons::print_positions_to_log_file: signal_types had size {}, but there were {} local neurons.", signal_types.size(), number_neurons);

    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto& local_neuron_id = neuron_id.get_neuron_id();

        const auto& [x, y, z] = positions[local_neuron_id];
        const auto& signal_type_name = (signal_types[local_neuron_id] == SignalType::Excitatory) ? "ex" : "in";
        const auto& area_name = area_names[local_neuron_id];

        LogFiles::write_to_file(LogFiles::EventType::Positions, false,
            "{1:<} {2:<.{0}} {3:<.{0}} {4:<.{0}} {5:<} {6:<}",
            Constants::print_precision, (local_neuron_id + 1), x, y, z, area_name, signal_type_name);
    }
}

void Neurons::print() {
    // Column widths
    const int cwidth_left = 6;
    const int cwidth = 20;

    std::stringstream ss{};

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

    std::stringstream ss{};
    std::string my_string{};

    // Heading
    ss << std::left << std::setw(cwidth_small) << "gid" << std::setw(cwidth_small) << "region" << std::setw(cwidth_medium) << "position";
    ss << std::setw(cwidth_big) << "axon (exist|connected)" << std::setw(cwidth_big) << "exc_den (exist|connected)";
    ss << std::setw(cwidth_big) << "inh_den (exist|connected)\n";

    // Values
    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        ss << std::left << std::setw(cwidth_small) << neuron_id;

        const auto [x, y, z] = extra_info->get_position(neuron_id);

        my_string = "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")";
        ss << std::setw(cwidth_medium) << my_string;

        my_string = std::to_string(axons_cnts[local_neuron_id]) + "|" + std::to_string(axons_connected_cnts[local_neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_exc_cnts[local_neuron_id]) + "|" + std::to_string(dendrites_exc_connected_cnts[local_neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        my_string = std::to_string(dendrites_inh_cnts[local_neuron_id]) + "|" + std::to_string(dendrites_inh_connected_cnts[local_neuron_id]);
        ss << std::setw(cwidth_big) << my_string;

        ss << "\n";
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, true, ss.str());
}

void Neurons::print_local_network_histogram(const size_t current_step) {
    const auto& out_histogram = axons->get_historgram();
    const auto& in_inhibitory_histogram = dendrites_inh->get_historgram();
    const auto& in_excitatory_histogram = dendrites_exc->get_historgram();

    const auto print_histogram = [current_step](const std::map<std::pair<unsigned int, unsigned int>, uint64_t>& hist) -> std::string {
        std::stringstream ss{};
        ss << '#' << current_step;
        for (const auto& [val, occurences] : hist) {
            const auto& [connected, grown] = val;
            ss << ";(" << connected << ',' << grown << "):" << occurences;
        }

        return ss.str();
    };

    LogFiles::write_to_file(LogFiles::EventType::NetworkOutHistogramLocal, false, print_histogram(out_histogram));
    LogFiles::write_to_file(LogFiles::EventType::NetworkInInhibitoryHistogramLocal, false, print_histogram(in_inhibitory_histogram));
    LogFiles::write_to_file(LogFiles::EventType::NetworkInExcitatoryHistogramLocal, false, print_histogram(in_excitatory_histogram));
}

void Neurons::print_calcium_values_to_file(const size_t current_step) {
    std::stringstream ss{};

    ss << '#' << current_step;
    for (auto val : calcium) {
        ss << ';' << val;
    }

    LogFiles::write_to_file(LogFiles::EventType::CalciumValues, false, ss.str());
}
