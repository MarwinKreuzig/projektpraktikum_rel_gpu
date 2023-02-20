/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapseDeletionFinder.h"

#include "Types.h"
#include "mpi/MPIWrapper.h"
#include "neurons/models/SynapticElements.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/NetworkGraph.h"
#include "util/Random.h"
#include "util/Timers.h"

#include <numeric>
#include <utility>

std::pair<uint64_t, uint64_t> SynapseDeletionFinder::delete_synapses() {
    auto deletion_helper = [this](const std::shared_ptr<SynapticElements>& synaptic_elements) {
        Timers::start(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

        Timers::start(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
        const auto to_delete = synaptic_elements->commit_updates();
        Timers::stop_and_add(TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);

        Timers::start(TimerRegion::FIND_SYNAPSES_TO_DELETE);
        const auto outgoing_deletion_requests = find_synapses_to_delete(synaptic_elements, to_delete);
        Timers::stop_and_add(TimerRegion::FIND_SYNAPSES_TO_DELETE);

        Timers::start(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);
        const auto incoming_deletion_requests = MPIWrapper::exchange_requests(outgoing_deletion_requests);
        Timers::stop_and_add(TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);

        Timers::start(TimerRegion::PROCESS_DELETE_REQUESTS);
        const auto newly_freed_dendrites = commit_deletions(incoming_deletion_requests, MPIWrapper::get_my_rank());
        Timers::stop_and_add(TimerRegion::PROCESS_DELETE_REQUESTS);

        Timers::stop_and_add(TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);

        return newly_freed_dendrites;
    };

    const auto axons_deleted = deletion_helper(axons);
    const auto excitatory_dendrites_deleted = deletion_helper(excitatory_dendrites);
    const auto inhibitory_dendrites_deleted = deletion_helper(inhibitory_dendrites);

    return { axons_deleted, excitatory_dendrites_deleted + inhibitory_dendrites_deleted };
}

CommunicationMap<SynapseDeletionRequest> SynapseDeletionFinder::find_synapses_to_delete(const std::shared_ptr<SynapticElements>& synaptic_elements, const std::pair<unsigned int, std::vector<unsigned int>>& to_delete) {
    const auto& [sum_to_delete, number_deletions] = to_delete;

    const auto number_ranks = MPIWrapper::get_num_ranks();

    const auto size_hint = std::min(size_t(number_ranks), synaptic_elements->get_size());
    CommunicationMap<SynapseDeletionRequest> deletion_requests(number_ranks, size_hint);

    if (sum_to_delete == 0) {
        return deletion_requests;
    }

    Timers::start(TimerRegion::FIND_SYNAPSES_TO_DELETE);

    const auto number_neurons = extra_info->get_size();
    const auto my_rank = MPIWrapper::get_my_rank();
    const auto element_type = synaptic_elements->get_element_type();

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

        const auto signal_type = synaptic_elements->get_signal_type(neuron_id);
        const auto affected_neuron_ids = find_synapses_on_neuron(neuron_id, element_type, signal_type, num_synapses_to_delete);

        for (const auto& [rank, other_neuron_id] : affected_neuron_ids) {
            SynapseDeletionRequest psd(neuron_id, other_neuron_id, element_type, signal_type);
            deletion_requests.append(rank, psd);

            if (my_rank == rank) {
                continue;
            }

            const auto weight = (SignalType::Excitatory == signal_type) ? -1 : 1;
            if (ElementType::Axon == element_type) {
                network_graph->add_synapse(DistantOutSynapse(RankNeuronId(rank, other_neuron_id), neuron_id, weight));
            } else {
                network_graph->add_synapse(DistantInSynapse(neuron_id, RankNeuronId(rank, other_neuron_id), weight));
            }
        }
    }

    Timers::stop_and_add(TimerRegion::FIND_SYNAPSES_TO_DELETE);

    return deletion_requests;
}

std::uint64_t SynapseDeletionFinder::commit_deletions(const CommunicationMap<SynapseDeletionRequest>& list, MPIRank my_rank) {
    auto num_synapses_deleted = std::uint64_t(0);

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
                    network_graph->add_synapse(
                        DistantOutSynapse(RankNeuronId(other_rank, other_neuron_id), my_neuron_id, weight));
                } else {
                    network_graph->add_synapse(
                        DistantInSynapse(my_neuron_id, RankNeuronId(other_rank, other_neuron_id), weight));
                }
            }

            if (ElementType::Dendrite == element_type) {
                axons->update_connected_elements(my_neuron_id, -1);
                continue;
            }

            if (SignalType::Excitatory == signal_type) {
                excitatory_dendrites->update_connected_elements(my_neuron_id, -1);
            } else {
                inhibitory_dendrites->update_connected_elements(my_neuron_id, -1);
            }
        }
    }

    return num_synapses_deleted;
}

std::vector<RankNeuronId> RandomSynapseDeletionFinder::find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, unsigned int num_synapses_to_delete) {
    // Only do something if necessary
    if (0 == num_synapses_to_delete) {
        return {};
    }

    auto register_edges = [](const std::vector<std::pair<RankNeuronId, RelearnTypes::synapse_weight>>& edges) {
        std::vector<RankNeuronId> neuron_ids{};
        neuron_ids.reserve(edges.size());

        for (const auto& [rni, weight] : edges) {
            /**
             * Create "edge weight" number of synapses and add them to the synapse list
             * NOTE: We take abs(it->second) here as DendriteType::Inhibitory synapses have count < 0
             */

            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                "RandomSynapseDeletionFinder::find_synapses_on_neuron:: The absolute weight was 0");

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

    RelearnException::check(num_synapses_to_delete <= number_synapses,
        "RandomSynapseDeletionFinder::find_synapses_on_neuron:: num_synapses_to_delete > current_synapses.size()");

    std::vector<size_t> drawn_indices{};
    drawn_indices.reserve(num_synapses_to_delete);

    uniform_int_distribution<unsigned int> uid{};

    for (unsigned int i = 0; i < num_synapses_to_delete; i++) {
        auto random_number = RandomHolder::get_random_uniform_integer(RandomHolderKey::Neurons, size_t(0),
            number_synapses - 1);
        while (std::ranges::find(drawn_indices, random_number) != drawn_indices.end()) {
            random_number = RandomHolder::get_random_uniform_integer(RandomHolderKey::Neurons, size_t(0),
                number_synapses - 1);
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
