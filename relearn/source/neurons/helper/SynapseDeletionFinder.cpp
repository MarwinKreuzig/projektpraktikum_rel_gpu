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
#include "util/ProbabilityPicker.h"
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

    const auto number_neurons = extra_info->get_size();
    const auto my_rank = MPIWrapper::get_my_rank();
    const auto element_type = synaptic_elements->get_element_type();

    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        if (!extra_info->does_update_plasticity(neuron_id)) {
            continue;
        }

        /**
         * Create and delete synaptic elements as required.
         * This function only deletes elements (bound and unbound), no synapses.
         */
        const auto local_neuron_id = neuron_id.get_neuron_id();
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
                network_graph->add_synapse(PlasticDistantOutSynapse(RankNeuronId(rank, other_neuron_id), neuron_id, weight));
            } else {
                network_graph->add_synapse(PlasticDistantInSynapse(neuron_id, RankNeuronId(rank, other_neuron_id), weight));
            }
        }
    }

    return deletion_requests;
}

std::uint64_t SynapseDeletionFinder::commit_deletions(const CommunicationMap<SynapseDeletionRequest>& deletions, const MPIRank my_rank) {
    auto num_synapses_deleted = std::uint64_t(0);

    for (const auto& [other_rank, requests] : deletions) {
        num_synapses_deleted += requests.size();

        for (const auto& [other_neuron_id, my_neuron_id, element_type, signal_type] : requests) {
            const auto weight = (SignalType::Excitatory == signal_type) ? -1 : 1;

            /**
             *  Update network graph
             */
            if (my_rank == other_rank) {
                if (ElementType::Dendrite == element_type) {
                    network_graph->add_synapse(PlasticLocalSynapse(other_neuron_id, my_neuron_id, weight));
                } else {
                    network_graph->add_synapse(PlasticLocalSynapse(my_neuron_id, other_neuron_id, weight));
                }
            } else {
                if (ElementType::Dendrite == element_type) {
                    network_graph->add_synapse(
                        PlasticDistantOutSynapse(RankNeuronId(other_rank, other_neuron_id), my_neuron_id, weight));
                } else {
                    network_graph->add_synapse(
                        PlasticDistantInSynapse(my_neuron_id, RankNeuronId(other_rank, other_neuron_id), weight));
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

std::vector<RankNeuronId> SynapseDeletionFinder::register_synapses(const NeuronID neuron_id, const ElementType element_type, const SignalType signal_type) {
    auto register_out_edges = [](const auto& distant_out_edges, const auto& local_out_edges) {
        std::vector<RankNeuronId> neuron_ids{};
        neuron_ids.reserve((distant_out_edges.size() + local_out_edges.size()) * 2);

        const auto my_rank = MPIWrapper::get_my_rank();

        for (const auto& [rni, weight] : distant_out_edges) {
            /**
             * Create "edge weight" number of synapses and add them to the synapse list
             * NOTE: We take abs(it->second) here as DendriteType::Inhibitory synapses have count < 0
             */

            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(rni);
            }
        }

        for (const auto& [neuron_id, weight] : local_out_edges) {
            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(my_rank, neuron_id);
            }
        }

        return neuron_ids;
    };

    auto register_in_edges = [signal_type](const auto& distant_in_edges, const auto& local_in_edges) {
        std::vector<RankNeuronId> neuron_ids{};
        neuron_ids.reserve((distant_in_edges.size() + local_in_edges.size()) * 2);

        const auto my_rank = MPIWrapper::get_my_rank();

        for (const auto& [rni, weight] : distant_in_edges) {
            if (weight < 0 && signal_type == SignalType::Excitatory) {
                // Searching excitatory synapses but found an inhibitory one
                continue;
            }

            if (weight > 0 && signal_type == SignalType::Inhibitory) {
                // Searching inhibitory synapses but found an excitatory one
                continue;
            }

            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(rni);
            }
        }

        for (const auto& [neuron_id, weight] : local_in_edges) {
            if (weight < 0 && signal_type == SignalType::Excitatory) {
                // Searching excitatory synapses but found an inhibitory one
                continue;
            }

            if (weight > 0 && signal_type == SignalType::Inhibitory) {
                // Searching inhibitory synapses but found an excitatory one
                continue;
            }

            const auto abs_synapse_weight = std::abs(weight);
            RelearnException::check(abs_synapse_weight > 0,
                "RandomSynapseDeletionFinder::find_synapses_on_neuron: The absolute weight was 0");

            for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                neuron_ids.emplace_back(my_rank, neuron_id);
            }
        }

        return neuron_ids;
    };

    std::vector<RankNeuronId> current_synapses{};
    if (element_type == ElementType::Axon) {
        const auto& [distant_out_edges, _1] = network_graph->get_distant_out_edges(neuron_id);
        const auto& [local_out_edges, _2] = network_graph->get_local_out_edges(neuron_id);

        current_synapses = register_out_edges(distant_out_edges, local_out_edges);
    } else {
        const auto& [distant_in_edges, _1] = network_graph->get_distant_in_edges(neuron_id);
        const auto& [local_in_edges, _2] = network_graph->get_local_in_edges(neuron_id);

        current_synapses = register_in_edges(distant_in_edges, local_in_edges);
    }

    return current_synapses;
}

std::vector<RankNeuronId> RandomSynapseDeletionFinder::find_synapses_on_neuron(const NeuronID neuron_id, const ElementType element_type, const SignalType signal_type, const unsigned int num_synapses_to_delete) {
    // Only do something if necessary
    if (0 == num_synapses_to_delete) {
        return {};
    }

    const auto& current_synapses = register_synapses(neuron_id, element_type, signal_type);
    const auto number_synapses = current_synapses.size();

    RelearnException::check(num_synapses_to_delete <= number_synapses, "RandomSynapseDeletionFinder::find_synapses_on_neuron:: num_synapses_to_delete > current_synapses.size()");

    const auto& drawn_indices = RandomHolder::get_random_uniform_indices(RandomHolderKey::SynapseDeletionFinder, num_synapses_to_delete, number_synapses);

    std::vector<RankNeuronId> affected_neurons{};
    affected_neurons.reserve(num_synapses_to_delete);

    for (const auto index : drawn_indices) {
        affected_neurons.emplace_back(current_synapses[index]);
    }

    return affected_neurons;
}

CommunicationMap<SynapseDeletionRequest> InverseLengthSynapseDeletionFinder::find_synapses_to_delete(const std::shared_ptr<SynapticElements>& synaptic_elements, const std::pair<unsigned int, std::vector<unsigned int>>& to_delete) {
    auto partners = find_partners_to_locate(synaptic_elements, to_delete);
    auto requests = MPIWrapper::exchange_requests(partners);

    auto my_positions = extra_info->get_positions_for(requests);
    auto responses = MPIWrapper::exchange_requests(my_positions);

    this->partners = std::move(partners);
    this->positions = std::move(responses);

    return SynapseDeletionFinder::find_synapses_to_delete(synaptic_elements, to_delete);
}

CommunicationMap<NeuronID> InverseLengthSynapseDeletionFinder::find_partners_to_locate(const std::shared_ptr<SynapticElements>& synaptic_elements, const std::pair<unsigned int, std::vector<unsigned int>>& to_delete) {
    const auto& [sum_to_delete, number_deletions] = to_delete;

    const auto number_ranks = MPIWrapper::get_num_ranks();

    const auto size_hint = std::min(size_t(number_ranks), synaptic_elements->get_size());
    CommunicationMap<NeuronID> partners_to_locate(number_ranks, size_hint);

    if (sum_to_delete == 0) {
        return partners_to_locate;
    }

    const auto number_neurons = extra_info->get_size();
    const auto my_rank = MPIWrapper::get_my_rank();
    const auto element_type = synaptic_elements->get_element_type();

    std::unordered_set<RankNeuronId> partners{};
    partners.reserve(sum_to_delete);

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        if (!extra_info->does_update_plasticity(neuron_id)) {
            continue;
        }

        /**
         * Create and delete synaptic elements as required.
         * This function only deletes elements (bound and unbound), no synapses.
         */
        const auto local_neuron_id = neuron_id.get_neuron_id();
        const auto num_synapses_to_delete = number_deletions[local_neuron_id];
        if (num_synapses_to_delete == 0) {
            continue;
        }

        const auto signal_type = synaptic_elements->get_signal_type(neuron_id);

        if (element_type == ElementType::Axon) {
            const auto& neuron_partners = network_graph->get_all_plastic_partners_outgoing(neuron_id);
            partners.insert(neuron_partners.begin(), neuron_partners.end());
        } else {
            const auto& neuron_partners = network_graph->get_all_plastic_partners_incoming(neuron_id, signal_type);
            partners.insert(neuron_partners.begin(), neuron_partners.end());
        }
    }

    for (const auto& [rank, id] : partners) {
        partners_to_locate.emplace_back(rank, id);
    }

    for (auto& [rank, requests] : partners_to_locate) {
        std::ranges::sort(requests);
    }

    return partners_to_locate;
}

std::vector<RankNeuronId> InverseLengthSynapseDeletionFinder::find_synapses_on_neuron(const NeuronID neuron_id, const ElementType element_type, const SignalType signal_type, const unsigned int num_synapses_to_delete) {
    if (0 == num_synapses_to_delete) {
        return {};
    }

    auto current_synapses = register_synapses(neuron_id, element_type, signal_type);
    const auto number_synapses = current_synapses.size();

    RelearnException::check(num_synapses_to_delete <= number_synapses,
        "RandomSynapseDeletionFinder::find_synapses_on_neuron:: num_synapses_to_delete > current_synapses.size()");

    const auto& my_position = extra_info->get_position(neuron_id);

    auto get_probabilities = [&my_position, &neuron_id, this](const std::vector<RankNeuronId>& others) -> std::vector<double> {
        std::vector<double> probabilities{};
        probabilities.reserve(others.size());

        std::transform(others.begin(), others.end(), std::back_inserter(probabilities), [&my_position, &neuron_id, this](const RankNeuronId& rni) {
            const auto& [other_rank, other_id] = rni;
            if (neuron_id == other_id) {
                // In case a neuron has a synapse to itself, return 1.0
                return 1.0;
            }

            const auto& relevant_ids = partners.get_requests(other_rank);

            const auto pos = std::lower_bound(relevant_ids.begin(), relevant_ids.end(), other_id);
            RelearnException::check(pos != relevant_ids.end(), "InverseLengthSynapseDeletionFinder::find_synapses_on_neuron: Did not find the id {} in the CommunicationMap at rank {}", other_id, other_rank);

            const auto distance = std::distance(relevant_ids.begin(), pos);

            const auto other_pos = positions.get_request(other_rank, distance);

            const auto& diff = other_pos - my_position;
            const auto euclidean_distance = diff.calculate_2_norm();

            return 1.0 / euclidean_distance;
        });

        return probabilities;
    };

    std::vector<RankNeuronId> affected_neurons{};
    affected_neurons.reserve(num_synapses_to_delete);

    auto probabilities = get_probabilities(current_synapses);
    for (auto i = 0U; i < num_synapses_to_delete; i++) {
        const auto idx = ProbabilityPicker::pick_target(probabilities, RandomHolderKey::SynapseDeletionFinder);

        affected_neurons.emplace_back(current_synapses[idx]);
        probabilities[idx] = 0.0;
    }

    return affected_neurons;
}
