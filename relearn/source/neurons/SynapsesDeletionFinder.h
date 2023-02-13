#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/helper/RankNeuronId.h"
#include "neurons/enums/ElementType.h"
#include "neurons/enums/SignalType.h"
#include "neurons/NetworkGraph.h"
#include "neurons/models/NeuronModels.h"
#include "util/Random.h"
#include "Types.h"

#include <vector>

/**
 * Searches a synapse for deletion for a single neuron.
 */
class SynapseDeletionFinder {
public:

    /**
     *
     * @param neuron_id Initiater neuron
     * @param element_type Element type of the initiater neuron
     * @param signal_type Signal type of the synapse
     * @param num_synapses_to_delete Number of synapses that shall be deleted from this neuron
     * @param network_graph_plastic The plastic network graph
     * @param neuron_model The neuron model
     * @return Vector of affected neurons
     */
    virtual std::vector<RankNeuronId> find_for_neuron(const NeuronID neuron_id,
                         const ElementType element_type,
                         const SignalType signal_type,
                         const unsigned int num_synapses_to_delete,
                                                      const std::shared_ptr<NetworkGraph>& network_graph_plastic,
                                                      const std::shared_ptr<NeuronModel>& neuron_model)  = 0;
};

/**
 * Deletes synapses uniformly at random
 */
class RandomSynapseDeletionFinder : public SynapseDeletionFinder {
public:
    std::vector<RankNeuronId> find_for_neuron(const NeuronID neuron_id,
                                              const ElementType element_type,
                                              const SignalType signal_type,
                                              const unsigned int num_synapses_to_delete,
                                              const std::shared_ptr<NetworkGraph>& network_graph_plastic,
                                              const std::shared_ptr<NeuronModel>& neuron_model) override {
        // Only do something if necessary
        if (0 == num_synapses_to_delete) {
            return {};
        }

        auto register_edges = [](const std::vector<std::pair<RankNeuronId, RelearnTypes::synapse_weight>> &edges) {
            std::vector<RankNeuronId> neuron_ids{};
            neuron_ids.reserve(edges.size());

            for (const auto &[rni, weight]: edges) {
                /**
                 * Create "edge weight" number of synapses and add them to the synapse list
                 * NOTE: We take abs(it->second) here as DendriteType::Inhibitory synapses have count < 0
                 */

                const auto abs_synapse_weight = std::abs(weight);
                RelearnException::check(abs_synapse_weight > 0,
                                        "Neurons::delete_synapses_find_synapses_on_neuron::delete_synapses_register_edges: The absolute weight was 0");

                for (auto synapse_id = 0; synapse_id < abs_synapse_weight; ++synapse_id) {
                    neuron_ids.emplace_back(rni);
                }
            }

            return neuron_ids;
        };

        std::vector<RankNeuronId> current_synapses{};
        if (element_type == ElementType::Axon) {
            // Walk through outgoing edges
            NetworkGraph::DistantEdges out_edges = network_graph_plastic->get_all_out_edges(neuron_id);
            current_synapses = register_edges(out_edges);
        } else {
            // Walk through ingoing edges
            NetworkGraph::DistantEdges in_edges = network_graph_plastic->get_all_in_edges(neuron_id, signal_type);
            current_synapses = register_edges(in_edges);
        }

        const auto number_synapses = current_synapses.size();

        RelearnException::check(num_synapses_to_delete <= number_synapses,
                                "Neurons::delete_synapses_find_synapses_on_neuron:: num_synapses_to_delete > current_synapses.size()");

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

        for (const auto index: drawn_indices) {
            affected_neurons.emplace_back(current_synapses[index]);
        }

        return affected_neurons;
    }
};

/**
 * Deletes neurons based on how often they spike together
 */
class CoActivationSynapseDeletionFinder : public SynapseDeletionFinder {
public:
    std::vector<RankNeuronId> find_for_neuron(const NeuronID neuron_id,
                                              const ElementType element_type,
                                              const SignalType signal_type,
                                              const unsigned int num_synapses_to_delete,
                                              const std::shared_ptr<NetworkGraph>& network_graph_plastic,
                                              const std::shared_ptr<NeuronModel>& neuron_model) override {
        // Only do something if necessary
        if (0 == num_synapses_to_delete) {
            return {};
        }

        NetworkGraph::DistantEdges current_synapses{};
        if (element_type == ElementType::Axon) {
            // Walk through outgoing edges
            current_synapses = network_graph_plastic->get_all_out_edges(neuron_id);
        } else {
            // Walk through ingoing edges
            current_synapses = network_graph_plastic->get_all_in_edges(neuron_id, signal_type);
        }

        const auto number_synapses = current_synapses.size();

        RelearnException::check(num_synapses_to_delete <= number_synapses,
                                "Neurons::delete_synapses_find_synapses_on_neuron:: num_synapses_to_delete > current_synapses.size()");

        RandomHolder::shuffle(RandomHolderKey::Neurons, current_synapses.begin(), current_synapses.end());

        std::vector<std::pair<RankNeuronId, double>> co_activations{};
        co_activations.reserve(number_synapses);
        for(const auto&[rank_neuron_id, weight] : current_synapses) {
            double co_activation;
            if(element_type == ElementType::Axon) {
                co_activation = calculate_co_activation(neuron_model->get_fire_history(neuron_id),
                                        neuron_model->get_fire_history(rank_neuron_id.get_neuron_id()));
            }
            else {
                co_activation = calculate_co_activation(neuron_model->get_fire_history(rank_neuron_id.get_neuron_id()),
                                                        neuron_model->get_fire_history(neuron_id));
            }
            for(auto i=0;i<std::abs(weight);i++) {
                co_activations.emplace_back(rank_neuron_id, co_activation);
            }
        }

        std::sort(co_activations.begin(), co_activations.end(), [](const auto& p1, const auto& p2) {return p1.second<p2.second;});

        std::vector<RankNeuronId> affected_neurons{};
        affected_neurons.reserve(num_synapses_to_delete);

        for (auto i=0;i<num_synapses_to_delete;i++) {
            affected_neurons.push_back(co_activations[i].first);
        }

        return affected_neurons;
    }
private:
    double calculate_co_activation(const boost::circular_buffer<FiredStatus>& pre_synaptic, const boost::circular_buffer<FiredStatus>& post_synaptic) {
        RelearnException::check(pre_synaptic.size() == post_synaptic.size(), "SynapseDeletionFinder::calculate_co_activation: Fire histories have different sizes");

        auto intersection=0U;
        for(auto i=0;i<pre_synaptic.size();i++) {
            if(FiredStatus::Fired == pre_synaptic[i] && pre_synaptic[i] == post_synaptic[i]) {
                intersection++;
            }
        }
        return static_cast<double>(intersection)/static_cast<double>(pre_synaptic.size());
    }
};