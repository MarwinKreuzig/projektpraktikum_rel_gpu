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

#include "adapter/random/RandomAdapter.h"
#include "adapter/tagged_id/TaggedIdAdapter.h"

#include "Types.h"
#include "neurons/NetworkGraph.h"
#include "util/TaggedID.h"

#include <map>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

class NetworkGraphAdapter {
public:
    constexpr static double bound_synapse_weight = 10.0;
    constexpr static int upper_bound_num_synapses = 1000;

    static size_t get_random_number_synapses(std::mt19937& mt) {
        return RandomAdapter::get_random_integer<size_t>(1, upper_bound_num_synapses, mt);
    }

    static RelearnTypes::synapse_weight get_random_synapse_weight(std::mt19937& mt) {
        RelearnTypes::synapse_weight weight = RandomAdapter::get_random_double<RelearnTypes::synapse_weight>(-bound_synapse_weight, bound_synapse_weight, mt);

        while (weight == 0) {
            weight = RandomAdapter::get_random_double<RelearnTypes::synapse_weight>(-bound_synapse_weight, bound_synapse_weight, mt);
        }

        return weight;
    }

    static std::vector<std::tuple<NeuronID, NeuronID, RelearnTypes::synapse_weight>> get_random_synapses(size_t number_neurons, size_t number_synapses, std::mt19937& mt) {
        std::vector<std::tuple<NeuronID, NeuronID, RelearnTypes::synapse_weight>> synapses(number_synapses);

        for (auto i = 0; i < number_synapses; i++) {
            const auto source_id = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);
            const auto target_id = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);
            const auto weight = get_random_synapse_weight(mt);

            synapses[i] = { source_id, target_id, weight };
        }

        return synapses;
    }

    static std::vector<LocalSynapse> generate_local_synapses(size_t number_neurons, std::mt19937& mt) {
        const auto number_synapses = get_random_number_synapses(mt);

        std::map<std::pair<NeuronID, NeuronID>, RelearnTypes::synapse_weight> synapse_map{};
        for (auto i = 0; i < number_synapses; i++) {
            const auto source = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);
            const auto target = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);
            const auto weight = get_random_synapse_weight(mt);

            synapse_map[{ target, source }] += weight;
        }

        std::vector<LocalSynapse> synapses{};
        synapses.reserve(synapse_map.size());

        for (const auto& [pair, weight] : synapse_map) {
            const auto& [target, source] = pair;
            if (weight != 0) {
                synapses.emplace_back(target, source, weight);
            }
        }

        return synapses;
    }

    static std::shared_ptr<NetworkGraph> create_network_graph_all_to_all(size_t number_neurons, MPIRank mpi_rank, std::mt19937& mt) {
        auto ptr = std::make_shared<NetworkGraph>(number_neurons, mpi_rank);

        for (const auto& source_id : NeuronID::range(number_neurons)) {
            for (const auto& target_id : NeuronID::range(number_neurons)) {
                if (source_id.get_neuron_id() == target_id.get_neuron_id()) {
                    continue;
                }

                const auto weight = get_random_synapse_weight(mt);
                LocalSynapse ls(target_id, source_id, weight);

                ptr->add_synapse(ls);
            }
        }

        return ptr;
    }

    static std::shared_ptr<NetworkGraph> create_network_graph(size_t number_neurons, MPIRank mpi_rank, unsigned long long number_connections_per_vertex, std::mt19937& mt) {
        auto ptr = std::make_shared<NetworkGraph>(number_neurons, mpi_rank);

        for (auto i = 0ULL; i < number_connections_per_vertex; i++) {
            const auto& source_ids = NeuronID::range(number_neurons);
            const auto& target_ids = RandomAdapter::get_random_derangement(number_neurons, mt);

            for (auto j = 0; j < number_neurons; j++) {

                const auto weight = get_random_synapse_weight(mt);
                LocalSynapse ls(NeuronID(false, target_ids[j]), source_ids[j], weight);
                ptr->add_synapse(ls);
            }
        }

        return ptr;
    }

    static std::shared_ptr<NetworkGraph> create_empty_network_graph(size_t number_neurons, MPIRank mpi_rank) {
        auto ptr = std::make_shared<NetworkGraph>(number_neurons, mpi_rank);
        return ptr;
    }
};
