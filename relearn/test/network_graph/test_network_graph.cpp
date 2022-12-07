/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_network_graph.h"

#include "neurons/NetworkGraph.h"

#include <cstddef>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

int NetworkGraphTest::num_ranks = 17;
int NetworkGraphTest::num_synapses_per_neuron = 2;

TEST_F(NetworkGraphTest, testNetworkGraphConstructor) {
    const auto number_neurons = get_random_number_neurons();
    NetworkGraph ng(number_neurons, 0);

    std::stringstream ss{};

    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        ss.clear();
        ss << number_neurons << ' ' << neuron_id << '\n';

        const auto id = NeuronID{ neuron_id };

        const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(id);
        const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(id);
        const auto out_edges_count = ng.get_number_out_edges(id);

        ASSERT_EQ(exc_in_edges_count, 0) << ss.str();
        ASSERT_EQ(inh_in_edges_count, 0) << ss.str();
        ASSERT_EQ(out_edges_count, 0) << ss.str();

        const auto& local_in_edges = ng.get_local_in_edges(id);
        const auto& distant_in_edges = ng.get_distant_in_edges(id);
        const auto& local_out_edges = ng.get_local_out_edges(id);
        const auto& distant_out_edges = ng.get_distant_out_edges(id);

        ASSERT_EQ(local_in_edges.size(), 0) << ss.str();
        ASSERT_EQ(distant_in_edges.size(), 0) << ss.str();
        ASSERT_EQ(local_out_edges.size(), 0) << ss.str();
        ASSERT_EQ(distant_out_edges.size(), 0) << ss.str();

        const auto& all_in_edges_excitatory = ng.get_all_in_edges(id, SignalType::Excitatory);
        const auto& all_in_edges_inhibitory = ng.get_all_in_edges(id, SignalType::Inhibitory);
        const auto& all_out_edges_excitatory = ng.get_all_out_edges(id, SignalType::Excitatory);
        const auto& all_out_edges_inhibitory = ng.get_all_out_edges(id, SignalType::Inhibitory);

        ASSERT_EQ(all_in_edges_excitatory.size(), 0) << ss.str();
        ASSERT_EQ(all_in_edges_inhibitory.size(), 0) << ss.str();
        ASSERT_EQ(all_out_edges_excitatory.size(), 0) << ss.str();
        ASSERT_EQ(all_out_edges_inhibitory.size(), 0) << ss.str();

        const auto& in_edges = ng.get_all_in_edges(id);
        const auto& out_edges = ng.get_all_out_edges(id);

        ASSERT_EQ(in_edges.size(), 0) << ss.str();
        ASSERT_EQ(out_edges.size(), 0) << ss.str();
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphConstructorExceptions) {
    const auto number_neurons = get_random_number_neurons();
    NetworkGraph ng(number_neurons, 0);

    const auto faulty_mpi_rank = -static_cast<int>(get_random_number_ranks());

    std::stringstream ss{};
    ss << faulty_mpi_rank << ' ' << number_neurons;

    ASSERT_THROW(NetworkGraph ng_exception(number_neurons, faulty_mpi_rank);, RelearnException) << ss.str();

    for (auto j = 0; j < number_neurons_out_of_scope; j++) {
        const auto neuron_id = number_neurons + get_random_number_neurons();

        ss.clear();
        ss << number_neurons << ' ' << neuron_id;

        const auto id = NeuronID{ neuron_id };

        ASSERT_THROW(const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(id);, RelearnException) << ss.str();
        ASSERT_THROW(const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(id);, RelearnException) << ss.str();
        ASSERT_THROW(const auto out_edges_count = ng.get_number_out_edges(id);, RelearnException) << ss.str();

        ASSERT_THROW(const auto& local_in_edges = ng.get_local_in_edges(id);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& distant_in_edges = ng.get_distant_in_edges(id);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& local_out_edges = ng.get_local_out_edges(id);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& distant_out_edges = ng.get_distant_out_edges(id);, RelearnException) << ss.str();

        ASSERT_THROW(const auto& all_in_edges_excitatory = ng.get_all_in_edges(id, SignalType::Excitatory);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& all_in_edges_inhibitory = ng.get_all_in_edges(id, SignalType::Inhibitory);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& all_out_edges_excitatory = ng.get_all_out_edges(id, SignalType::Excitatory);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& all_out_edges_inhibitory = ng.get_all_out_edges(id, SignalType::Inhibitory);, RelearnException) << ss.str();
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphCreateNeurons) {
    const auto initial_num_neurons = get_random_number_neurons();
    NetworkGraph ng(initial_num_neurons, 0);

    const auto new_neurons = get_random_number_neurons();
    ng.create_neurons(new_neurons);

    std::stringstream ss{};

    const auto number_neurons = initial_num_neurons + new_neurons;

    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        ss.clear();
        ss << number_neurons << ' ' << neuron_id;

        const auto id = NeuronID{ neuron_id };

        const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(id);
        const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(id);
        const auto out_edges_count = ng.get_number_out_edges(id);

        ASSERT_EQ(exc_in_edges_count, 0) << ss.str();
        ASSERT_EQ(inh_in_edges_count, 0) << ss.str();
        ASSERT_EQ(out_edges_count, 0) << ss.str();

        const auto& local_in_edges = ng.get_local_in_edges(id);
        const auto& distant_in_edges = ng.get_distant_in_edges(id);
        const auto& local_out_edges = ng.get_local_out_edges(id);
        const auto& distant_out_edges = ng.get_distant_out_edges(id);

        ASSERT_EQ(local_in_edges.size(), 0) << ss.str();
        ASSERT_EQ(distant_in_edges.size(), 0) << ss.str();
        ASSERT_EQ(local_out_edges.size(), 0) << ss.str();
        ASSERT_EQ(distant_out_edges.size(), 0) << ss.str();

        const auto& all_in_edges_excitatory = ng.get_all_in_edges(id, SignalType::Excitatory);
        const auto& all_in_edges_inhibitory = ng.get_all_in_edges(id, SignalType::Inhibitory);
        const auto& all_out_edges_excitatory = ng.get_all_out_edges(id, SignalType::Excitatory);
        const auto& all_out_edges_inhibitory = ng.get_all_out_edges(id, SignalType::Inhibitory);

        ASSERT_EQ(all_in_edges_excitatory.size(), 0) << ss.str();
        ASSERT_EQ(all_in_edges_inhibitory.size(), 0) << ss.str();
        ASSERT_EQ(all_out_edges_excitatory.size(), 0) << ss.str();
        ASSERT_EQ(all_out_edges_inhibitory.size(), 0) << ss.str();

        const auto& in_edges = ng.get_all_in_edges(id);
        const auto& out_edges = ng.get_all_out_edges(id);

        ASSERT_EQ(in_edges.size(), 0) << ss.str();
        ASSERT_EQ(out_edges.size(), 0) << ss.str();
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphCreateNeuronsException) {
    const auto initial_num_neurons = get_random_number_neurons();
    NetworkGraph ng(initial_num_neurons, 0);

    const auto new_neurons = get_random_number_neurons();
    ng.create_neurons(new_neurons);

    const auto number_neurons = initial_num_neurons + new_neurons;
    std::stringstream ss{};

    for (auto j = 0; j < number_neurons_out_of_scope; j++) {
        const auto neuron_id = number_neurons + get_random_number_neurons();

        ss.clear();
        ss << number_neurons << ' ' << neuron_id;
        const auto id = NeuronID{ neuron_id };

        ASSERT_THROW(const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(id);, RelearnException) << ss.str();
        ASSERT_THROW(const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(id);, RelearnException) << ss.str();
        ASSERT_THROW(const auto out_edges_count = ng.get_number_out_edges(id);, RelearnException) << ss.str();

        ASSERT_THROW(const auto& local_in_edges = ng.get_local_in_edges(id);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& distant_in_edges = ng.get_distant_in_edges(id);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& local_out_edges = ng.get_local_out_edges(id);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& distant_out_edges = ng.get_distant_out_edges(id);, RelearnException) << ss.str();

        ASSERT_THROW(const auto& all_in_edges_excitatory = ng.get_all_in_edges(id, SignalType::Excitatory);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& all_in_edges_inhibitory = ng.get_all_in_edges(id, SignalType::Inhibitory);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& all_out_edges_excitatory = ng.get_all_out_edges(id, SignalType::Excitatory);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& all_out_edges_inhibitory = ng.get_all_out_edges(id, SignalType::Inhibitory);, RelearnException) << ss.str();

        ASSERT_THROW(const auto& in_edges = ng.get_all_in_edges(id);, RelearnException) << ss.str();
        ASSERT_THROW(const auto& out_edges = ng.get_all_out_edges(id);, RelearnException) << ss.str();
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphLocalEdges) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto number_neurons = get_random_number_neurons();
    const auto num_synapses = get_random_number_synapses() + number_neurons;

    NetworkGraph ng(number_neurons, 0);

    std::map<size_t, std::map<size_t, int>> incoming_edges{};
    std::map<size_t, std::map<size_t, int>> outgoing_edges{};

    for (size_t synapse_id = 0; synapse_id < num_synapses; synapse_id++) {
        const auto weight = get_random_synapse_weight();
        const auto source_id = get_random_neuron_id(number_neurons);
        const auto target_id = get_random_neuron_id(number_neurons);

        ng.add_synapse(LocalSynapse(target_id, source_id, weight));
        incoming_edges[target_id.get_neuron_id()][source_id.get_neuron_id()] += weight;
        outgoing_edges[source_id.get_neuron_id()][target_id.get_neuron_id()] += weight;
    }

    erase_empties(incoming_edges);
    erase_empties(outgoing_edges);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& golden_in_edges = incoming_edges[neuron_id.get_neuron_id()];
        const auto& golden_out_edges = outgoing_edges[neuron_id.get_neuron_id()];

        const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);
        const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);
        const auto out_edges_count = ng.get_number_out_edges(neuron_id);

        const auto golden_excitatory_in_edges_count = std::accumulate(golden_in_edges.cbegin(), golden_in_edges.cend(),
            size_t{ 0 }, [](const std::size_t previous, const std::pair<size_t, int>& p) {
                if (p.second < 0) {
                    return previous;
                }
                return previous + static_cast<size_t>(p.second);
            });

        const auto golden_inhibitory_in_edges_count = std::accumulate(golden_in_edges.cbegin(), golden_in_edges.cend(),
            size_t{ 0 }, [](const std::size_t previous, const std::pair<size_t, int>& p) {
                if (p.second > 0) {
                    return previous;
                }
                return previous + static_cast<size_t>(std::abs(p.second));
            });

        const auto golden_out_edges_count = std::accumulate(golden_out_edges.cbegin(), golden_out_edges.cend(),
            size_t{ 0 }, [](const std::size_t previous, const std::pair<size_t, int>& p) {
                return previous + std::abs(p.second);
            });

        ASSERT_EQ(exc_in_edges_count, golden_excitatory_in_edges_count);
        ASSERT_EQ(inh_in_edges_count, golden_inhibitory_in_edges_count);
        ASSERT_EQ(out_edges_count, golden_out_edges_count);

        const auto& local_in_edges = ng.get_local_in_edges(neuron_id);
        const auto& distant_in_edges = ng.get_distant_in_edges(neuron_id);
        const auto& local_out_edges = ng.get_local_out_edges(neuron_id);
        const auto& distant_out_edges = ng.get_distant_out_edges(neuron_id);

        const auto golden_local_in_edges = incoming_edges[neuron_id.get_neuron_id()].size();
        const auto golden_local_out_edges = outgoing_edges[neuron_id.get_neuron_id()].size();

        ASSERT_EQ(local_in_edges.size(), golden_local_in_edges);
        ASSERT_EQ(distant_in_edges.size(), 0);
        ASSERT_EQ(local_out_edges.size(), golden_local_out_edges);
        ASSERT_EQ(distant_out_edges.size(), 0);

        for (const auto& [other_neuron_id, weight] : local_in_edges) {
            ASSERT_EQ(weight, incoming_edges[neuron_id.get_neuron_id()][other_neuron_id.get_neuron_id()]);
        }

        for (const auto& [other_neuron_id, weight] : local_out_edges) {
            ASSERT_EQ(weight, outgoing_edges[neuron_id.get_neuron_id()][other_neuron_id.get_neuron_id()]);
        }

        const auto& all_in_edges_excitatory = ng.get_all_in_edges(neuron_id, SignalType::Excitatory);
        const auto& all_in_edges_inhibitory = ng.get_all_in_edges(neuron_id, SignalType::Inhibitory);
        const auto& all_out_edges_excitatory = ng.get_all_out_edges(neuron_id, SignalType::Excitatory);
        const auto& all_out_edges_inhibitory = ng.get_all_out_edges(neuron_id, SignalType::Inhibitory);

        ASSERT_EQ(all_in_edges_excitatory.size() + all_in_edges_inhibitory.size(), golden_local_in_edges);
        ASSERT_EQ(all_out_edges_excitatory.size() + all_out_edges_inhibitory.size(), golden_local_out_edges);

        for (const auto& [rank_neuron_id, weight] : all_in_edges_excitatory) {
            ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
            ASSERT_EQ(weight, incoming_edges[neuron_id.get_neuron_id()][rank_neuron_id.get_neuron_id().get_neuron_id()]);
        }

        for (const auto& [rank_neuron_id, weight] : all_in_edges_inhibitory) {
            ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
            ASSERT_EQ(weight, incoming_edges[neuron_id.get_neuron_id()][rank_neuron_id.get_neuron_id().get_neuron_id()]);
        }

        for (const auto& [rank_neuron_id, weight] : all_out_edges_excitatory) {
            ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
            ASSERT_EQ(weight, outgoing_edges[neuron_id.get_neuron_id()][rank_neuron_id.get_neuron_id().get_neuron_id()]);
        }

        for (const auto& [rank_neuron_id, weight] : all_out_edges_inhibitory) {
            ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
            ASSERT_EQ(weight, outgoing_edges[neuron_id.get_neuron_id()][rank_neuron_id.get_neuron_id().get_neuron_id()]);
        }

        const auto& in_edges = ng.get_all_in_edges(neuron_id);
        const auto& out_edges = ng.get_all_out_edges(neuron_id);

        ASSERT_EQ(in_edges.size(), golden_local_in_edges);
        ASSERT_EQ(out_edges.size(), golden_local_out_edges);
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphEdges) {
    const auto number_neurons = get_random_number_neurons();
    const auto number_synapses = get_random_number_synapses() + number_neurons;

    NetworkGraph ng(number_neurons, 0);

    std::map<size_t, std::map<RankNeuronId, RelearnTypes::synapse_weight>> in_edges{};
    std::map<size_t, std::map<RankNeuronId, RelearnTypes::synapse_weight>> out_edges{};

    for (size_t edge_id = 0; edge_id < number_synapses; edge_id++) {
        const int other_rank = static_cast<int>(get_random_number_ranks());
        const auto my_neuron_id = get_random_neuron_id(number_neurons);
        const auto other_neuron_id = get_random_neuron_id(number_neurons);

        const RelearnTypes::synapse_weight weight = get_random_synapse_weight();
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ 0, my_neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (is_in_synapse) {
            ng.add_synapse(DistantInSynapse(my_neuron_id, other_id, weight));
            in_edges[my_neuron_id.get_neuron_id()][{ other_rank, other_neuron_id }] += weight;

            if (in_edges[my_neuron_id.get_neuron_id()][{ other_rank, other_neuron_id }] == 0) {
                in_edges[my_neuron_id.get_neuron_id()].erase({ other_rank, other_neuron_id });
            }
        } else {
            ng.add_synapse(DistantOutSynapse(other_id, my_neuron_id, weight));
            out_edges[my_neuron_id.get_neuron_id()][{ other_rank, other_neuron_id }] += weight;

            if (out_edges[my_neuron_id.get_neuron_id()][{ other_rank, other_neuron_id }] == 0) {
                out_edges[my_neuron_id.get_neuron_id()].erase({ other_rank, other_neuron_id });
            }
        }
    }

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto exc_in_edges_count_ng = ng.get_number_excitatory_in_edges(neuron_id);
        const auto inh_in_edges_count_ng = ng.get_number_inhibitory_in_edges(neuron_id);
        const auto out_edges_count_ng = ng.get_number_out_edges(neuron_id);

        const auto& in_edges_ng = ng.get_all_in_edges(neuron_id);
        const auto& out_edges_ng = ng.get_all_out_edges(neuron_id);

        auto exc_in_edges_count_meta = 0.0;
        auto inh_in_edges_count_meta = 0.0;
        auto out_edges_count_meta = 0.0;

        for (const auto& it : in_edges[neuron_id.get_neuron_id()]) {
            if (it.second > 0) {
                exc_in_edges_count_meta += it.second;
            } else {
                inh_in_edges_count_meta += -it.second;
            }
        }

        for (const auto& it : out_edges[neuron_id.get_neuron_id()]) {
            out_edges_count_meta += std::abs(it.second);
        }

        ASSERT_EQ(exc_in_edges_count_ng, exc_in_edges_count_meta);
        ASSERT_EQ(inh_in_edges_count_ng, inh_in_edges_count_meta);
        ASSERT_EQ(out_edges_count_ng, out_edges_count_meta);

        for (const auto& [key, weight_meta] : in_edges[neuron_id.get_neuron_id()]) {
            const auto found_it = std::find(in_edges_ng.begin(), in_edges_ng.end(), std::make_pair(key, weight_meta));
            ASSERT_TRUE(found_it != in_edges_ng.end());
        }

        for (const auto& [key, weight_meta] : out_edges[neuron_id.get_neuron_id()]) {
            const auto found_it = std::find(out_edges_ng.begin(), out_edges_ng.end(), std::make_pair(key, weight_meta));
            ASSERT_TRUE(found_it != out_edges_ng.end());
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphEdges2) {
    const auto number_neurons = get_random_number_neurons();
    const auto number_synapses = get_random_number_synapses() + number_neurons;

    NetworkGraph ng_golden(number_neurons, 0);
    NetworkGraph ng(number_neurons, 0);

    std::vector<LocalSynapse> local_synapses{};
    std::vector<DistantInSynapse> distant_in_synapses{};
    std::vector<DistantOutSynapse> distant_out_synapses{};

    for (size_t edge_id = 0; edge_id < number_synapses; edge_id++) {
        const int other_rank = static_cast<int>(get_random_number_ranks());
        const auto my_neuron_id = get_random_neuron_id(number_neurons);
        const auto other_neuron_id = get_random_neuron_id(number_neurons);

        const auto weight = get_random_synapse_weight();
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ 0, my_neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (is_in_synapse) {
            ng_golden.add_synapse(DistantInSynapse(my_neuron_id, other_id, weight));
            distant_in_synapses.emplace_back(my_neuron_id, other_id, weight);
        } else {
            ng_golden.add_synapse(DistantOutSynapse(other_id, my_neuron_id, weight));
            distant_out_synapses.emplace_back(other_id, my_neuron_id, weight);
        }
    }

    for (size_t synapse_id = 0; synapse_id < number_synapses; synapse_id++) {
        const auto weight = get_random_synapse_weight();
        const auto source_id = get_random_neuron_id(number_neurons);
        const auto target_id = get_random_neuron_id(number_neurons);

        ng_golden.add_synapse(LocalSynapse(target_id, source_id, weight));
        local_synapses.emplace_back(target_id, source_id, weight);
    }

    ng.add_edges(local_synapses, distant_in_synapses, distant_out_synapses);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        auto local_in_golden = ng_golden.get_local_in_edges(neuron_id);
        auto local_out_golden = ng_golden.get_local_out_edges(neuron_id);
        auto distant_in_golden = ng_golden.get_distant_in_edges(neuron_id);
        auto distant_out_golden = ng_golden.get_distant_out_edges(neuron_id);

        std::ranges::sort(local_in_golden);
        std::ranges::sort(local_out_golden);
        std::ranges::sort(distant_in_golden);
        std::ranges::sort(distant_out_golden);

        auto local_in = ng.get_local_in_edges(neuron_id);
        auto local_out = ng.get_local_out_edges(neuron_id);
        auto distant_in = ng.get_distant_in_edges(neuron_id);
        auto distant_out = ng.get_distant_out_edges(neuron_id);

        std::ranges::sort(local_in);
        std::ranges::sort(local_out);
        std::ranges::sort(distant_in);
        std::ranges::sort(distant_out);

        ASSERT_EQ(local_in_golden, local_in);
        ASSERT_EQ(local_out_golden, local_out);
        ASSERT_EQ(distant_in_golden, distant_in);
        ASSERT_EQ(distant_out_golden, distant_out);
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphEdgesSplit) {
    const auto number_neurons = get_random_number_neurons();
    const auto num_edges = get_random_number_synapses() + number_neurons;

    NetworkGraph ng(number_neurons, 0);

    for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
        const int other_rank = static_cast<int>(get_random_number_ranks());
        const auto neuron_id = get_random_neuron_id(number_neurons);
        const auto other_neuron_id = get_random_neuron_id(number_neurons);

        const auto weight = get_random_synapse_weight();
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ 0, neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (is_in_synapse) {
            ng.add_synapse(DistantInSynapse(neuron_id, other_id, weight));
        } else {
            ng.add_synapse(DistantOutSynapse(other_id, neuron_id, weight));
        }
    }

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& in_edges_ng = ng.get_all_in_edges(neuron_id);
        const auto& out_edges_ng = ng.get_all_out_edges(neuron_id);

        auto in_edges_ng_ex = ng.get_all_in_edges(neuron_id, SignalType::Excitatory);
        const auto& in_edges_ng_in = ng.get_all_in_edges(neuron_id, SignalType::Inhibitory);
        const auto& out_edges_ng_ex = ng.get_all_out_edges(neuron_id, SignalType::Excitatory);
        auto out_edges_ng_in = ng.get_all_out_edges(neuron_id, SignalType::Inhibitory);

        ASSERT_EQ(in_edges_ng.size(), in_edges_ng_ex.size() + in_edges_ng_in.size());
        ASSERT_EQ(out_edges_ng.size(), out_edges_ng_ex.size() + out_edges_ng_in.size());

        for (const auto& [_, edge_val] : in_edges_ng) {
            ASSERT_TRUE(edge_val < 0);
        }

        for (const auto& [_, edge_val] : out_edges_ng_in) {
            ASSERT_TRUE(edge_val < 0);
        }

        for (const auto& [_, edge_val] : in_edges_ng_ex) {
            ASSERT_TRUE(edge_val > 0);
        }

        for (const auto& [_, edge_val] : out_edges_ng_ex) {
            ASSERT_TRUE(edge_val > 0);
        }

        for (const auto& val : in_edges_ng_in) {
            in_edges_ng_ex.emplace_back(val);
        }

        for (const auto& val : out_edges_ng_ex) {
            out_edges_ng_in.emplace_back(val);
        }

        ASSERT_EQ(in_edges_ng.size(), in_edges_ng_ex.size());
        ASSERT_EQ(out_edges_ng.size(), out_edges_ng_ex.size());

        for (const auto& [edge_key, edge_val] : in_edges_ng) {
            const auto found_it = std::find(in_edges_ng_ex.begin(), in_edges_ng_ex.end(), std::make_pair(edge_key, edge_val));
            ASSERT_TRUE(found_it != in_edges_ng_ex.end());
        }

        for (const auto& [edge_key, edge_val] : out_edges_ng) {
            const auto found_it = std::find(out_edges_ng_ex.begin(), out_edges_ng_ex.end(), std::make_pair(edge_key, edge_val));
            ASSERT_TRUE(found_it != out_edges_ng_ex.end());
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphEdgesRemoval) {
    const auto number_neurons = 10; // get_random_number_neurons();
    const auto num_edges = 10; // get_random_number_synapses() + number_neurons;

    NetworkGraph ng(number_neurons, 0);

    std::vector<std::tuple<NeuronID, int, NeuronID, int, int>> synapses(num_edges);

    for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
        const int other_rank = static_cast<int>(get_random_number_ranks());
        const auto neuron_id = get_random_neuron_id(number_neurons);
        const auto other_neuron_id = get_random_neuron_id(number_neurons);

        const auto weight = get_random_synapse_weight();
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ 0, neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (other_rank == 0) {
            ng.add_synapse(LocalSynapse(neuron_id, other_neuron_id, weight));
            synapses[edge_id] = std::make_tuple(neuron_id, 0, other_neuron_id, 0, weight);
            continue;
        }

        if (is_in_synapse) {
            ng.add_synapse(DistantInSynapse(neuron_id, other_id, weight));
            synapses[edge_id] = std::make_tuple(neuron_id, 0, other_neuron_id, other_rank, weight);
        } else {
            ng.add_synapse(DistantOutSynapse(other_id, neuron_id, weight));
            synapses[edge_id] = std::make_tuple(other_neuron_id, other_rank, neuron_id, 0, weight);
        }
    }

    std::shuffle(synapses.begin(), synapses.end(), mt);

    for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
        const auto& current_synapse = synapses[edge_id];
        const auto& [target_neuron_id, target_rank, source_neuron_id, source_rank, weight] = current_synapse;

        RankNeuronId target_id{ target_rank, target_neuron_id };
        RankNeuronId source_id{ source_rank, source_neuron_id };

        if (source_rank == 0 && target_rank == 0) {
            ng.add_synapse(LocalSynapse(target_neuron_id, source_neuron_id, -weight));
            continue;
        }

        if (source_rank == 0) {
            ng.add_synapse(DistantOutSynapse(target_id, source_neuron_id, -weight));
            continue;
        }

        ng.add_synapse(DistantInSynapse(target_neuron_id, source_id, -weight));
    }

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);
        const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);
        const auto out_edges_count = ng.get_number_out_edges(neuron_id);

        ASSERT_EQ(exc_in_edges_count, 0);
        ASSERT_EQ(inh_in_edges_count, 0);
        ASSERT_EQ(out_edges_count, 0);

        const auto& in_edges = ng.get_all_in_edges(neuron_id);
        const auto& out_edges = ng.get_all_out_edges(neuron_id);

        ASSERT_EQ(in_edges.size(), 0);
        ASSERT_EQ(out_edges.size(), 0);
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphCreate) {
    const auto number_neurons = get_random_number_neurons();
    const auto num_edges = get_random_number_synapses() + number_neurons;

    NetworkGraph ng(number_neurons, 0);

    std::map<RankNeuronId, std::map<RankNeuronId, RelearnTypes::synapse_weight>> in_edges;
    std::map<RankNeuronId, std::map<RankNeuronId, RelearnTypes::synapse_weight>> out_edges;

    for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
        const int other_rank = static_cast<int>(get_random_number_ranks());
        const auto neuron_id = get_random_neuron_id(number_neurons);
        const auto other_neuron_id = get_random_neuron_id(number_neurons);

        const auto weight = get_random_synapse_weight();
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ 0, neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (is_in_synapse) {
            ng.add_synapse(DistantInSynapse(neuron_id, other_id, weight));
            in_edges[my_id][other_id] += weight;
        } else {
            ng.add_synapse(DistantOutSynapse(other_id, neuron_id, weight));
            out_edges[my_id][other_id] += weight;
        }
    }

    const auto num_new_neurons = get_random_number_neurons();
    const auto num_new_edges = get_random_number_synapses();

    const auto total_number_neurons = number_neurons + num_new_neurons;
    const auto total_num_edges = num_edges + num_new_edges;

    ng.create_neurons(num_new_neurons);

    for (size_t edge_id = num_edges; edge_id < total_num_edges; edge_id++) {
        const int other_rank = static_cast<int>(get_random_number_ranks());
        const auto neuron_id = get_random_neuron_id(number_neurons);
        const auto other_neuron_id = get_random_neuron_id(number_neurons);

        const auto weight = get_random_synapse_weight();
        const auto is_in_synapse = weight < 0;

        RankNeuronId my_id{ 0, neuron_id };
        RankNeuronId other_id{ other_rank, other_neuron_id };

        if (is_in_synapse) {
            ng.add_synapse(DistantInSynapse(neuron_id, other_id, weight));
            in_edges[my_id][other_id] += weight;
        } else {
            ng.add_synapse(DistantOutSynapse(other_id, neuron_id, weight));
            out_edges[my_id][other_id] += weight;
        }
    }

    for (auto neuron_id : NeuronID::range(total_number_neurons)) {
        const auto exc_in_edges_count_ng = ng.get_number_excitatory_in_edges(neuron_id);
        const auto inh_in_edges_count_ng = ng.get_number_inhibitory_in_edges(neuron_id);
        const auto out_edges_count_ng = ng.get_number_out_edges(neuron_id);

        const auto& in_edges_ng = ng.get_all_in_edges(neuron_id);
        const auto& out_edges_ng = ng.get_all_out_edges(neuron_id);

        auto exc_in_edges_count_meta = 0.0;
        auto inh_in_edges_count_meta = 0.0;
        auto out_edges_count_meta = 0.0;

        for (const auto& it : in_edges[{ 0, neuron_id }]) {
            if (it.second > 0) {
                exc_in_edges_count_meta += it.second;
            } else {
                inh_in_edges_count_meta += -it.second;
            }
        }

        for (const auto& it : out_edges[{ 0, neuron_id }]) {
            out_edges_count_meta += std::abs(it.second);
        }

        ASSERT_EQ(exc_in_edges_count_ng, exc_in_edges_count_meta);
        ASSERT_EQ(inh_in_edges_count_ng, inh_in_edges_count_meta);
        ASSERT_EQ(out_edges_count_ng, out_edges_count_meta);

        for (const auto& [key, weight_meta] : in_edges[{ 0, neuron_id }]) {
            const auto found_it = std::find(in_edges_ng.begin(), in_edges_ng.end(), std::make_pair(key, weight_meta));
            ASSERT_TRUE(found_it != in_edges_ng.end());
        }

        for (const auto& [key, weight_meta] : out_edges[{ 0, neuron_id }]) {
            const auto found_it = std::find(out_edges_ng.begin(), out_edges_ng.end(), std::make_pair(key, weight_meta));
            ASSERT_TRUE(found_it != out_edges_ng.end());
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphHistogramPositiveWeight) {
    const auto number_neurons = get_random_number_neurons();
    const auto number_synapses = get_random_number_synapses() + number_neurons;

    const auto& synapses = get_random_synapses(number_neurons, number_synapses);

    NetworkGraph ng(number_neurons, 0);

    std::map<std::pair<NeuronID, NeuronID>, int> reduced_synapses{};

    for (const auto& [source_id, target_id, weight] : synapses) {
        if (weight == 0) {
            continue;
        }

        const auto abs_weight = std::abs(weight);

        ng.add_synapse(LocalSynapse(target_id, source_id, abs_weight));
        reduced_synapses[{ source_id, target_id }] += abs_weight;
    }

    std::map<size_t, int> in_synapses{};
    std::map<size_t, int> out_synapses{};

    for (const auto& [source_target, weight] : reduced_synapses) {
        const auto& [source_id, target_id] = source_target;

        out_synapses[source_id.get_neuron_id()] += weight;
        in_synapses[target_id.get_neuron_id()] += weight;
    }

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        out_synapses[neuron_id] = out_synapses[neuron_id];
        in_synapses[neuron_id] = in_synapses[neuron_id];
    }

    std::map<int, size_t> golden_in_histogram{};
    std::map<int, size_t> golden_out_histogram{};

    for (const auto& [_, weight] : in_synapses) {
        golden_in_histogram[weight]++;
    }

    for (const auto& [_, weight] : out_synapses) {
        golden_out_histogram[weight]++;
    }

    const auto& in_histogram = ng.get_edges_histogram(NetworkGraph::EdgeDirection::In);
    const auto& out_histogram = ng.get_edges_histogram(NetworkGraph::EdgeDirection::Out);

    for (auto i = 0; i < in_histogram.size(); i++) {
        const auto golden_number_in_edges = golden_in_histogram[i];
        const auto number_in_edges = in_histogram[i];

        ASSERT_EQ(golden_number_in_edges, number_in_edges);
    }

    for (auto i = 0; i < out_histogram.size(); i++) {
        const auto golden_number_out_edges = golden_out_histogram[i];
        const auto number_out_edges = out_histogram[i];

        ASSERT_EQ(golden_number_out_edges, number_out_edges);
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphHistogram) {
    const auto number_neurons = get_random_number_neurons();
    const auto number_synapses = get_random_number_synapses() + number_neurons;

    const auto& synapses = get_random_synapses(number_neurons, number_synapses);

    NetworkGraph ng(number_neurons, 0);

    std::map<std::pair<NeuronID, NeuronID>, int> reduced_synapses{};

    for (const auto& [source_id, target_id, weight] : synapses) {
        if (weight == 0) {
            continue;
        }

        ng.add_synapse(LocalSynapse(target_id, source_id, weight));
        reduced_synapses[{ source_id, target_id }] += weight;
    }

    std::map<size_t, int> in_synapses{};
    std::map<size_t, int> out_synapses{};

    for (const auto& [source_target, weight] : reduced_synapses) {
        const auto& [source_id, target_id] = source_target;

        out_synapses[source_id.get_neuron_id()] += std::abs(weight);
        in_synapses[target_id.get_neuron_id()] += std::abs(weight);
    }

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        out_synapses[neuron_id] = out_synapses[neuron_id];
        in_synapses[neuron_id] = in_synapses[neuron_id];
    }

    std::map<int, size_t> golden_in_histogram{};
    std::map<int, size_t> golden_out_histogram{};

    for (const auto& [_, weight] : in_synapses) {
        golden_in_histogram[weight]++;
    }

    for (const auto& [_, weight] : out_synapses) {
        golden_out_histogram[weight]++;
    }

    const auto& in_histogram = ng.get_edges_histogram(NetworkGraph::EdgeDirection::In);
    const auto& out_histogram = ng.get_edges_histogram(NetworkGraph::EdgeDirection::Out);

    for (auto i = 0; i < in_histogram.size(); i++) {
        const auto golden_number_in_edges = golden_in_histogram[i];
        const auto number_in_edges = in_histogram[i];

        ASSERT_EQ(golden_number_in_edges, number_in_edges);
    }

    for (auto i = 0; i < out_histogram.size(); i++) {
        const auto golden_number_out_edges = golden_out_histogram[i];
        const auto number_out_edges = out_histogram[i];

        ASSERT_EQ(golden_number_out_edges, number_out_edges);
    }
}
