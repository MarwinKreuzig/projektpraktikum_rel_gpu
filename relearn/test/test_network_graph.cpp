#include "../googletest/include/gtest/gtest.h"

#include <map>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "RelearnTest.hpp"

#include "../source/neurons/NetworkGraph.h"
#include "../source/util/RelearnException.h"

TEST_F(NetworkGraphTest, testNetworkGraphConstructor) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);

    for (auto i = 0; i < iterations; i++) {
        const auto number_neurons = uid_num_neurons(mt);

        NetworkGraph ng(number_neurons, 0);

        for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);
            const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);
            const auto out_edges_count = ng.get_number_out_edges(neuron_id);

            ASSERT_EQ(exc_in_edges_count, 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(inh_in_edges_count, 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(out_edges_count, 0) << i << number_neurons << neuron_id;

            const auto& local_in_edges = ng.get_local_in_edges(neuron_id);
            const auto& distant_in_edges = ng.get_distant_in_edges(neuron_id);
            const auto& local_out_edges = ng.get_local_out_edges(neuron_id);
            const auto& distant_out_edges = ng.get_distant_out_edges(neuron_id);

            ASSERT_EQ(local_in_edges.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(distant_in_edges.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(local_out_edges.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(distant_out_edges.size(), 0) << i << number_neurons << neuron_id;

            const auto& all_in_edges_excitatory = ng.get_all_in_edges(neuron_id, SignalType::EXCITATORY);
            const auto& all_in_edges_inhibitory = ng.get_all_in_edges(neuron_id, SignalType::INHIBITORY);
            const auto& all_out_edges_excitatory = ng.get_all_out_edges(neuron_id, SignalType::EXCITATORY);
            const auto& all_out_edges_inhibitory = ng.get_all_out_edges(neuron_id, SignalType::INHIBITORY);

            ASSERT_EQ(all_in_edges_excitatory.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(all_in_edges_inhibitory.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(all_out_edges_excitatory.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(all_out_edges_inhibitory.size(), 0) << i << number_neurons << neuron_id;

            const auto& in_edges = ng.get_all_in_edges(neuron_id);
            const auto& out_edges = ng.get_all_out_edges(neuron_id);

            ASSERT_EQ(in_edges.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(out_edges.size(), 0) << i << number_neurons << neuron_id;
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphConstructorExceptions) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);

    for (auto i = 0; i < iterations; i++) {
        size_t number_neurons = uid_num_neurons(mt);

        NetworkGraph ng(number_neurons, 0);

        ASSERT_THROW(NetworkGraph ng_exception(number_neurons, -number_neurons - 1);, RelearnException) << i << number_neurons;

        for (auto j = 0; j < iterations; j++) {
            const auto neuron_id = number_neurons + 1 + uid_num_neurons(mt);

            ASSERT_THROW(const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto out_edges_count = ng.get_number_out_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;

            ASSERT_THROW(const auto& local_in_edges = ng.get_local_in_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& distant_in_edges = ng.get_distant_in_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& local_out_edges = ng.get_local_out_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& distant_out_edges = ng.get_distant_out_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;

            ASSERT_THROW(const auto& all_in_edges_excitatory = ng.get_all_in_edges(neuron_id, SignalType::EXCITATORY);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& all_in_edges_inhibitory = ng.get_all_in_edges(neuron_id, SignalType::INHIBITORY);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& all_out_edges_excitatory = ng.get_all_out_edges(neuron_id, SignalType::EXCITATORY);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& all_out_edges_inhibitory = ng.get_all_out_edges(neuron_id, SignalType::INHIBITORY);, RelearnException) << i << number_neurons << neuron_id;
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphCreateNeurons) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);

    for (auto i = 0; i < iterations; i++) {
        const auto initial_num_neurons = uid_num_neurons(mt);
        NetworkGraph ng(initial_num_neurons, 0);

        const auto new_neurons = uid_num_neurons(mt);
        ng.create_neurons(new_neurons);

        const auto number_neurons = initial_num_neurons + new_neurons;

        for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);
            const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);
            const auto out_edges_count = ng.get_number_out_edges(neuron_id);

            ASSERT_EQ(exc_in_edges_count, 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(inh_in_edges_count, 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(out_edges_count, 0) << i << number_neurons << neuron_id;

            const auto& local_in_edges = ng.get_local_in_edges(neuron_id);
            const auto& distant_in_edges = ng.get_distant_in_edges(neuron_id);
            const auto& local_out_edges = ng.get_local_out_edges(neuron_id);
            const auto& distant_out_edges = ng.get_distant_out_edges(neuron_id);

            ASSERT_EQ(local_in_edges.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(distant_in_edges.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(local_out_edges.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(distant_out_edges.size(), 0) << i << number_neurons << neuron_id;

            const auto& all_in_edges_excitatory = ng.get_all_in_edges(neuron_id, SignalType::EXCITATORY);
            const auto& all_in_edges_inhibitory = ng.get_all_in_edges(neuron_id, SignalType::INHIBITORY);
            const auto& all_out_edges_excitatory = ng.get_all_out_edges(neuron_id, SignalType::EXCITATORY);
            const auto& all_out_edges_inhibitory = ng.get_all_out_edges(neuron_id, SignalType::INHIBITORY);

            ASSERT_EQ(all_in_edges_excitatory.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(all_in_edges_inhibitory.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(all_out_edges_excitatory.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(all_out_edges_inhibitory.size(), 0) << i << number_neurons << neuron_id;

            const auto& in_edges = ng.get_all_in_edges(neuron_id);
            const auto& out_edges = ng.get_all_out_edges(neuron_id);

            ASSERT_EQ(in_edges.size(), 0) << i << number_neurons << neuron_id;
            ASSERT_EQ(out_edges.size(), 0) << i << number_neurons << neuron_id;
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphCreateNeuronsException) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);

    for (auto i = 0; i < iterations; i++) {
        const auto initial_num_neurons = uid_num_neurons(mt);
        NetworkGraph ng(initial_num_neurons, 0);

        const auto new_neurons = uid_num_neurons(mt);
        ng.create_neurons(new_neurons);

        const auto number_neurons = initial_num_neurons + new_neurons;

        for (auto j = 0; j < iterations; j++) {
            const auto neuron_id = number_neurons + 1 + uid_num_neurons(mt);

            ASSERT_THROW(const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto out_edges_count = ng.get_number_out_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;

            ASSERT_THROW(const auto& local_in_edges = ng.get_local_in_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& distant_in_edges = ng.get_distant_in_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& local_out_edges = ng.get_local_out_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& distant_out_edges = ng.get_distant_out_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;

            ASSERT_THROW(const auto& all_in_edges_excitatory = ng.get_all_in_edges(neuron_id, SignalType::EXCITATORY);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& all_in_edges_inhibitory = ng.get_all_in_edges(neuron_id, SignalType::INHIBITORY);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& all_out_edges_excitatory = ng.get_all_out_edges(neuron_id, SignalType::EXCITATORY);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& all_out_edges_inhibitory = ng.get_all_out_edges(neuron_id, SignalType::INHIBITORY);, RelearnException) << i << number_neurons << neuron_id;

            ASSERT_THROW(const auto& in_edges = ng.get_all_in_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
            ASSERT_THROW(const auto& out_edges = ng.get_all_out_edges(neuron_id);, RelearnException) << i << number_neurons << neuron_id;
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphLocalEdges) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<size_t> uid_num_synapses(0, upper_bound_num_neurons * num_synapses_per_neuron);
    std::uniform_int_distribution<int> uid_edge_weight(-bound_synapse_weight, bound_synapse_weight);

    const auto my_rank = MPIWrapper::get_my_rank();

    for (auto i = 0; i < iterations; i++) {
        const auto number_neurons = uid_num_neurons(mt);
        size_t num_synapses = uid_num_synapses(mt) + number_neurons;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, number_neurons - 1);

        NetworkGraph ng(number_neurons, 0);

        std::map<size_t, std::map<size_t, int>> incoming_edges{};
        std::map<size_t, std::map<size_t, int>> outgoing_edges{};

        for (size_t synapse_id = 0; synapse_id < num_synapses; synapse_id++) {
            auto weight = uid_edge_weight(mt);
            if (weight == 0) {
                weight++;
            }

            const auto source_id = uid_actual_num_neurons(mt);
            const auto target_id = uid_actual_num_neurons(mt);

            ng.add_edge_weight(RankNeuronId(my_rank, target_id), RankNeuronId(my_rank, source_id), weight);
            incoming_edges[target_id][source_id] += weight;
            outgoing_edges[source_id][target_id] += weight;
        }

        erase_empties(incoming_edges);
        erase_empties(outgoing_edges);

        for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            const auto& golden_in_edges = incoming_edges[neuron_id];
            const auto& golden_out_edges = outgoing_edges[neuron_id];

            const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);
            const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);
            const auto out_edges_count = ng.get_number_out_edges(neuron_id);

            const auto golden_excitatory_in_edges_count = std::accumulate(golden_in_edges.cbegin(), golden_in_edges.cend(),
                0, [](const std::size_t previous, const std::pair<size_t, int>& p) {
                    if (p.second < 0) {
                        return previous;
                    }
                    return previous + p.second;
                });

            const auto golden_inhibitory_in_edges_count = std::accumulate(golden_in_edges.cbegin(), golden_in_edges.cend(),
                0, [](const std::size_t previous, const std::pair<size_t, int>& p) {
                    if (p.second > 0) {
                        return previous;
                    }
                    return previous - p.second;
                });

            const auto golden_out_edges_count = std::accumulate(golden_out_edges.cbegin(), golden_out_edges.cend(),
                0, [](const std::size_t previous, const std::pair<size_t, int>& p) {
                    return previous + std::abs(p.second);
                });

            ASSERT_EQ(exc_in_edges_count, golden_excitatory_in_edges_count);
            ASSERT_EQ(inh_in_edges_count, golden_inhibitory_in_edges_count);
            ASSERT_EQ(out_edges_count, golden_out_edges_count);

            const auto& local_in_edges = ng.get_local_in_edges(neuron_id);
            const auto& distant_in_edges = ng.get_distant_in_edges(neuron_id);
            const auto& local_out_edges = ng.get_local_out_edges(neuron_id);
            const auto& distant_out_edges = ng.get_distant_out_edges(neuron_id);

            const auto golden_local_in_edges = incoming_edges[neuron_id].size();
            const auto golden_local_out_edges = outgoing_edges[neuron_id].size();

            ASSERT_EQ(local_in_edges.size(), golden_local_in_edges);
            ASSERT_EQ(distant_in_edges.size(), 0);
            ASSERT_EQ(local_out_edges.size(), golden_local_out_edges);
            ASSERT_EQ(distant_out_edges.size(), 0);

            for (const auto& [other_neuron_id, weight] : local_in_edges) {
                ASSERT_EQ(weight, incoming_edges[neuron_id][other_neuron_id]);
            }

            for (const auto& [other_neuron_id, weight] : local_out_edges) {
                ASSERT_EQ(weight, outgoing_edges[neuron_id][other_neuron_id]);
            }

            const auto& all_in_edges_excitatory = ng.get_all_in_edges(neuron_id, SignalType::EXCITATORY);
            const auto& all_in_edges_inhibitory = ng.get_all_in_edges(neuron_id, SignalType::INHIBITORY);
            const auto& all_out_edges_excitatory = ng.get_all_out_edges(neuron_id, SignalType::EXCITATORY);
            const auto& all_out_edges_inhibitory = ng.get_all_out_edges(neuron_id, SignalType::INHIBITORY);

            ASSERT_EQ(all_in_edges_excitatory.size() + all_in_edges_inhibitory.size(), golden_local_in_edges);
            ASSERT_EQ(all_out_edges_excitatory.size() + all_out_edges_inhibitory.size(), golden_local_out_edges);

            for (const auto& [rank_neuron_id, weight] : all_in_edges_excitatory) {
                ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
                ASSERT_EQ(weight, incoming_edges[neuron_id][rank_neuron_id.get_neuron_id()]);
            }

            for (const auto& [rank_neuron_id, weight] : all_in_edges_inhibitory) {
                ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
                ASSERT_EQ(weight, incoming_edges[neuron_id][rank_neuron_id.get_neuron_id()]);
            }

            for (const auto& [rank_neuron_id, weight] : all_out_edges_excitatory) {
                ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
                ASSERT_EQ(weight, outgoing_edges[neuron_id][rank_neuron_id.get_neuron_id()]);
            }

            for (const auto& [rank_neuron_id, weight] : all_out_edges_inhibitory) {
                ASSERT_EQ(rank_neuron_id.get_rank(), my_rank);
                ASSERT_EQ(weight, outgoing_edges[neuron_id][rank_neuron_id.get_neuron_id()]);
            }

            const auto& in_edges = ng.get_all_in_edges(neuron_id);
            const auto& out_edges = ng.get_all_out_edges(neuron_id);

            ASSERT_EQ(in_edges.size(), golden_local_in_edges);
            ASSERT_EQ(out_edges.size(), golden_local_out_edges);
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphDistantEdges) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<size_t> uid_num_synapses(0, upper_bound_num_neurons * num_synapses_per_neuron);
    std::uniform_int_distribution<int> uid_edge_weight(-bound_synapse_weight, bound_synapse_weight);

    std::uniform_int_distribution<int> uid_rank(0, 1);

    for (auto i = 0; i < iterations; i++) {
        const auto num_neurons_1 = uid_num_neurons(mt);
        const auto num_neurons_2 = uid_num_neurons(mt);

        const auto num_synapses = uid_num_synapses(mt);

        NetworkGraph ng_1(num_neurons_1, 0);
        NetworkGraph ng_2(num_neurons_2, 1);

        std::uniform_int_distribution<size_t> uid_actual_neurons_1(0, num_neurons_1 - 1);
        std::uniform_int_distribution<size_t> uid_actual_neurons_2(0, num_neurons_2 - 1);

        std::map<std::tuple<RankNeuronId, RankNeuronId>, int> golden_connections{};

        for (auto synapse_id = 0; synapse_id < num_synapses; synapse_id++) {
            const auto source_rank = uid_rank(mt);
            const auto target_rank = uid_rank(mt);

            auto source_id = uid_actual_neurons_1(mt);
            auto target_id = uid_actual_neurons_1(mt);

            if (source_rank == 1) {
                source_id = uid_actual_neurons_2(mt);
            }

            if (target_rank == 1) {
                target_id = uid_actual_neurons_2(mt);
            }

            const auto is_rank_0_touched = source_rank == 0 || target_rank == 0;
            const auto is_rank_1_touched = source_rank == 1 || target_rank == 1;

            RankNeuronId rn_target{ target_rank, target_id };
            RankNeuronId rn_source{ source_rank, source_id };

            auto weight = uid_edge_weight(mt);
            if (weight == 0) {
                weight++;
            }

            if (is_rank_0_touched) {
                ng_1.add_edge_weight(rn_target, rn_source, weight);
            }

            if (is_rank_1_touched) {
                ng_2.add_edge_weight(rn_target, rn_source, weight);
            }

            std::tuple<RankNeuronId, RankNeuronId> key{ rn_target, rn_source };
            golden_connections[key] += weight;
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphEdges) {
    std::uniform_int_distribution<int> uid_num_ranks(1, num_ranks);
    std::uniform_int_distribution<int> uid_edge_weight(-bound_synapse_weight, bound_synapse_weight);

    for (auto i = 0; i < iterations; i++) {
        const auto number_neurons = get_random_number_neurons();
        const auto number_synapses = get_random_number_synapses() + number_neurons;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, number_neurons - 1);

        NetworkGraph ng(number_neurons, 0);

        std::map<size_t, std::map<RankNeuronId, int>> in_edges;
        std::map<size_t, std::map<RankNeuronId, int>> out_edges;

        for (size_t edge_id = 0; edge_id < number_synapses; edge_id++) {
            int other_rank = uid_num_ranks(mt);
            size_t my_neuron_id = uid_actual_num_neurons(mt);
            size_t other_neuron_id = uid_actual_num_neurons(mt);

            int weight = uid_edge_weight(mt);

            while (weight == 0) {
                weight = uid_edge_weight(mt);
            }

            bool is_in_synapse = weight < 0;

            RankNeuronId my_id{ 0, my_neuron_id };
            RankNeuronId other_id{ other_rank, other_neuron_id };

            if (is_in_synapse) {
                ng.add_edge_weight(my_id, other_id, weight);
                in_edges[my_neuron_id][{ other_rank, other_neuron_id }] += weight;

                if (in_edges[my_neuron_id][{ other_rank, other_neuron_id }] == 0) {
                    in_edges[my_neuron_id].erase({ other_rank, other_neuron_id });
                }
            } else {
                ng.add_edge_weight(other_id, my_id, weight);
                out_edges[my_neuron_id][{ other_rank, other_neuron_id }] += weight;

                if (out_edges[my_neuron_id][{ other_rank, other_neuron_id }] == 0) {
                    out_edges[my_neuron_id].erase({ other_rank, other_neuron_id });
                }
            }
        }

        for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            size_t exc_in_edges_count_ng = ng.get_number_excitatory_in_edges(neuron_id);
            size_t inh_in_edges_count_ng = ng.get_number_inhibitory_in_edges(neuron_id);
            size_t out_edges_count_ng = ng.get_number_out_edges(neuron_id);

            const std::vector<std::pair<RankNeuronId, int>>& in_edges_ng = ng.get_all_in_edges(neuron_id);
            const std::vector<std::pair<RankNeuronId, int>>& out_edges_ng = ng.get_all_out_edges(neuron_id);

            size_t exc_in_edges_count_meta = 0;
            size_t inh_in_edges_count_meta = 0;
            size_t out_edges_count_meta = 0;

            for (const auto& it : in_edges[neuron_id]) {
                if (it.second > 0) {
                    exc_in_edges_count_meta += it.second;
                } else {
                    inh_in_edges_count_meta += -it.second;
                }
            }

            for (const auto& it : out_edges[neuron_id]) {
                out_edges_count_meta += std::abs(it.second);
            }

            ASSERT_EQ(exc_in_edges_count_ng, exc_in_edges_count_meta);
            ASSERT_EQ(inh_in_edges_count_ng, inh_in_edges_count_meta);
            ASSERT_EQ(out_edges_count_ng, out_edges_count_meta);

            for (const auto& it : in_edges[neuron_id]) {
                int weight_meta = it.second;
                RankNeuronId key = it.first;
                auto found_it = std::find(in_edges_ng.begin(), in_edges_ng.end(), std::make_pair(key, weight_meta));
                ASSERT_TRUE(found_it != in_edges_ng.end());
            }

            for (const auto& it : out_edges[neuron_id]) {
                int weight_meta = it.second;
                RankNeuronId key = it.first;
                auto found_it = std::find(out_edges_ng.begin(), out_edges_ng.end(), std::make_pair(key, weight_meta));
                ASSERT_TRUE(found_it != out_edges_ng.end());
            }
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphEdgesSplit) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<size_t> uid_num_edges(0, upper_bound_num_neurons * num_synapses_per_neuron);

    std::uniform_int_distribution<int> uid_num_ranks(1, num_ranks);
    std::uniform_int_distribution<int> uid_edge_weight(-bound_synapse_weight, bound_synapse_weight);

    for (auto i = 0; i < iterations; i++) {
        size_t number_neurons = uid_num_neurons(mt);
        size_t num_edges = uid_num_edges(mt) + number_neurons;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, number_neurons - 1);

        NetworkGraph ng(number_neurons, 0);

        for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
            int other_rank = uid_num_ranks(mt);
            size_t neuron_id = uid_actual_num_neurons(mt);
            size_t other_neuron_id = uid_actual_num_neurons(mt);

            int weight = uid_edge_weight(mt);

            while (weight == 0) {
                weight = uid_edge_weight(mt);
            }

            bool is_in_synapse = weight < 0;

            RankNeuronId my_id{ 0, neuron_id };
            RankNeuronId other_id{ other_rank, other_neuron_id };

            if (is_in_synapse) {
                ng.add_edge_weight(my_id, other_id, weight);
            } else {
                ng.add_edge_weight(other_id, my_id, weight);
            }
        }

        for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            size_t exc_in_edges_count_ng = ng.get_number_excitatory_in_edges(neuron_id);
            size_t inh_in_edges_count_ng = ng.get_number_inhibitory_in_edges(neuron_id);
            size_t out_edges_count_ng = ng.get_number_out_edges(neuron_id);

            const std::vector<std::pair<RankNeuronId, int>>& in_edges_ng = ng.get_all_in_edges(neuron_id);
            const std::vector<std::pair<RankNeuronId, int>>& out_edges_ng = ng.get_all_out_edges(neuron_id);

            std::vector<std::pair<RankNeuronId, int>> in_edges_ng_ex = ng.get_all_in_edges(neuron_id, SignalType::EXCITATORY);
            std::vector<std::pair<RankNeuronId, int>> in_edges_ng_in = ng.get_all_in_edges(neuron_id, SignalType::INHIBITORY);
            std::vector<std::pair<RankNeuronId, int>> out_edges_ng_ex = ng.get_all_out_edges(neuron_id, SignalType::EXCITATORY);
            std::vector<std::pair<RankNeuronId, int>> out_edges_ng_in = ng.get_all_out_edges(neuron_id, SignalType::INHIBITORY);

            ASSERT_EQ(in_edges_ng.size(), in_edges_ng_ex.size() + in_edges_ng_in.size());
            ASSERT_EQ(out_edges_ng.size(), out_edges_ng_ex.size() + out_edges_ng_in.size());

            for (const auto& [edge_key, edge_val] : in_edges_ng) {
                ASSERT_TRUE(edge_val < 0);
            }

            for (const auto& [edge_key, edge_val] : out_edges_ng_in) {
                ASSERT_TRUE(edge_val < 0);
            }

            for (const auto& [edge_key, edge_val] : in_edges_ng_ex) {
                ASSERT_TRUE(edge_val > 0);
            }

            for (const auto& [edge_key, edge_val] : out_edges_ng_ex) {
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
                auto found_it = std::find(in_edges_ng_ex.begin(), in_edges_ng_ex.end(), std::make_pair(edge_key, edge_val));
                ASSERT_TRUE(found_it != in_edges_ng_ex.end());
            }

            for (const auto& [edge_key, edge_val] : out_edges_ng) {
                auto found_it = std::find(out_edges_ng_ex.begin(), out_edges_ng_ex.end(), std::make_pair(edge_key, edge_val));
                ASSERT_TRUE(found_it != out_edges_ng_ex.end());
            }
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphEdgesRemoval) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<size_t> uid_num_edges(0, upper_bound_num_neurons * num_synapses_per_neuron);

    std::uniform_int_distribution<int> uid_num_ranks(1, num_ranks);
    std::uniform_int_distribution<int> uid_edge_weight(-bound_synapse_weight, bound_synapse_weight);

    for (auto i = 0; i < iterations; i++) {
        size_t number_neurons = uid_num_neurons(mt);
        size_t num_edges = uid_num_edges(mt) + number_neurons;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, number_neurons - 1);

        NetworkGraph ng(number_neurons, 0);

        std::vector<std::tuple<size_t, int, size_t, int, int>> synapses(num_edges);

        for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
            int other_rank = uid_num_ranks(mt);
            size_t neuron_id = uid_actual_num_neurons(mt);
            size_t other_neuron_id = uid_actual_num_neurons(mt);

            int weight = uid_edge_weight(mt);

            while (weight == 0) {
                weight = uid_edge_weight(mt);
            }

            bool is_in_synapse = weight < 0;

            RankNeuronId my_id{ 0, neuron_id };
            RankNeuronId other_id{ other_rank, other_neuron_id };

            if (is_in_synapse) {
                ng.add_edge_weight(my_id, other_id, weight);
                synapses[edge_id] = std::make_tuple(neuron_id, 0, other_neuron_id, other_rank, weight);
            } else {
                ng.add_edge_weight(other_id, my_id, weight);
                synapses[edge_id] = std::make_tuple(other_neuron_id, other_rank, neuron_id, 0, weight);
            }
        }

        std::shuffle(synapses.begin(), synapses.end(), mt);

        for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
            const auto& current_synapse = synapses[edge_id];

            const auto target_neuron_id = std::get<0>(current_synapse);
            const auto target_rank = std::get<1>(current_synapse);
            const auto source_neuron_id = std::get<2>(current_synapse);
            const auto source_rank = std::get<3>(current_synapse);
            const auto weight = std::get<4>(current_synapse);

            RankNeuronId target_id{ target_rank, target_neuron_id };
            RankNeuronId source_id{ source_rank, source_neuron_id };

            ng.add_edge_weight(target_id, source_id, -weight);
        }

        for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            size_t exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);
            size_t inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);
            size_t out_edges_count = ng.get_number_out_edges(neuron_id);

            ASSERT_EQ(exc_in_edges_count, 0);
            ASSERT_EQ(inh_in_edges_count, 0);
            ASSERT_EQ(out_edges_count, 0);

            const NetworkGraph::DistantEdges& in_edges = ng.get_all_in_edges(neuron_id);
            const NetworkGraph::DistantEdges& out_edges = ng.get_all_out_edges(neuron_id);

            ASSERT_EQ(in_edges.size(), 0);
            ASSERT_EQ(out_edges.size(), 0);
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphCreate) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<size_t> uid_num_edges(0, upper_bound_num_neurons * num_synapses_per_neuron);

    std::uniform_int_distribution<int> uid_num_ranks(1, num_ranks);
    std::uniform_int_distribution<int> uid_edge_weight(-bound_synapse_weight, bound_synapse_weight);

    for (auto i = 0; i < iterations; i++) {
        size_t number_neurons = uid_num_neurons(mt);
        size_t num_edges = uid_num_edges(mt) + number_neurons;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, number_neurons - 1);

        NetworkGraph ng(number_neurons, 0);

        std::map<RankNeuronId, std::map<RankNeuronId, int>> in_edges;
        std::map<RankNeuronId, std::map<RankNeuronId, int>> out_edges;

        for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
            int other_rank = uid_num_ranks(mt);
            size_t neuron_id = uid_actual_num_neurons(mt);
            size_t other_neuron_id = uid_actual_num_neurons(mt);

            int weight = uid_edge_weight(mt);

            while (weight == 0) {
                weight = uid_edge_weight(mt);
            }

            bool is_in_synapse = weight < 0;

            RankNeuronId my_id{ 0, neuron_id };
            RankNeuronId other_id{ other_rank, other_neuron_id };

            if (is_in_synapse) {
                ng.add_edge_weight(my_id, other_id, weight);
                in_edges[my_id][other_id] += weight;
            } else {
                ng.add_edge_weight(other_id, my_id, weight);
                out_edges[my_id][other_id] += weight;
            }
        }

        const auto num_new_neurons = uid_num_neurons(mt);
        const auto num_new_edges = uid_num_edges(mt);

        const auto total_number_neurons = number_neurons + num_new_neurons;
        const auto total_num_edges = num_edges + num_new_edges;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons_2(0, total_number_neurons - 1);

        ng.create_neurons(num_new_neurons);

        for (size_t edge_id = num_edges; edge_id < total_num_edges; edge_id++) {
            int other_rank = uid_num_ranks(mt);
            size_t neuron_id = uid_actual_num_neurons(mt);
            size_t other_neuron_id = uid_actual_num_neurons(mt);

            int weight = uid_edge_weight(mt);

            while (weight == 0) {
                weight = uid_edge_weight(mt);
            }

            bool is_in_synapse = weight < 0;

            RankNeuronId my_id{ 0, neuron_id };
            RankNeuronId other_id{ other_rank, other_neuron_id };

            if (is_in_synapse) {
                ng.add_edge_weight(my_id, other_id, weight);
                in_edges[my_id][other_id] += weight;
            } else {
                ng.add_edge_weight(other_id, my_id, weight);
                out_edges[my_id][other_id] += weight;
            }
        }

        for (size_t neuron_id = 0; neuron_id < total_number_neurons; neuron_id++) {
            size_t exc_in_edges_count_ng = ng.get_number_excitatory_in_edges(neuron_id);
            size_t inh_in_edges_count_ng = ng.get_number_inhibitory_in_edges(neuron_id);
            size_t out_edges_count_ng = ng.get_number_out_edges(neuron_id);

            const std::vector<std::pair<RankNeuronId, int>>& in_edges_ng = ng.get_all_in_edges(neuron_id);
            const std::vector<std::pair<RankNeuronId, int>>& out_edges_ng = ng.get_all_out_edges(neuron_id);

            size_t exc_in_edges_count_meta = 0;
            size_t inh_in_edges_count_meta = 0;
            size_t out_edges_count_meta = 0;

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

            for (const auto& it : in_edges[{ 0, neuron_id }]) {
                int weight_meta = it.second;
                RankNeuronId key = it.first;
                auto found_it = std::find(in_edges_ng.begin(), in_edges_ng.end(), std::make_pair(key, weight_meta));
                ASSERT_TRUE(found_it != in_edges_ng.end());
            }

            for (const auto& it : out_edges[{ 0, neuron_id }]) {
                int weight_meta = it.second;
                RankNeuronId key = it.first;
                auto found_it = std::find(out_edges_ng.begin(), out_edges_ng.end(), std::make_pair(key, weight_meta));
                ASSERT_TRUE(found_it != out_edges_ng.end());
            }
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphHistogramPositiveWeight) {
    for (auto i = 0; i < iterations; i++) {
        const auto number_neurons = get_random_number_neurons();
        const auto number_synapses = get_random_number_synapses() + number_neurons;

        const auto& synapses = get_random_synapses(number_neurons, number_synapses);

        NetworkGraph ng(number_neurons, 0);

        std::map<std::pair<size_t, size_t>, int> reduced_synapses{};

        for (const auto& [source_id, target_id, weight] : synapses) {
            if (weight == 0) {
                continue;
            }

            const auto abs_weight = std::abs(weight);

            ng.add_edge_weight({ 0, target_id }, { 0, source_id }, abs_weight);
            reduced_synapses[{ source_id, target_id }] += abs_weight;
        }

        std::map<size_t, int> in_synapses{};
        std::map<size_t, int> out_synapses{};

        for (const auto& [source_target, weight] : reduced_synapses) {
            const auto& [source_id, target_id] = source_target;

            out_synapses[source_id] += weight;
            in_synapses[target_id] += weight;
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
}

TEST_F(NetworkGraphTest, testNetworkGraphHistogram) {
    for (auto i = 0; i < iterations; i++) {
        const auto number_neurons = get_random_number_neurons();
        const auto number_synapses = get_random_number_synapses() + number_neurons;

        const auto& synapses = get_random_synapses(number_neurons, number_synapses);

        NetworkGraph ng(number_neurons, 0);

        std::map<std::pair<size_t, size_t>, int> reduced_synapses{};

        for (const auto& [source_id, target_id, weight] : synapses) {
            if (weight == 0) {
                continue;
            }

            ng.add_edge_weight({ 0, target_id }, { 0, source_id }, weight);
            reduced_synapses[{ source_id, target_id }] += weight;
        }

        std::map<size_t, int> in_synapses{};
        std::map<size_t, int> out_synapses{};

        for (const auto& [source_target, weight] : reduced_synapses) {
            const auto& [source_id, target_id] = source_target;

            out_synapses[source_id] += std::abs(weight);
            in_synapses[target_id] += std::abs(weight);
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
}
