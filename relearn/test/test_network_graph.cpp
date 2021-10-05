#include "../googletest/include/gtest/gtest.h"

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
        const auto num_neurons = uid_num_neurons(mt);

        NetworkGraph ng(num_neurons);

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);
            const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);
            const auto out_edges_count = ng.get_number_out_edges(neuron_id);

            ASSERT_EQ(exc_in_edges_count, 0);
            ASSERT_EQ(inh_in_edges_count, 0);
            ASSERT_EQ(out_edges_count, 0);

            const auto& local_in_edges = ng.get_local_in_edges(neuron_id);
            const auto& distant_in_edges = ng.get_distant_in_edges(neuron_id);
            const auto& local_out_edges = ng.get_local_out_edges(neuron_id);
            const auto& distant_out_edges = ng.get_distant_out_edges(neuron_id);

            ASSERT_EQ(local_in_edges.size(), 0);
            ASSERT_EQ(distant_in_edges.size(), 0);
            ASSERT_EQ(local_out_edges.size(), 0);
            ASSERT_EQ(distant_out_edges.size(), 0);

            const auto& all_in_edges_excitatory = ng.get_all_in_edges(neuron_id, SignalType::EXCITATORY);
            const auto& all_in_edges_inhibitory = ng.get_all_in_edges(neuron_id, SignalType::INHIBITORY);
            const auto& all_out_edges_excitatory = ng.get_all_out_edges(neuron_id, SignalType::EXCITATORY);
            const auto& all_out_edges_inhibitory = ng.get_all_out_edges(neuron_id, SignalType::INHIBITORY);

            ASSERT_EQ(all_in_edges_excitatory.size(), 0);
            ASSERT_EQ(all_in_edges_inhibitory.size(), 0);
            ASSERT_EQ(all_out_edges_excitatory.size(), 0);
            ASSERT_EQ(all_out_edges_inhibitory.size(), 0);

            const auto& in_edges = ng.get_all_in_edges(neuron_id);
            const auto& out_edges = ng.get_all_out_edges(neuron_id);

            ASSERT_EQ(in_edges.size(), 0);
            ASSERT_EQ(out_edges.size(), 0);
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphConstructorExceptions) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);

        NetworkGraph ng(num_neurons);

        for (size_t neuron_id = num_neurons; neuron_id < num_neurons + num_neurons; neuron_id++) {
            ASSERT_THROW(const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);, RelearnException);
            ASSERT_THROW(const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);, RelearnException);
            ASSERT_THROW(const auto out_edges_count = ng.get_number_out_edges(neuron_id);, RelearnException);

            ASSERT_THROW(const auto& local_in_edges = ng.get_local_in_edges(neuron_id);, RelearnException);
            ASSERT_THROW(const auto& distant_in_edges = ng.get_distant_in_edges(neuron_id);, RelearnException);
            ASSERT_THROW(const auto& local_out_edges = ng.get_local_out_edges(neuron_id);, RelearnException);
            ASSERT_THROW(const auto& distant_out_edges = ng.get_distant_out_edges(neuron_id);, RelearnException);

            ASSERT_THROW(const auto& all_in_edges_excitatory = ng.get_all_in_edges(neuron_id, SignalType::EXCITATORY);, RelearnException);
            ASSERT_THROW(const auto& all_in_edges_inhibitory = ng.get_all_in_edges(neuron_id, SignalType::INHIBITORY);, RelearnException);
            ASSERT_THROW(const auto& all_out_edges_excitatory = ng.get_all_out_edges(neuron_id, SignalType::EXCITATORY);, RelearnException);
            ASSERT_THROW(const auto& all_out_edges_inhibitory = ng.get_all_out_edges(neuron_id, SignalType::INHIBITORY);, RelearnException);
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphCreateNeurons) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);

    for (auto i = 0; i < iterations; i++) {
        const auto initial_num_neurons = uid_num_neurons(mt);
        NetworkGraph ng(initial_num_neurons);

        const auto new_neurons = uid_num_neurons(mt);
        ng.create_neurons(new_neurons);

        const auto num_neurons = initial_num_neurons + new_neurons;

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);
            const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);
            const auto out_edges_count = ng.get_number_out_edges(neuron_id);

            ASSERT_EQ(exc_in_edges_count, 0);
            ASSERT_EQ(inh_in_edges_count, 0);
            ASSERT_EQ(out_edges_count, 0);

            const auto& local_in_edges = ng.get_local_in_edges(neuron_id);
            const auto& distant_in_edges = ng.get_distant_in_edges(neuron_id);
            const auto& local_out_edges = ng.get_local_out_edges(neuron_id);
            const auto& distant_out_edges = ng.get_distant_out_edges(neuron_id);

            ASSERT_EQ(local_in_edges.size(), 0);
            ASSERT_EQ(distant_in_edges.size(), 0);
            ASSERT_EQ(local_out_edges.size(), 0);
            ASSERT_EQ(distant_out_edges.size(), 0);

            const auto& all_in_edges_excitatory = ng.get_all_in_edges(neuron_id, SignalType::EXCITATORY);
            const auto& all_in_edges_inhibitory = ng.get_all_in_edges(neuron_id, SignalType::INHIBITORY);
            const auto& all_out_edges_excitatory = ng.get_all_out_edges(neuron_id, SignalType::EXCITATORY);
            const auto& all_out_edges_inhibitory = ng.get_all_out_edges(neuron_id, SignalType::INHIBITORY);

            ASSERT_EQ(all_in_edges_excitatory.size(), 0);
            ASSERT_EQ(all_in_edges_inhibitory.size(), 0);
            ASSERT_EQ(all_out_edges_excitatory.size(), 0);
            ASSERT_EQ(all_out_edges_inhibitory.size(), 0);

            const auto& in_edges = ng.get_all_in_edges(neuron_id);
            const auto& out_edges = ng.get_all_out_edges(neuron_id);

            ASSERT_EQ(in_edges.size(), 0);
            ASSERT_EQ(out_edges.size(), 0);
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphCreateNeuronsException) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);

    for (auto i = 0; i < iterations; i++) {
        const auto initial_num_neurons = uid_num_neurons(mt);
        NetworkGraph ng(initial_num_neurons);

        const auto new_neurons = uid_num_neurons(mt);
        ng.create_neurons(new_neurons);

        const auto num_neurons = initial_num_neurons + new_neurons;

        for (size_t neuron_id = num_neurons; neuron_id < num_neurons + num_neurons; neuron_id++) {
            ASSERT_THROW(const auto exc_in_edges_count = ng.get_number_excitatory_in_edges(neuron_id);, RelearnException);
            ASSERT_THROW(const auto inh_in_edges_count = ng.get_number_inhibitory_in_edges(neuron_id);, RelearnException);
            ASSERT_THROW(const auto out_edges_count = ng.get_number_out_edges(neuron_id);, RelearnException);

            ASSERT_THROW(const auto& local_in_edges = ng.get_local_in_edges(neuron_id);, RelearnException);
            ASSERT_THROW(const auto& distant_in_edges = ng.get_distant_in_edges(neuron_id);, RelearnException);
            ASSERT_THROW(const auto& local_out_edges = ng.get_local_out_edges(neuron_id);, RelearnException);
            ASSERT_THROW(const auto& distant_out_edges = ng.get_distant_out_edges(neuron_id);, RelearnException);

            ASSERT_THROW(const auto& all_in_edges_excitatory = ng.get_all_in_edges(neuron_id, SignalType::EXCITATORY);, RelearnException);
            ASSERT_THROW(const auto& all_in_edges_inhibitory = ng.get_all_in_edges(neuron_id, SignalType::INHIBITORY);, RelearnException);
            ASSERT_THROW(const auto& all_out_edges_excitatory = ng.get_all_out_edges(neuron_id, SignalType::EXCITATORY);, RelearnException);
            ASSERT_THROW(const auto& all_out_edges_inhibitory = ng.get_all_out_edges(neuron_id, SignalType::INHIBITORY);, RelearnException);

            ASSERT_THROW(const auto& in_edges = ng.get_all_in_edges(neuron_id);, RelearnException);
            ASSERT_THROW(const auto& out_edges = ng.get_all_out_edges(neuron_id);, RelearnException);
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphLocalEdges) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<size_t> uid_num_synapses(0, upper_bound_num_neurons * num_synapses_per_neuron);
    std::uniform_int_distribution<int> uid_edge_weight(-bound_synapse_weight, bound_synapse_weight);

    const auto my_rank = MPIWrapper::get_my_rank();

    for (auto i = 0; i < iterations; i++) {
        const auto num_neurons = uid_num_neurons(mt);
        size_t num_synapses = uid_num_synapses(mt) + num_neurons;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, num_neurons - 1);

        NetworkGraph ng(num_neurons);

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

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
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

TEST_F(NetworkGraphTest, testNetworkGraphEdges) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<size_t> uid_num_edges(0, upper_bound_num_neurons * num_synapses_per_neuron);

    std::uniform_int_distribution<int> uid_num_ranks(1, num_ranks);
    std::uniform_int_distribution<int> uid_edge_weight(-bound_synapse_weight, bound_synapse_weight);

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);
        size_t num_edges = uid_num_edges(mt) + num_neurons;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, num_neurons - 1);

        NetworkGraph ng(num_neurons);

        std::map<size_t, std::map<RankNeuronId, int>> in_edges;
        std::map<size_t, std::map<RankNeuronId, int>> out_edges;

        for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
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

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
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
        size_t num_neurons = uid_num_neurons(mt);
        size_t num_edges = uid_num_edges(mt) + num_neurons;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, num_neurons - 1);

        NetworkGraph ng(num_neurons);

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

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
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
        size_t num_neurons = uid_num_neurons(mt);
        size_t num_edges = uid_num_edges(mt) + num_neurons;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, num_neurons - 1);

        NetworkGraph ng(num_neurons);

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

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
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
        size_t num_neurons = uid_num_neurons(mt);
        size_t num_edges = uid_num_edges(mt) + num_neurons;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, num_neurons - 1);

        NetworkGraph ng(num_neurons);

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

        const auto total_num_neurons = num_neurons + num_new_neurons;
        const auto total_num_edges = num_edges + num_new_edges;

        std::uniform_int_distribution<size_t> uid_actual_num_neurons_2(0, total_num_neurons - 1);

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

        for (size_t neuron_id = 0; neuron_id < total_num_neurons; neuron_id++) {
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
