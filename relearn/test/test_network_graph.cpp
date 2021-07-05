#include "../googletest/include/gtest/gtest.h"

#include <map>
#include <random>
#include <tuple>
#include <vector>

#include "RelearnTest.hpp"

#include "../source/neurons/NetworkGraph.h"
#include "../source/util/RelearnException.h"

constexpr size_t upper_bound_num_neurons = 10000;
constexpr int bound_synapse_weight = 10;
constexpr int num_ranks = 17;
constexpr int num_synapses_per_neuron = 2;

TEST_F(NetworkGraphTest, testNetworkGraphConstructor) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);

        NetworkGraph ng(num_neurons);

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            size_t exc_in_edges_count = ng.get_num_in_edges_ex(neuron_id);
            size_t inh_in_edges_count = ng.get_num_in_edges_in(neuron_id);
            size_t out_edges_count = ng.get_num_out_edges(neuron_id);

            ASSERT_EQ(exc_in_edges_count, 0);
            ASSERT_EQ(inh_in_edges_count, 0);
            ASSERT_EQ(out_edges_count, 0);

            const NetworkGraph::Edges& in_edges = ng.get_in_edges(neuron_id);
            const NetworkGraph::Edges& out_edges = ng.get_out_edges(neuron_id);

            ASSERT_EQ(in_edges.size(), 0);
            ASSERT_EQ(out_edges.size(), 0);
        }
    }
}

TEST_F(NetworkGraphTest, testNetworkGraphConstructorExceptions) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<int> uid_num_ranks(1, num_ranks);

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);

        NetworkGraph ng(num_neurons);

        for (size_t neuron_id = num_neurons; neuron_id < num_neurons + num_neurons; neuron_id++) {
            ASSERT_THROW(size_t exc_in_edges_count = ng.get_num_in_edges_ex(neuron_id), RelearnException);
            ASSERT_THROW(size_t inh_in_edges_count = ng.get_num_in_edges_in(neuron_id), RelearnException);
            ASSERT_THROW(size_t out_edges_count = ng.get_num_out_edges(neuron_id), RelearnException);

            ASSERT_THROW(const NetworkGraph::Edges& in_edges = ng.get_in_edges(neuron_id), RelearnException);
            ASSERT_THROW(const NetworkGraph::Edges& out_edges = ng.get_out_edges(neuron_id), RelearnException);
        }

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            size_t other_neuron_id = uid_num_neurons(mt);

            int other_rank = uid_num_ranks(mt);
            RankNeuronId my_id{ 0, neuron_id };
            RankNeuronId other_id{ other_rank, other_neuron_id };

            ASSERT_THROW(ng.add_edge_weight(my_id, other_id, 0), RelearnException);
            ASSERT_THROW(ng.add_edge_weight(other_id, my_id, 0), RelearnException);
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

        std::map<size_t, std::map<std::pair<int, size_t>, int>> in_edges;
        std::map<size_t, std::map<std::pair<int, size_t>, int>> out_edges;

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
            size_t exc_in_edges_count_ng = ng.get_num_in_edges_ex(neuron_id);
            size_t inh_in_edges_count_ng = ng.get_num_in_edges_in(neuron_id);
            size_t out_edges_count_ng = ng.get_num_out_edges(neuron_id);

            const std::vector<std::pair<std::pair<int, size_t>, int>>& in_edges_ng = ng.get_in_edges(neuron_id);
            const std::vector<std::pair<std::pair<int, size_t>, int>>& out_edges_ng = ng.get_out_edges(neuron_id);

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
                std::pair<int, size_t> key = it.first;
                auto found_it = std::find(in_edges_ng.begin(), in_edges_ng.end(), std::make_pair(key, weight_meta));
                ASSERT_TRUE(found_it != in_edges_ng.end());
            }

            for (const auto& it : out_edges[neuron_id]) {
                int weight_meta = it.second;
                std::pair<int, size_t> key = it.first;
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
            size_t exc_in_edges_count_ng = ng.get_num_in_edges_ex(neuron_id);
            size_t inh_in_edges_count_ng = ng.get_num_in_edges_in(neuron_id);
            size_t out_edges_count_ng = ng.get_num_out_edges(neuron_id);

            const std::vector<std::pair<std::pair<int, size_t>, int>>& in_edges_ng = ng.get_in_edges(neuron_id);
            const std::vector<std::pair<std::pair<int, size_t>, int>>& out_edges_ng = ng.get_out_edges(neuron_id);

            std::vector<std::pair<std::pair<int, size_t>, int>> in_edges_ng_ex = ng.get_in_edges(neuron_id, SignalType::EXCITATORY);
            std::vector<std::pair<std::pair<int, size_t>, int>> in_edges_ng_in = ng.get_in_edges(neuron_id, SignalType::INHIBITORY);
            std::vector<std::pair<std::pair<int, size_t>, int>> out_edges_ng_ex = ng.get_out_edges(neuron_id, SignalType::EXCITATORY);
            std::vector<std::pair<std::pair<int, size_t>, int>> out_edges_ng_in = ng.get_out_edges(neuron_id, SignalType::INHIBITORY);

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
            size_t exc_in_edges_count = ng.get_num_in_edges_ex(neuron_id);
            size_t inh_in_edges_count = ng.get_num_in_edges_in(neuron_id);
            size_t out_edges_count = ng.get_num_out_edges(neuron_id);

            ASSERT_EQ(exc_in_edges_count, 0);
            ASSERT_EQ(inh_in_edges_count, 0);
            ASSERT_EQ(out_edges_count, 0);

            const NetworkGraph::Edges& in_edges = ng.get_in_edges(neuron_id);
            const NetworkGraph::Edges& out_edges = ng.get_out_edges(neuron_id);

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
            size_t exc_in_edges_count_ng = ng.get_num_in_edges_ex(neuron_id);
            size_t inh_in_edges_count_ng = ng.get_num_in_edges_in(neuron_id);
            size_t out_edges_count_ng = ng.get_num_out_edges(neuron_id);

            const std::vector<std::pair<std::pair<int, size_t>, int>>& in_edges_ng = ng.get_in_edges(neuron_id);
            const std::vector<std::pair<std::pair<int, size_t>, int>>& out_edges_ng = ng.get_out_edges(neuron_id);

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
                auto found_it = std::find(in_edges_ng.begin(), in_edges_ng.end(), std::make_pair(std::make_pair(key.get_rank(), key.get_neuron_id()), weight_meta));
                ASSERT_TRUE(found_it != in_edges_ng.end());
            }

            for (const auto& it : out_edges[{ 0, neuron_id }]) {
                int weight_meta = it.second;
                RankNeuronId key = it.first;
                auto found_it = std::find(out_edges_ng.begin(), out_edges_ng.end(), std::make_pair(std::make_pair(key.get_rank(), key.get_neuron_id()), weight_meta));
                ASSERT_TRUE(found_it != out_edges_ng.end());
            }
        }
    }
}
