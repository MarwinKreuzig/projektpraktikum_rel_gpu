#include "../googletest/include/gtest/gtest.h"

#include <map>
#include <random>
#include <tuple>
#include <vector>

#include "commons.h"

#include "../source/NetworkGraph.h"
#include "../source/RelearnException.h"

constexpr const size_t upper_bound_num_neurons = 10000;
constexpr const int bound_synapse_weight = 10;

TEST(TestNetworkGraph, testNetworkGraphConstructor) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);

        NetworkGraph ng(num_neurons);

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            size_t exc_in_edges_count = ng.get_num_in_edges_ex(neuron_id);
            size_t inh_in_edges_count = ng.get_num_in_edges_in(neuron_id);
            size_t out_edges_count = ng.get_num_out_edges(neuron_id);

            EXPECT_EQ(exc_in_edges_count, 0);
            EXPECT_EQ(inh_in_edges_count, 0);
            EXPECT_EQ(out_edges_count, 0);

            const NetworkGraph::Edges& in_edges = ng.get_in_edges(neuron_id);
            const NetworkGraph::Edges& out_edges = ng.get_out_edges(neuron_id);

            EXPECT_EQ(in_edges.size(), 0);
            EXPECT_EQ(out_edges.size(), 0);
        }
    }
}

TEST(TestNetworkGraph, testNetworkGraphConstructorExceptions) {
    setup();

    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<int> uid_num_ranks(1, 17);

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);

        NetworkGraph ng(num_neurons);

        for (size_t neuron_id = num_neurons; neuron_id < num_neurons + num_neurons; neuron_id++) {
            EXPECT_THROW(size_t exc_in_edges_count = ng.get_num_in_edges_ex(neuron_id), RelearnException);
            EXPECT_THROW(size_t inh_in_edges_count = ng.get_num_in_edges_in(neuron_id), RelearnException);
            EXPECT_THROW(size_t out_edges_count = ng.get_num_out_edges(neuron_id), RelearnException);

            EXPECT_THROW(const NetworkGraph::Edges& in_edges = ng.get_in_edges(neuron_id), RelearnException);
            EXPECT_THROW(const NetworkGraph::Edges& out_edges = ng.get_out_edges(neuron_id), RelearnException);
        }

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            size_t other_neuron = uid_num_neurons(mt);
            int other_rank = uid_num_ranks(mt);

            EXPECT_THROW(ng.add_edge_weight(neuron_id, 0, other_neuron, other_rank, 0), RelearnException);
            EXPECT_THROW(ng.add_edge_weight(other_neuron, other_rank, neuron_id, 0, 0), RelearnException);
        }
    }
}

TEST(TestNetworkGraph, testNetworkGraphEdges) {
    setup();

    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<size_t> uid_num_edges(0, upper_bound_num_neurons * 2);

    std::uniform_int_distribution<int> uid_num_ranks(1, 17);
    std::uniform_int_distribution<int> uid_edge_weight(-bound_synapse_weight, bound_synapse_weight);

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);
        size_t num_edges = uid_num_edges(mt);

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, num_neurons - 1);

        NetworkGraph ng(num_neurons);

        std::map<size_t, std::map<std::pair<int, size_t>, int>> in_edges;
        std::map<size_t, std::map<std::pair<int, size_t>, int>> out_edges;

        for (size_t edge_id = 0; edge_id < num_neurons; edge_id++) {
            int other_rank = uid_num_ranks(mt);
            size_t my_neuron_id = uid_actual_num_neurons(mt);
            size_t other_neuron_id = uid_actual_num_neurons(mt);

            int weight = uid_edge_weight(mt);

            while (weight == 0) {
                weight = uid_edge_weight(mt);
            }

            bool is_in_synapse = weight < 0;

            if (is_in_synapse) {
                ng.add_edge_weight(my_neuron_id, 0, other_neuron_id, other_rank, weight);
                in_edges[my_neuron_id][{ other_rank, other_neuron_id }] += weight;

                if (in_edges[my_neuron_id][{ other_rank, other_neuron_id }] == 0) {
                    in_edges[my_neuron_id].erase({ other_rank, other_neuron_id });
                }
            } else {
                ng.add_edge_weight(other_neuron_id, other_rank, my_neuron_id, 0, weight);
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

            EXPECT_EQ(exc_in_edges_count_ng, exc_in_edges_count_meta);
            EXPECT_EQ(inh_in_edges_count_ng, inh_in_edges_count_meta);
            EXPECT_EQ(out_edges_count_ng, out_edges_count_meta);

            for (const auto& it : in_edges[neuron_id]) {
                int weight_meta = it.second;
                std::pair<int, size_t> key = it.first;
                auto it = std::find(in_edges_ng.begin(), in_edges_ng.end(), std::make_pair(key, weight_meta));
                EXPECT_TRUE(it != in_edges_ng.end());
            }

            for (const auto& it : out_edges[neuron_id]) {
                int weight_meta = it.second;
                std::pair<int, size_t> key = it.first;
                auto it = std::find(out_edges_ng.begin(), out_edges_ng.end(), std::make_pair(key, weight_meta));
                EXPECT_TRUE(it != out_edges_ng.end());
            }
        }
    }
}

TEST(TestNetworkGraph, testNetworkGraphEdgesSplit) {
    setup();

    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<size_t> uid_num_edges(0, upper_bound_num_neurons * 2);

    std::uniform_int_distribution<int> uid_num_ranks(1, 17);
    std::uniform_int_distribution<int> uid_edge_weight(-bound_synapse_weight, bound_synapse_weight);

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);
        size_t num_edges = uid_num_edges(mt);

        std::uniform_int_distribution<size_t> uid_actual_num_neurons(0, num_neurons - 1);

        NetworkGraph ng(num_neurons);

        for (size_t edge_id = 0; edge_id < num_neurons; edge_id++) {
            int other_rank = uid_num_ranks(mt);
            size_t my_neuron_id = uid_actual_num_neurons(mt);
            size_t other_neuron_id = uid_actual_num_neurons(mt);

            int weight = uid_edge_weight(mt);

            while (weight == 0) {
                weight = uid_edge_weight(mt);
            }

            bool is_in_synapse = weight < 0;

            if (is_in_synapse) {
                ng.add_edge_weight(my_neuron_id, 0, other_neuron_id, other_rank, weight);
            } else {
                ng.add_edge_weight(other_neuron_id, other_rank, my_neuron_id, 0, weight);
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

            EXPECT_EQ(in_edges_ng.size(), in_edges_ng_ex.size() + in_edges_ng_in.size());
            EXPECT_EQ(out_edges_ng.size(), out_edges_ng_ex.size() + out_edges_ng_in.size());

            for (const auto& [edge_key, edge_val] : in_edges_ng) {
                EXPECT_TRUE(edge_val < 0);
            }

            for (const auto& [edge_key, edge_val] : out_edges_ng_in) {
                EXPECT_TRUE(edge_val < 0);
            }

            for (const auto& [edge_key, edge_val] : in_edges_ng_ex) {
                EXPECT_TRUE(edge_val > 0);
            }

            for (const auto& [edge_key, edge_val] : out_edges_ng_ex) {
                EXPECT_TRUE(edge_val > 0);
            }

            for (const auto& val : in_edges_ng_in) {
                in_edges_ng_ex.emplace_back(val);
            }

            for (const auto& val : out_edges_ng_ex) {
                out_edges_ng_in.emplace_back(val);
            }

            //in_edges_ng_ex.merge(in_edges_ng_in);
            //out_edges_ng_ex.merge(out_edges_ng_in);

            EXPECT_EQ(in_edges_ng.size(), in_edges_ng_ex.size());
            EXPECT_EQ(out_edges_ng.size(), out_edges_ng_ex.size());

            for (const auto& [edge_key, edge_val] : in_edges_ng) {
                auto it = std::find(in_edges_ng_ex.begin(), in_edges_ng_ex.end(), std::make_pair(edge_key, edge_val));
                EXPECT_TRUE(it != in_edges_ng_ex.end());
            }

            for (const auto& [edge_key, edge_val] : out_edges_ng) {
                auto it = std::find(out_edges_ng_ex.begin(), out_edges_ng_ex.end(), std::make_pair(edge_key, edge_val));
                EXPECT_TRUE(it != out_edges_ng_ex.end());
            }
        }
    }
}
