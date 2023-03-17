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
#include "adapter/mpi/MpiRankAdapter.h"

#include "Types.h"
#include "neurons/NetworkGraph.h"
#include "util/TaggedID.h"
#include "neurons/helper/SynapseDeletionRequests.h"

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

    /**
     * @brief Adds for each neurons the specified number of outgoing connections to the network graph on a single rank
     * @param network_graph The network graph to which is modified for the current rank
     * @param signal_types Vector of signl_types on current rank
     * @param number_neurons Number of neurons per rank
     * @param number_outgoing_connections_per_neuron Number of outgoing connection that shall be added to each neuron
     * @param number_ranks Number of neurons per rank
     * @param my_rank Current (simulated) mpi rank
     * @param mt seed
     */
    static void create_dense_network(const std::shared_ptr<NetworkGraph>& network_graph, const std::span<const SignalType>& signal_types,size_t number_neurons, size_t number_outgoing_connections_per_neuron, int number_ranks, const MPIRank& my_rank, std::mt19937& mt) {
        for(auto local_neuron_id=0;local_neuron_id<number_neurons;local_neuron_id++) {
            std::unordered_set<NeuronID> target_neurons;
            for(auto i=0;i<number_outgoing_connections_per_neuron;i++) {
                NeuronID target_neuron_id;
                do {
                    target_neuron_id = TaggedIdAdapter::get_random_neuron_id(number_neurons,
                                                                                        NeuronID(local_neuron_id),
                                                                                        mt);
                } while (target_neurons.contains(target_neuron_id));
                target_neurons.insert(target_neuron_id);
                const auto rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);
                const auto weight = signal_types[local_neuron_id] == SignalType::Excitatory ? 1.0 : -1.0;
                if(rank == my_rank) {
                    network_graph->add_synapse(LocalSynapse(target_neuron_id, NeuronID(local_neuron_id), weight));
                }
                else {
                    network_graph->add_synapse(DistantOutSynapse(RankNeuronId(rank, target_neuron_id), NeuronID(local_neuron_id), weight));
                }
            }
        }
    }

    /**
     * @brief Adds missing distant ingoing connections to the network_graphs if there is a corresponding distant outgoing connections
     * @param network_graphs Vector of network_graphs. Rank i has network_graphs[i]
     * @param num_neurons Number of neurons per rank
     */
    static void harmonize_network_graphs_from_different_ranks(std::vector<std::shared_ptr<NetworkGraph>> network_graphs, const size_t num_neurons) {
        for(int rank=0;rank<network_graphs.size();rank++) {
            const auto cur_network_graph = network_graphs[rank];

            for(auto source_id=0;source_id<num_neurons;source_id++) {
                const auto distant_out_edges = cur_network_graph->get_distant_out_edges(NeuronID{source_id});
                const auto source_rank_id = RankNeuronId{MPIRank(rank), NeuronID(source_id)};

                for(const auto& [target,weight]:distant_out_edges) {
                    const auto target_rank = target.get_rank().get_rank();
                    ASSERT_NE(rank, target_rank );

                    auto& other_network_graph = network_graphs[target_rank];
                    other_network_graph->add_synapse(DistantInSynapse(target.get_neuron_id(),source_rank_id, weight));
                }
            }
        }
    }
/**
 * @brief Checks if the signal_types of synapses corresponds with the weights in the network graph and if the distant outgoing and ingoing edges match
 * @param network_graphs Vector of network_graphs. Rank i has network_graphs[i]
 * @param signal_types Vector of vector of signal types. Neuron j on rank i has signal_type[i][j]
 * @param num_neurons Number of neurons per rank
 */
    static void check_validity_of_network_graphs(std::vector<std::shared_ptr<NetworkGraph>> network_graphs,const std::vector<std::vector<SignalType>>& signal_types, const size_t num_neurons
        ){for(int rank=0;rank<network_graphs.size();rank++) {
            const auto cur_network_graph = network_graphs[rank];

            for(auto neuron_id=0; neuron_id < num_neurons; neuron_id++) {

                const auto& local_out_edges = cur_network_graph->get_local_out_edges(NeuronID(neuron_id));
                for(const auto& [target,weight]:local_out_edges) {
                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    ASSERT_EQ(signal_type, signal_types[rank][neuron_id]);
                }

                const auto distant_out_edges = cur_network_graph->get_distant_out_edges(NeuronID{neuron_id});
                for(const auto& [target,weight]:distant_out_edges) {
                    const auto target_rank = target.get_rank().get_rank();
                    ASSERT_NE(rank, target_rank );
                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    ASSERT_EQ(signal_type, signal_types[rank][neuron_id]);

                    auto& other_network_graph = network_graphs[target_rank];
                    const auto& other_distant_in_edges = other_network_graph->get_distant_in_edges(target.get_neuron_id());
                    bool found_edge = false;
                    for(const auto& [other_source, other_weight]:other_distant_in_edges) {
                        if(other_source.get_rank().get_rank() == rank && other_source.get_neuron_id().get_neuron_id() == neuron_id) {
                            found_edge = true;
                            break;
                        }
                    }
                    ASSERT_TRUE(found_edge);
                }

                const auto& local_in_edges = cur_network_graph->get_local_in_edges(NeuronID(neuron_id));
                for(const auto& [source,weight]:local_in_edges) {
                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    ASSERT_EQ(signal_type, signal_types[rank][source.get_neuron_id()]);
                }

                const auto distant_in_edges = cur_network_graph->get_distant_in_edges(NeuronID{neuron_id});
                for(const auto& [source,weight]:distant_in_edges) {
                    const auto source_rank = source.get_rank().get_rank();
                    ASSERT_NE(rank, source_rank );
                    const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
                    ASSERT_EQ(signal_type, signal_types[source_rank][source.get_neuron_id().get_neuron_id()]);

                    auto& other_network_graph = network_graphs[source_rank];
                    const auto& other_distant_out_edges = other_network_graph->get_distant_out_edges(source.get_neuron_id());
                    bool found_edge = false;
                    for(const auto& [other_target, other_weight]:other_distant_out_edges) {
                        if(other_target.get_rank().get_rank() == rank && other_target.get_neuron_id().get_neuron_id() == neuron_id) {
                            ASSERT_EQ(weight, other_weight);
                            found_edge = true;
                            break;
                        }
                    }
                        ASSERT_TRUE(found_edge);
                }
            }
        }
    }
};
