/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include "../Config.h"
#include "../mpi/MPIWrapper.h"
#include "../util/Vec3.h"
#include "../util/RelearnException.h"
#include "helper/RankNeuronId.h"
#include "SignalType.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

class NeuronsExtraInfo;
class Partition;

/**
  * An object of type NetworkGraph stores the synaptic connections between neurons, that are relevant for the current MPI rank.
  * The neurons are refered to by indices in the range [0, num_local_neurons).
  * The class does not perform any communication or synchronization with other MPI ranks.
  */
class NetworkGraph {
public:
    /**
	 * Type definitions
	 */
    using EdgesKey = std::pair<int, size_t>; // Pair of (rank, neuron id)
    using EdgesVal = int;
    using Edges = std::vector<std::pair<EdgesKey, EdgesVal>>; // Map of neuron id to edge weight

    using NeuronInNeighborhood = std::vector<Edges>;
    using NeuronOutNeighborhood = std::vector<Edges>;

    /**
     * @brief Constructs an object that has enough space to store the given number of neurons
     * @param num_neurons The number of neurons that the object shall handle
     */
    explicit NetworkGraph(const size_t num_neurons)
        : neuron_in_neighborhood(num_neurons)
        , neuron_out_neighborhood(num_neurons)
        , my_num_neurons(num_neurons) {
    }

    /**
     * @brief Returns a constant reference to all in-edges to a neuron, i.e., a view on all neurons that connect to the specified one via a synapse
     * @param neuron_id The id of the neuron
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all in-edges that is invalidated when the state of the object changes
     */
    [[nodiscard]] const Edges& get_in_edges(const size_t neuron_id) const {
        RelearnException::check(neuron_id < neuron_in_neighborhood.size(), "NetworkGraph::get_in_edges: Tried with a too large id of {}", neuron_id);
        return neuron_in_neighborhood[neuron_id];
    }

    /**
     * @brief Returns a constant reference to all out-edges to a neuron, i.e., a view on all neurons that the specified one connectes to via a synapse
     * @param neuron_id The id of the neuron
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all out-edges that is invalidated when the state of the object changes
     */
    [[nodiscard]] const Edges& get_out_edges(const size_t neuron_id) const {
        RelearnException::check(neuron_id < neuron_out_neighborhood.size(), "NetworkGraph::get_out_edges: Tried with a too large id of {}", neuron_id);
        return neuron_out_neighborhood[neuron_id];
    }

    /**
     * @brief Returns a copy of all in-edges to a neuron, i.e., a copy of all neurons that connect to the specified one via a synapse, of a specified type
     * @param neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     * @return A copy of all in-edges from a certain neuron signal type
     */
    [[nodiscard]] Edges get_in_edges(const size_t neuron_id, const SignalType signal_type) const {
        const Edges& all_edges = get_in_edges(neuron_id);

        Edges filtered_edges{};
        filtered_edges.reserve(all_edges.size());

        for (const auto& [edge_key, edge_val] : all_edges) {
            if (signal_type == SignalType::EXCITATORY && edge_val > 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }

            if (signal_type == SignalType::INHIBITORY && edge_val < 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }
        }

        return filtered_edges;
    }

    /**
     * @brief Returns a copy of all out-edges from a neuron, i.e., a copy of all neurons that the specified one connectes to via a synapse, of a specified type
     * @param neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     * @return A copy of all out-edges to a certain neuron signal type
     */
    [[nodiscard]] Edges get_out_edges(const size_t neuron_id, const SignalType signal_type) const {
        const Edges& all_edges = get_out_edges(neuron_id);

        Edges filtered_edges{};
        filtered_edges.reserve(all_edges.size());

        for (const auto& [edge_key, edge_val] : all_edges) {
            if (signal_type == SignalType::EXCITATORY && edge_val > 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }

            if (signal_type == SignalType::INHIBITORY && edge_val < 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }
        }

        return filtered_edges;
    }

    /**
     * @brief Returns the number of all in-edges to a neuron (countings multiplicities) from excitatory neurons
     * @param neuron_id The id of the neuron
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     * @return The number of incoming synapses that the specified neuron formed from excitatory neurons
     */
    [[nodiscard]] size_t get_num_in_edges_ex(const size_t neuron_id) const {
        RelearnException::check(neuron_id < neuron_in_neighborhood.size(),
            "NetworkGraph::get_num_in_edges_ex: Tried with a too large id: {} {}", neuron_id, my_num_neurons);

        size_t total_num_ports = 0;

        for (const auto& [_, connection_strength] : neuron_in_neighborhood[neuron_id]) {
            if (connection_strength > 0) {
                total_num_ports += connection_strength;
            }
        }

        return total_num_ports;
    }

    /**
     * @brief Returns the number of all in-edges to a neuron (countings multiplicities) from inhibitory neurons
     * @param neuron_id The id of the neuron
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     * @return The number of incoming synapses that the specified neuron formed from inhibitory neurons
     */
    [[nodiscard]] size_t get_num_in_edges_in(const size_t neuron_id) const {
        RelearnException::check(neuron_id < neuron_in_neighborhood.size(),
            "NetworkGraph::get_num_in_edges_in: Tried with a too large id: {} {}", neuron_id, my_num_neurons);

        size_t total_num_ports = 0;

        for (const auto& [_, connection_strength] : neuron_in_neighborhood[neuron_id]) {
            if (connection_strength < 0) {
                total_num_ports += -connection_strength;
            }
        }

        return total_num_ports;
    }

    /**
     * @brief Returns the number of all out-edges from a neuron (countings multiplicities)
     * @param neuron_id The id of the neuron
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     * @return The number of outgoing synapses that the specified neuron formed
     */
    [[nodiscard]] size_t get_num_out_edges(const size_t neuron_id) const {
        RelearnException::check(neuron_id < neuron_out_neighborhood.size(),
            "NetworkGraph::get_num_out_edges: Tried with a too large id: {} {}", neuron_id, my_num_neurons);

        size_t total_num_ports = 0;

        for (const auto& [_, connection_strength] : neuron_out_neighborhood[neuron_id]) {
            total_num_ports += std::abs(connection_strength);
        }

        return total_num_ports;
    }

    /**
     * @brief Resizes the network graph by adding space for more neurons. Invalidates iterators
     * @param creation_count The number of additional neurons the network graph should handle
     * @exception Throws an exception if the allocation of memory fails
     */
    void create_neurons(const size_t creation_count) {
        const auto old_size = my_num_neurons;
        const auto new_size = old_size + creation_count;

        neuron_in_neighborhood.resize(new_size);
        neuron_out_neighborhood.resize(new_size);

        my_num_neurons = new_size;
    }

    /**
     * @brief Adds the specified weight to the synapse from the neuron specified by source to the neuron specified by target.
     * If there was no edge before, it is created. If the updated weight is 0, it is deleted. Only updates the local part of the network graph.
	 * @param target The target neuron's id and rank
     * @param source The source neuron's id and rank
     * @param weight The weight that should be added onto the current connections, not zero
     * @exception Throws a RelearnException if: 
     * (a) the weight is zero, 
     * (b) neither the target nor the source are on the current rank,
     * (c) a local neuron id is larger than the number of neurons
	 */
    void add_edge_weight(const RankNeuronId& target, const RankNeuronId& source, const int weight) {
        RelearnException::check(weight != 0, "NetworkGraph::add_edge_weight: weight of edge to add is zero");

        const auto target_rank = target.get_rank();
        const auto target_neuron_id = target.get_neuron_id();

        const auto source_rank = source.get_rank();
        const auto source_neuron_id = source.get_neuron_id();

        const auto my_rank = MPIWrapper::get_my_rank();

        // Target neuron is mine
        if (target_rank == my_rank) {
            RelearnException::check(target_neuron_id < my_num_neurons,
                "NetworkGraph::add_edge_weight: Want to add an in-edge with a too large target id: {} {}", target_neuron_id, my_num_neurons);

            Edges& in_edges = neuron_in_neighborhood[target_neuron_id];
            add_edge(in_edges, source_rank, source_neuron_id, weight);
        }

        if (source_rank == my_rank) {
            RelearnException::check(source_neuron_id < my_num_neurons,
                "NetworkGraph::add_edge_weight: Want to add an out-edge with a too large source id: {} {}", target_neuron_id, my_num_neurons);

            Edges& out_edges = neuron_out_neighborhood[source_neuron_id];
            add_edge(out_edges, target_rank, target_neuron_id, weight);
        }

        if (target_rank != my_rank && source_rank != my_rank) {
            RelearnException::fail("NetworkGraph::add_edge_weight: In NetworkGraph::add_edge_weight, neither the target nor the source rank were for me.");
        }
    }

    /**
     * @brief Prints all stored connections to the out-stream. Uses the global neuron ids and starts with 1. The format is <target id> <source id> <weight>
     * @param os The out-stream to which the network graph is printed
     * @param informations The NeuronsExtraInfo that is used to translate between local neuron id and global neuron id
     * @exception Throws a RelearnException if the translation of a neuron id fails
     */
    void print(std::ostream& os, const std::unique_ptr<NeuronsExtraInfo>& informations) const;

    /**
     * @brief Loads all edges from the file that are relevant for the local network graph.
     * @param path_synapses The path to the file in which the synapses are stored (with the global neuron ids starting at 1)
     * @param path_neurons The path to the file in which the neurons are stored (with the global neuron ids starting at 1 and their positions)
     * @param partition The Partition object that is used to determine which neurons are local
     * @exception Throws a RelearnException if (a) the parsing of the files failed, (b) the network graph was not initialized with enough storage space
     */
    void add_edges_from_file(const std::string& path_synapses, const std::string& path_neurons, const Partition& partition);

    /**
     * @brief Checks if the specified file contains only synapses between neurons with specified ids (only works locally).
     * @param path_synapses The path to the file in which the synapses are stored (with the global neuron ids starting at 1)
     * @param neuron_ids The neuron ids between which the synapses should be formed. Must be sorted ascendingly
     * @return Returns true iff the file has the correct format and only ids in neuron_ids are present
     */
    [[nodiscard]] static bool check_edges_from_file(const std::string& path_synapses, const std::vector<size_t>& neuron_ids);

    /**
     * @brief Performs a debug check on the local portion of the network graph. All stored ranks must be greater or equal to zero, no weight must be equal to zero, and all purely local edges must have a matching counterpart.
     * @exception Throws a RelearnException if any of the conditions is violated
     */
    void debug_check() const;

    std::vector<unsigned int> get_in_edges_histogram() const noexcept {
        std::vector<unsigned int> result{};

        auto latest_result = 0;

        for (const auto& in_neighborhood : neuron_in_neighborhood) {
            auto sum = 0;

            for (const auto& [_, val] : in_neighborhood) {
                if (val < 0) {
                    sum -= val;
                } else {
                    sum += val;
                }
            }

            if (result.size() <= sum) {
                result.resize(sum * 2 + 1);
                latest_result = sum;
            }

            result[sum]++;
        }

        result.resize(latest_result + 1);
        return result;
    }

    std::vector<unsigned int> get_out_edges_histogram() const noexcept {
        std::vector<unsigned int> result{};

        auto latest_result = 0;

        for (const auto& in_neighborhood : neuron_out_neighborhood) {
            auto sum = 0;

            for (const auto& [_, val] : in_neighborhood) {
                if (val < 0) {
                    sum -= val;
                } else {
                    sum += val;
                }
            }

            if (result.size() <= sum) {
                result.resize(sum * 2 + 1);
                latest_result = sum;
            }

            result[sum]++;
        }

        result.resize(latest_result + 1);
        return result;
    }

private:
    // NOLINTNEXTLINE
    static void add_edge(Edges& edges, const int rank, const size_t neuron_id, const int weight) {
        const EdgesKey rank_neuron_id_pair{ rank, neuron_id };

        size_t idx = 0;

        for (auto& [key, val] : edges) {
            if (key == rank_neuron_id_pair) {
                const int sum = val + weight;
                val = sum;

                if (sum == 0) {
                    const auto idx_last = edges.size() - 1;
                    std::swap(edges[idx], edges[idx_last]);
                    edges.erase(edges.cend() - 1);
                }

                return;
            }

            idx++;
        }

        edges.emplace_back(rank_neuron_id_pair, weight);
    }

    // NOLINTNEXTLINE
    static void translate_global_to_local(const std::map<size_t, int>& id_to_rank, const Partition& partition, std::map<size_t, size_t>& global_id_to_local_id);

    // NOLINTNEXTLINE
    static void load_neuron_positions(const std::string& path_neurons, std::set<size_t>& foreing_ids, std::map<size_t, Vec3d>& id_to_pos);

    // NOLINTNEXTLINE
    static void load_synapses(const std::string& path_synapses, const Partition& partition, std::set<size_t>& foreing_ids, std::vector<std::tuple<size_t, size_t, int>>& local_synapses, std::vector<std::tuple<size_t, size_t, int>>& out_synapses, std::vector<std::tuple<size_t, size_t, int>>& in_synapses);

    void write_synapses_to_file(const std::string& filename, const Partition& partition) const;

    NeuronInNeighborhood neuron_in_neighborhood{};
    NeuronOutNeighborhood neuron_out_neighborhood{};

    size_t my_num_neurons{ Constants::uninitialized }; // My number of neurons
};
