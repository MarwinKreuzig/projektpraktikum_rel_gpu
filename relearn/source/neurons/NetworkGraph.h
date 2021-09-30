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

    using LocalEdgesKey = size_t;
    using LocalEdges = std::vector<std::pair<LocalEdgesKey, EdgesVal>>;

    using NeuronLocalInNeighborhood = std::vector<LocalEdges>;
    using NeuronLocalOutNeighborhood = std::vector<LocalEdges>;

    enum class EdgeDirection {
        In,
        Out
    };

    /**
     * @brief Constructs an object that has enough space to store the given number of neurons
     * @param num_neurons The number of neurons that the object shall handle
     */
    explicit NetworkGraph(const size_t num_neurons)
        : neuron_in_neighborhood(num_neurons)
        , neuron_out_neighborhood(num_neurons)
        , neuron_distant_in_neighborhood(num_neurons)
        , neuron_distant_out_neighborhood(num_neurons)
        , neuron_local_in_neighborhood(num_neurons)
        , neuron_local_out_neighborhood(num_neurons)
        , my_num_neurons(num_neurons) {
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

        neuron_distant_in_neighborhood.resize(new_size);
        neuron_distant_out_neighborhood.resize(new_size);

        neuron_local_in_neighborhood.resize(new_size);
        neuron_local_out_neighborhood.resize(new_size);

        my_num_neurons = new_size;
    }

    /**
     * @brief Returns a constant reference to all in-edges to a neuron, i.e., a view on all neurons that connect to the specified one via a synapse
     * @param local_neuron_id The id of the neuron
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all in-edges that is invalidated when the state of the object changes
     */
    [[nodiscard]] const Edges& get_in_edges(const size_t local_neuron_id) const {
        RelearnException::check(local_neuron_id < neuron_in_neighborhood.size(), "NetworkGraph::get_in_edges: Tried with a too large id of {}", local_neuron_id);
        return neuron_in_neighborhood[local_neuron_id];
    }

    /**
     * @brief Returns a constant reference to all out-edges to a neuron, i.e., a view on all neurons that the specified one connectes to via a synapse
     * @param local_neuron_id The id of the neuron
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all out-edges that is invalidated when the state of the object changes
     */
    [[nodiscard]] const Edges& get_out_edges(const size_t local_neuron_id) const {
        RelearnException::check(local_neuron_id < neuron_out_neighborhood.size(), "NetworkGraph::get_out_edges: Tried with a too large id of {}", local_neuron_id);
        return neuron_out_neighborhood[local_neuron_id];
    }

    [[nodiscard]] const Edges& get_distant_in_edges(const size_t local_neuron_id) const {
        RelearnException::check(local_neuron_id < neuron_distant_in_neighborhood.size(), "NetworkGraph::get_distant_in_edges: Tried with a too large id of {}", local_neuron_id);
        return neuron_distant_in_neighborhood[local_neuron_id];
    }

    [[nodiscard]] const Edges& get_distant_out_edges(const size_t local_neuron_id) const {
        RelearnException::check(local_neuron_id < neuron_distant_out_neighborhood.size(), "NetworkGraph::get_distant_out_edges: Tried with a too large id of {}", local_neuron_id);
        return neuron_distant_out_neighborhood[local_neuron_id];
    }

    [[nodiscard]] const LocalEdges& get_local_in_edges(const size_t local_neuron_id) const {
        RelearnException::check(local_neuron_id < neuron_local_in_neighborhood.size(), "NetworkGraph::get_local_in_edges: Tried with a too large id of {}", local_neuron_id);
        return neuron_local_in_neighborhood[local_neuron_id];
    }

    [[nodiscard]] const LocalEdges& get_local_out_edges(const size_t local_neuron_id) const {
        RelearnException::check(local_neuron_id < neuron_local_out_neighborhood.size(), "NetworkGraph::get_local_out_edges: Tried with a too large id of {}", local_neuron_id);
        return neuron_local_out_neighborhood[local_neuron_id];
    }

    /**
     * @brief Returns a copy of all in-edges to a neuron, i.e., a copy of all neurons that connect to the specified one via a synapse, of a specified type
     * @param local_neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     * @return A copy of all in-edges from a certain neuron signal type
     */
    [[nodiscard]] Edges get_in_edges(const size_t local_neuron_id, const SignalType signal_type) const {
        const auto my_rank = MPIWrapper::get_my_rank();

        const Edges& all_distant_edges = get_distant_in_edges(local_neuron_id);
        const LocalEdges& all_local_edges = get_local_in_edges(local_neuron_id);

        Edges filtered_edges{};
        filtered_edges.reserve(all_distant_edges.size() + all_local_edges.size());

        for (const auto& [edge_key, edge_val] : all_distant_edges) {
            if (signal_type == SignalType::EXCITATORY && edge_val > 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }

            if (signal_type == SignalType::INHIBITORY && edge_val < 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }
        }

        for (const auto& [edge_key, edge_val] : all_local_edges) {
            if (signal_type == SignalType::EXCITATORY && edge_val > 0) {
                filtered_edges.emplace_back(std::pair<int, size_t>(my_rank, edge_key), edge_val);
            }

            if (signal_type == SignalType::INHIBITORY && edge_val < 0) {
                filtered_edges.emplace_back(std::pair<int, size_t>(my_rank, edge_key), edge_val);
            }
        }

        return filtered_edges;
    }

    /**
     * @brief Returns a copy of all out-edges from a neuron, i.e., a copy of all neurons that the specified one connectes to via a synapse, of a specified type
     * @param local_neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     * @return A copy of all out-edges to a certain neuron signal type
     */
    [[nodiscard]] Edges get_out_edges(const size_t local_neuron_id, const SignalType signal_type) const {
        const auto my_rank = MPIWrapper::get_my_rank();

        const Edges& all_distant_edges = get_distant_out_edges(local_neuron_id);
        const LocalEdges& all_local_edges = get_local_out_edges(local_neuron_id);

        Edges filtered_edges{};
        filtered_edges.reserve(all_distant_edges.size() + all_local_edges.size());

        for (const auto& [edge_key, edge_val] : all_distant_edges) {
            if (signal_type == SignalType::EXCITATORY && edge_val > 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }

            if (signal_type == SignalType::INHIBITORY && edge_val < 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }
        }

        for (const auto& [edge_key, edge_val] : all_local_edges) {
            if (signal_type == SignalType::EXCITATORY && edge_val > 0) {
                filtered_edges.emplace_back(std::pair<int, size_t>(my_rank, edge_key), edge_val);
            }

            if (signal_type == SignalType::INHIBITORY && edge_val < 0) {
                filtered_edges.emplace_back(std::pair<int, size_t>(my_rank, edge_key), edge_val);
            }
        }

        return filtered_edges;
    }

    /**
     * @brief Returns the number of all in-edges to a neuron (countings multiplicities) from excitatory neurons
     * @param local_neuron_id The id of the neuron
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     * @return The number of incoming synapses that the specified neuron formed from excitatory neurons
     */
    [[nodiscard]] size_t get_number_excitatory_in_edges(const size_t local_neuron_id) const {
        const Edges& all_distant_edges = get_distant_in_edges(local_neuron_id);
        const LocalEdges& all_local_edges = get_local_in_edges(local_neuron_id);

        size_t total_num_ports = 0;

        for (const auto& [_, connection_strength] : all_distant_edges) {
            if (connection_strength > 0) {
                total_num_ports += connection_strength;
            }
        }

        for (const auto& [_, connection_strength] : all_local_edges) {
            if (connection_strength > 0) {
                total_num_ports += connection_strength;
            }
        }

        return total_num_ports;
    }

    /**
     * @brief Returns the number of all in-edges to a neuron (countings multiplicities) from inhibitory neurons
     * @param local_neuron_id The id of the neuron
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     * @return The number of incoming synapses that the specified neuron formed from inhibitory neurons
     */
    [[nodiscard]] size_t get_number_inhibitory_in_edges(const size_t local_neuron_id) const {
        const Edges& all_distant_edges = get_distant_in_edges(local_neuron_id);
        const LocalEdges& all_local_edges = get_local_in_edges(local_neuron_id);

        size_t total_num_ports = 0;

        for (const auto& [_, connection_strength] : all_distant_edges) {
            if (connection_strength < 0) {
                total_num_ports += -connection_strength;
            }
        }

        for (const auto& [_, connection_strength] : all_local_edges) {
            if (connection_strength < 0) {
                total_num_ports += -connection_strength;
            }
        }

        return total_num_ports;
    }

    /**
     * @brief Returns the number of all out-edges from a neuron (countings multiplicities)
     * @param local_neuron_id The id of the neuron
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     * @return The number of outgoing synapses that the specified neuron formed
     */
    [[nodiscard]] size_t get_number_out_edges(const size_t local_neuron_id) const {
        const Edges& all_distant_edges = get_distant_out_edges(local_neuron_id);
        const LocalEdges& all_local_edges = get_local_out_edges(local_neuron_id);

        size_t total_num_ports = 0;

        for (const auto& [_, connection_strength] : all_distant_edges) {
            total_num_ports += std::abs(connection_strength);
        }

        for (const auto& [_, connection_strength] : all_local_edges) {
            total_num_ports += std::abs(connection_strength);
        }

        return total_num_ports;
    }

    /**
     * @brief Adds the specified weight to the synapse from the neuron specified by source_id to the neuron specified by target_id.
     * If there was no edge before, it is created. If the updated weight is 0, it is deleted. Only updates the local part of the network graph.
	 * @param target_id The target_id neuron's id and rank
     * @param source_id The source_id neuron's id and rank
     * @param weight The weight that should be added onto the current connections, not zero
     * @exception Throws a RelearnException if: 
     * (a) the weight is zero, 
     * (b) neither the target_id nor the source_id are on the current rank,
     * (c) a local neuron id is larger than the number of neurons
	 */
    void add_edge_weight(const RankNeuronId& target_id, const RankNeuronId& source_id, const int weight) {
        RelearnException::check(weight != 0, "NetworkGraph::add_edge_weight: weight of edge to add is zero");

        const auto target_rank = target_id.get_rank();
        const auto target_neuron_id = target_id.get_neuron_id();

        const auto source_rank = source_id.get_rank();
        const auto source_neuron_id = source_id.get_neuron_id();

        const auto my_rank = MPIWrapper::get_my_rank();

        if (target_rank != my_rank && source_rank != my_rank) {
            RelearnException::fail("NetworkGraph::add_edge_weight: In NetworkGraph::add_edge_weight, neither the target nor the source rank were for me.");
        }

        if (target_rank == my_rank) {
            RelearnException::check(source_neuron_id < my_num_neurons,
                "NetworkGraph::add_edge_weight: Want to add an in-edge with a too large source id: {} {}", target_neuron_id, my_num_neurons);
        }

        if (source_rank == my_rank) {
            RelearnException::check(source_neuron_id < my_num_neurons,
                "NetworkGraph::add_edge_weight: Want to add an out-edge with a too large source id: {} {}", source_neuron_id, my_num_neurons);
        }

        if (target_rank == source_rank) {
            LocalEdges& in_edges = neuron_local_in_neighborhood[target_neuron_id];
            LocalEdges& out_edges = neuron_local_out_neighborhood[source_neuron_id];

            add_local_edge(in_edges, source_neuron_id, weight);
            add_local_edge(out_edges, target_neuron_id, weight);
        }

        // Target neuron is mine
        if (target_rank == my_rank) {
            Edges& in_edges = neuron_in_neighborhood[target_neuron_id];
            add_edge(in_edges, source_rank, source_neuron_id, weight);

            Edges& distant_in_edges = neuron_distant_in_neighborhood[target_neuron_id];
            add_distant_edge(distant_in_edges, source_rank, source_neuron_id, weight);
        }

        if (source_rank == my_rank) {
            Edges& out_edges = neuron_out_neighborhood[source_neuron_id];
            add_edge(out_edges, target_rank, target_neuron_id, weight);

            Edges& distant_out_edges = neuron_distant_out_neighborhood[source_neuron_id];
            add_distant_edge(distant_out_edges, target_rank, target_neuron_id, weight);
        }
    }

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
     * @brief Returns a histogram of the local neurons' connectivity
     * @param edge_direction An enum that indicates if in edges or out edges should be considered
     * @return A histogram of the connectivity, i.e., <return>[i] == c indicates that c local neurons have i edges in the requested direction
     */
    std::vector<unsigned int> get_edges_histogram(EdgeDirection edge_direction) const noexcept {
        std::vector<unsigned int> result{};

        auto latest_result = 0;

        const auto& neighborhood = (edge_direction == EdgeDirection::In) ? neuron_in_neighborhood : neuron_out_neighborhood;

        for (const auto& in_neighborhood : neighborhood) {
            auto sum = 0;

            for (const auto& [_, val] : in_neighborhood) {
                if (val < 0) {
                    sum -= val;
                } else {
                    sum += val;
                }
            }

            if (result.size() <= sum) {
                result.resize(sum * 2ull + 1);
                latest_result = sum;
            }

            result[sum]++;
        }

        result.resize(latest_result + 1ull);
        return result;
    }

    /**
     * @brief Prints all stored connections to the out-stream. Uses the global neuron ids and starts with 1. The format is <target_id id> <source_id id> <weight>
     * @param os The out-stream to which the network graph is printed
     * @param informations The NeuronsExtraInfo that is used to translate between local neuron id and global neuron id
     * @exception Throws a RelearnException if the translation of a neuron id fails
     */
    void print(std::ostream& os, const std::unique_ptr<NeuronsExtraInfo>& informations) const;

    /**
     * @brief Performs a debug check on the local portion of the network graph. All stored ranks must be greater or equal to zero, no weight must be equal to zero, and all purely local edges must have a matching counterpart.
     * @exception Throws a RelearnException if any of the conditions is violated
     */
    void debug_check() const;

private:
    // NOLINTNEXTLINE
    static void add_edge(Edges& edges, const int rank, const size_t local_neuron_id, const int weight) {
        const EdgesKey rank_neuron_id_pair{ rank, local_neuron_id };

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

    static void add_local_edge(LocalEdges& edges, const size_t local_neuron_id, const int weight) {
        size_t idx = 0;

        for (auto& [key, val] : edges) {
            if (key == local_neuron_id) {
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

        edges.emplace_back(local_neuron_id, weight);
    }

    static void add_distant_edge(Edges& edges, const int rank, const size_t local_neuron_id, const int weight) {
        add_edge(edges, rank, local_neuron_id, weight);
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

    NeuronInNeighborhood neuron_distant_in_neighborhood{};
    NeuronOutNeighborhood neuron_distant_out_neighborhood{};

    NeuronLocalInNeighborhood neuron_local_in_neighborhood{};
    NeuronLocalOutNeighborhood neuron_local_out_neighborhood{};

    size_t my_num_neurons{ Constants::uninitialized }; // My number of neurons
};
