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

#include "Config.h"
#include "SignalType.h"
#include "Types.h"
#include "neurons/NeuronsExtraInfo.h"
#include "util/RelearnException.h"

#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

/**
 * An object of type NetworkGraph stores the synaptic connections between neurons, that are relevant for the current MPI rank.
 * The neurons are refered to by indices in the range [0, num_local_neurons).
 * The class does not perform any communication or synchronization with other MPI ranks when messing with edges;
 * it only does so when calling NetworkGraph::translate_global_to_local, and in that, it does not
 * mess with edges.
 * NetworkGraph differentiates between local edges (from the current MPI rank to the current MPI rank) and
 * distant edges (another MPI rank is the owner of the target or source neuron).
 */
class NetworkGraph {
public:
    /**
	 * Type definitions
	 */
    using DistantEdgesKey = RankNeuronId; // Pair of (mpi rank, local neuron id)
    using DistantEdges = std::vector<std::pair<DistantEdgesKey, RelearnTypes::synapse_weight>>;

    using NeuronDistantInNeighborhood = std::vector<DistantEdges>;
    using NeuronDistantOutNeighborhood = std::vector<DistantEdges>;

    using LocalEdges = std::vector<std::pair<NeuronID, RelearnTypes::synapse_weight>>;

    using NeuronLocalInNeighborhood = std::vector<LocalEdges>;
    using NeuronLocalOutNeighborhood = std::vector<LocalEdges>;

    using position_type = RelearnTypes::position_type;

    enum class EdgeDirection {
        In,
        Out
    };

    /**
     * @brief Constructs an object that has enough space to store the given number of neurons
     * @param number_neurons The number of neurons that the object shall handle
     * @param mpi_rank The mpi rank that handles this portion of the graph, must be >= 0
     * @exception Throws an exception if the allocation of memory fails
     *      Throws a RelearnException if mpi_rank < 0
     */
    NetworkGraph(const size_t number_neurons, const int mpi_rank)
        : neuron_distant_in_neighborhood(number_neurons)
        , neuron_distant_out_neighborhood(number_neurons)
        , neuron_local_in_neighborhood(number_neurons)
        , neuron_local_out_neighborhood(number_neurons)
        , number_local_neurons(number_neurons)
        , mpi_rank(mpi_rank) {

        RelearnException::check(mpi_rank >= 0, "NetworkGraph::NetworkGraph: mpi_rank was negative: {}", mpi_rank);
    }

    /**
     * @brief Resizes the network graph by adding space for more neurons. Invalidates iterators
     * @param creation_count The number of additional neurons the network graph should handle
     * @exception Throws an exception if the allocation of memory fails
     */
    void create_neurons(const size_t creation_count) {
        const auto old_size = number_local_neurons;
        const auto new_size = old_size + creation_count;

        neuron_distant_in_neighborhood.resize(new_size);
        neuron_distant_out_neighborhood.resize(new_size);

        neuron_local_in_neighborhood.resize(new_size);
        neuron_local_out_neighborhood.resize(new_size);

        number_local_neurons = new_size;
    }

    [[nodiscard]] double get_average_euclidean_distance(const std::unique_ptr<NeuronsExtraInfo>& infos) {
        auto total_euclidean_distance = 0.0;
        auto number_edges = 0;

        const auto& positions = infos->get_positions();

        for (auto target_id = 0; target_id < neuron_local_in_neighborhood.size(); target_id++) {
            const auto& neighborhood = neuron_local_in_neighborhood[target_id];

            for (const auto& [source_neuron_id, weight] : neighborhood) {
                const auto& source_id = source_neuron_id.get_local_id();

                const auto& source_position = positions[source_id];
                const auto& target_position = positions[target_id];

                const auto& difference = target_position - source_position;
                const auto& length = difference.calculate_2_norm();

                const auto& abs_weight = std::abs(weight);

                number_edges += abs_weight;
                total_euclidean_distance += (abs_weight * length);
            }
        }

        const auto average_euclidean_distance = total_euclidean_distance / number_edges;
        return average_euclidean_distance;
    }

    /**
     * @brief Returns a constant reference to all distant in-edges to a neuron, i.e., a view on neurons that connect to the specified one via a synapse
     *      and belong to another MPI rank
     * @param local_neuron_id The id of the neuron
     * @exception Throws a RelearnException if local_neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all distant in-edges.
     */
    [[nodiscard]] const DistantEdges& get_distant_in_edges(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_local_id();

        RelearnException::check(local_neuron_id < neuron_distant_in_neighborhood.size(),
            "NetworkGraph::get_distant_in_edges: Tried with a too large id of {}", neuron_id);

        return neuron_distant_in_neighborhood[local_neuron_id];
    }

    /**
     * @brief Returns a constant reference to all distant out-edges to a neuron, i.e., a view on all neurons that the specified one connectes to via a synapse
     *      and belong to another MPI rank
     * @param local_neuron_id The id of the neuron
     * @exception Throws a RelearnException if local_neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all distant out-edges.
     */
    [[nodiscard]] const DistantEdges& get_distant_out_edges(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_local_id();

        RelearnException::check(local_neuron_id < neuron_distant_out_neighborhood.size(),
            "NetworkGraph::get_distant_out_edges: Tried with a too large id of {}", neuron_id);

        return neuron_distant_out_neighborhood[local_neuron_id];
    }

    /**
     * @brief Returns a constant reference to all local in-edges to a neuron, i.e., a view on neurons that connect to the specified one via a synapse
     *      and belong to the current MPI rank
     * @param local_neuron_id The id of the neuron
     * @exception Throws a RelearnException if local_neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all local in-edges.
     */
    [[nodiscard]] const LocalEdges& get_local_in_edges(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_local_id();

        RelearnException::check(local_neuron_id < neuron_local_in_neighborhood.size(),
            "NetworkGraph::get_local_in_edges: Tried with a too large id of {}", neuron_id);

        return neuron_local_in_neighborhood[local_neuron_id];
    }

    /**
     * @brief Returns a constant reference to all local out-edges to a neuron, i.e., a view on all neurons that the specified one connectes to via a synapse
     *      and belong to the current MPI rank
     * @param local_neuron_id The id of the neuron
     * @exception Throws a RelearnException if local_neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all local out-edges.
     */
    [[nodiscard]] const LocalEdges& get_local_out_edges(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_local_id();

        RelearnException::check(local_neuron_id < neuron_local_out_neighborhood.size(),
            "NetworkGraph::get_local_out_edges: Tried with a too large id of {}", neuron_id);

        return neuron_local_out_neighborhood[local_neuron_id];
    }

    /**
     * @brief Returns a copy of all in-edges to a neuron, i.e., a copy of all neurons that connect to the specified one via a synapse, of a specified type.
     *      All local in-edges are added with the current MPI rank.
     * @param local_neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     *      Throws an exception if the allocation of memory fails
     * @return A copy of all in-edges from a certain neuron signal type
     */
    [[nodiscard]] DistantEdges get_all_in_edges(const NeuronID& local_neuron_id, const SignalType signal_type) const {
        const auto my_rank = mpi_rank;

        const DistantEdges& all_distant_edges = get_distant_in_edges(local_neuron_id);
        const LocalEdges& all_local_edges = get_local_in_edges(local_neuron_id);

        DistantEdges filtered_edges{};
        filtered_edges.reserve(all_distant_edges.size() + all_local_edges.size());

        for (const auto& [neuron_id, edge_val] : all_local_edges) {
            if (signal_type == SignalType::Excitatory && edge_val > 0) {
                filtered_edges.emplace_back(RankNeuronId(my_rank, neuron_id), edge_val);
            }

            if (signal_type == SignalType::Inhibitory && edge_val < 0) {
                filtered_edges.emplace_back(RankNeuronId(my_rank, neuron_id), edge_val);
            }
        }

        for (const auto& [edge_key, edge_val] : all_distant_edges) {
            if (signal_type == SignalType::Excitatory && edge_val > 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }

            if (signal_type == SignalType::Inhibitory && edge_val < 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }
        }

        return filtered_edges;
    }

    /**
     * @brief Returns a copy of all out-edges from a neuron, i.e., a copy of all neurons that the specified one connectes to via a synapse, of a specified type
     *      All local in-edges are added with the current MPI rank.
     * @param local_neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     *      Throws an exception if the allocation of memory fails
     * @return A copy of all out-edges to a certain neuron signal type
     */
    [[nodiscard]] DistantEdges get_all_out_edges(const NeuronID& local_neuron_id, const SignalType signal_type) const {
        const auto my_rank = mpi_rank;

        const DistantEdges& all_distant_edges = get_distant_out_edges(local_neuron_id);
        const LocalEdges& all_local_edges = get_local_out_edges(local_neuron_id);

        DistantEdges filtered_edges{};
        filtered_edges.reserve(all_distant_edges.size() + all_local_edges.size());

        for (const auto& [edge_key, edge_val] : all_local_edges) {
            if (signal_type == SignalType::Excitatory && edge_val > 0) {
                filtered_edges.emplace_back(RankNeuronId(my_rank, edge_key), edge_val);
            }

            if (signal_type == SignalType::Inhibitory && edge_val < 0) {
                filtered_edges.emplace_back(RankNeuronId(my_rank, edge_key), edge_val);
            }
        }

        for (const auto& [edge_key, edge_val] : all_distant_edges) {
            if (signal_type == SignalType::Excitatory && edge_val > 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }

            if (signal_type == SignalType::Inhibitory && edge_val < 0) {
                filtered_edges.emplace_back(edge_key, edge_val);
            }
        }

        return filtered_edges;
    }

    /**
     * @brief Returns a copy of all in-edges to a neuron, i.e., a copy of all neurons that connect to the specified one via a synapse.
     *      All local in-edges are added with the current MPI rank.
     * @param local_neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     *      Throws an exception if the allocation of memory fails
     * @return A copy of all in-edges
     */
    [[nodiscard]] DistantEdges get_all_in_edges(const NeuronID& local_neuron_id) const {
        const auto my_rank = mpi_rank;

        const DistantEdges& all_distant_edges = get_distant_in_edges(local_neuron_id);
        const LocalEdges& all_local_edges = get_local_in_edges(local_neuron_id);

        DistantEdges filtered_edges{};
        filtered_edges.reserve(all_distant_edges.size() + all_local_edges.size());

        for (const auto& [edge_key, edge_val] : all_local_edges) {
            filtered_edges.emplace_back(RankNeuronId(my_rank, edge_key), edge_val);
        }

        for (const auto& [edge_key, edge_val] : all_distant_edges) {
            filtered_edges.emplace_back(edge_key, edge_val);
        }

        return filtered_edges;
    }

    /**
     * @brief Returns a copy of all out-edges from a neuron, i.e., a copy of all neurons that the specified one connectes to via a synapse
     *      All local in-edges are added with the current MPI rank.
     * @param local_neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     *      Throws an exception if the allocation of memory fails
     * @return A copy of all out-edges
     */
    [[nodiscard]] DistantEdges get_all_out_edges(const NeuronID& local_neuron_id) const {
        const auto my_rank = mpi_rank;

        const DistantEdges& all_distant_edges = get_distant_out_edges(local_neuron_id);
        const LocalEdges& all_local_edges = get_local_out_edges(local_neuron_id);

        DistantEdges filtered_edges{};
        filtered_edges.reserve(all_distant_edges.size() + all_local_edges.size());

        for (const auto& [edge_key, edge_val] : all_local_edges) {
            filtered_edges.emplace_back(RankNeuronId(my_rank, edge_key), edge_val);
        }

        for (const auto& [edge_key, edge_val] : all_distant_edges) {
            filtered_edges.emplace_back(edge_key, edge_val);
        }

        return filtered_edges;
    }

    /**
     * @brief Returns the number of all in-edges to a neuron (countings multiplicities) from excitatory neurons
     * @param local_neuron_id The id of the neuron
     * @exception Throws a ReleanException if local_neuron_id is larger or equal to the number of neurons stored
     * @return The number of incoming synapses that the specified neuron formed from excitatory neurons
     */
    [[nodiscard]] size_t get_number_excitatory_in_edges(const NeuronID& local_neuron_id) const {
        const DistantEdges& all_distant_edges = get_distant_in_edges(local_neuron_id);
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
    [[nodiscard]] size_t get_number_inhibitory_in_edges(const NeuronID& local_neuron_id) const {
        const DistantEdges& all_distant_edges = get_distant_in_edges(local_neuron_id);
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
    [[nodiscard]] size_t get_number_out_edges(const NeuronID& local_neuron_id) const {
        const DistantEdges& all_distant_edges = get_distant_out_edges(local_neuron_id);
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
     *      If there was no edge before, it is created. If the updated weight is 0, it is deleted. Only updates the local part of the network graph.
     *      A call to this method can change the order in which the edges are stored.
     * @param target_id The target_id neuron's id and rank
     * @param source_id The source_id neuron's id and rank
     * @param weight The weight that should be added onto the current connections, not zero
     * @exception Throws a RelearnException if:
     *      (a) the weight is zero,
     *      (b) neither the target_id nor the source_id are on the current rank,
     *      (c) a local neuron id is larger than the number of neurons
     *      Throws an exception if the allocation of memory fails
	 */
    void add_edge_weight(const RankNeuronId& target_id, const RankNeuronId& source_id, const RelearnTypes::synapse_weight& weight) {
        RelearnException::check(weight != 0, "NetworkGraph::add_edge_weight: weight of edge to add is zero");

        const auto target_rank = target_id.get_rank();
        const auto target_neuron_id = target_id.get_neuron_id();
        const auto local_target_neuron_id = target_neuron_id.get_local_id();

        const auto source_rank = source_id.get_rank();
        const auto source_neuron_id = source_id.get_neuron_id();
        const auto local_source_neuron_id = source_neuron_id.get_local_id();

        const auto my_rank = mpi_rank;

        if (target_rank != my_rank && source_rank != my_rank) {
            RelearnException::fail("NetworkGraph::add_edge_weight: In NetworkGraph::add_edge_weight, neither the target {} nor the source rank {} were for me.", target_rank, source_rank);
        }

        if (target_rank == my_rank) {
            RelearnException::check(local_target_neuron_id < number_local_neurons,
                "NetworkGraph::add_edge_weight: Want to add an in-edge with a too large target id: {} {}", target_neuron_id, number_local_neurons);
        }

        if (source_rank == my_rank) {
            RelearnException::check(local_source_neuron_id < number_local_neurons,
                "NetworkGraph::add_edge_weight: Want to add an out-edge with a too large source id: {} {}", source_neuron_id, number_local_neurons);
        }

        if (target_rank == source_rank) {
            LocalEdges& in_edges = neuron_local_in_neighborhood[local_target_neuron_id];
            LocalEdges& out_edges = neuron_local_out_neighborhood[local_source_neuron_id];

            add_edge<LocalEdges, NeuronID>(in_edges, source_neuron_id, weight);
            add_edge<LocalEdges, NeuronID>(out_edges, target_neuron_id, weight);
        }

        // Target neuron is mine but source neuron is not
        if (target_rank == my_rank && source_rank != my_rank) {
            DistantEdges& distant_in_edges = neuron_distant_in_neighborhood[local_target_neuron_id];
            add_edge<DistantEdges, DistantEdgesKey>(distant_in_edges, source_id, weight);
        }

        // Source neuron is mine but target neuron is not
        if (source_rank == my_rank && target_rank != my_rank) {
            DistantEdges& distant_out_edges = neuron_distant_out_neighborhood[local_source_neuron_id];
            add_edge<DistantEdges, DistantEdgesKey>(distant_out_edges, target_id, weight);
        }
    }

    /**
     * @brief Adds a local synapse to the networgh graph
     * @param synapse The local synapse
     * @exception Throws a RelearnException if
     *      (a) The target is larger than the number neurons
     *      (b) The source is larger than the number neurons
     *      (c) The weight is equal to 0
     */
    void add_synapse(const LocalSynapse& synapse) {
        const auto& [target, source, weight] = synapse;

        const auto local_target_id = target.get_local_id();
        const auto local_source_id = source.get_local_id();

        RelearnException::check(local_target_id < number_local_neurons, "NetworkGraph::add_synapse: Local synapse had a too large target: {} vs {}", target, number_local_neurons);
        RelearnException::check(local_source_id < number_local_neurons, "NetworkGraph::add_synapse: Local synapse had a too large source: {} vs {}", source, number_local_neurons);
        RelearnException::check(weight != 0, "NetworkGraph::add_synapse: Local synapse had weight 0");

        LocalEdges& in_edges = neuron_local_in_neighborhood[local_target_id];
        LocalEdges& out_edges = neuron_local_out_neighborhood[local_source_id];

        add_edge<LocalEdges, NeuronID>(in_edges, source, weight);
        add_edge<LocalEdges, NeuronID>(out_edges, target, weight);
    }

    /**
     * @brief Adds a distant in-synapse to the networgh graph (it might actually come from the same node, that's no problem)
     * @param synapse The distant in-synapse, must come from another rank
     * @exception Throws a RelearnException if
     *      (a) The target is larger than the number neurons
     *      (b) The weight is equal to 0
     *      (c) The distant rank is the same as the local one
     */
    void add_synapse(const DistantInSynapse& synapse) {
        const auto& [target, source_rni, weight] = synapse;
        const auto local_target_id = target.get_local_id();

        const auto& [source_rank, source_id] = source_rni;

        RelearnException::check(local_target_id < number_local_neurons, "NetworkGraph::add_synapse: Distant in-synapse had a too large target: {} vs {}", target, number_local_neurons);
        RelearnException::check(source_rank != mpi_rank, "NetworkGraph::add_synapse: Distant in-synapse was on my rank: {}", source_rank);
        RelearnException::check(weight != 0, "NetworkGraph::add_synapse: Local synapse had weight 0");

        DistantEdges& distant_in_edges = neuron_distant_in_neighborhood[local_target_id];
        add_edge<DistantEdges, DistantEdgesKey>(distant_in_edges, source_rni, weight);
    }

    /**
     * @brief Adds a distant out-synapse to the networgh graph (it might actually come from the same node, that's no problem)
     * @param synapse The distant out-synapse, must come from another rank
     * @exception Throws a RelearnException if
     *      (a) The target rank is the same as the current rank
     *      (b) The weight is equal to 0
     *      (c) The distant rank is the same as the local one
     */
    void add_synapse(const DistantOutSynapse& synapse) {
        const auto& [target_rni, source, weight] = synapse;
        const auto local_source_id = source.get_local_id();

        const auto& [target_rank, target_id] = target_rni;

        RelearnException::check(local_source_id < number_local_neurons, "NetworkGraph::add_synapse: Distant out-synapse had a too large target: {} vs {}", source, number_local_neurons);
        RelearnException::check(target_rank != mpi_rank, "NetworkGraph::add_synapse: Distant out-synapse was on my rank: {}", target_rank);
        RelearnException::check(weight != 0, "NetworkGraph::add_synapse: Local synapse had weight 0");

        DistantEdges& distant_out_edges = neuron_distant_out_neighborhood[local_source_id];
        add_edge<DistantEdges, DistantEdgesKey>(distant_out_edges, target_rni, weight);
    }

    /**
     * @brief Adds all provided edges into the network graph at once.
     * @param local_edges All edges between two neurons on the current MPI rank
     * @param in_edges All edges that have a target on the current MPI rank and a source from another rank
     * @param out_edges All edges that have a source on the current MPI rank and a target from another rank
    */
    void add_edges(const LocalSynapses& local_edges, const DistantInSynapses& in_edges, const DistantOutSynapses& out_edges) {
        for (const auto& [target_id, source_id, weight] : local_edges) {
            const auto local_target_id = target_id.get_local_id();
            const auto local_source_id = source_id.get_local_id();

            RelearnException::check(local_target_id < neuron_local_in_neighborhood.size(),
                "NetworkGraph::add_edges: local_in_neighborhood is too small: {} vs {}", target_id, neuron_local_in_neighborhood.size());
            RelearnException::check(local_source_id < neuron_local_out_neighborhood.size(),
                "NetworkGraph::add_edges: local_out_neighborhood is too small: {} vs {}", source_id, neuron_distant_out_neighborhood.size());

            LocalEdges& in_edges = neuron_local_in_neighborhood[local_target_id];
            LocalEdges& out_edges = neuron_local_out_neighborhood[local_source_id];

            add_edge<LocalEdges, NeuronID>(in_edges, source_id, weight);
            add_edge<LocalEdges, NeuronID>(out_edges, target_id, weight);
        }

        for (const auto& [target_id, source_rni, weight] : in_edges) {
            const auto local_target_id = target_id.get_local_id();

            RelearnException::check(local_target_id < neuron_distant_in_neighborhood.size(),
                "NetworkGraph::add_edges: distant_in_neighborhood is too small: {} vs {}", target_id, neuron_distant_in_neighborhood.size());

            DistantEdges& distant_in_edges = neuron_distant_in_neighborhood[local_target_id];
            add_edge<DistantEdges, DistantEdgesKey>(distant_in_edges, source_rni, weight);
        }

        for (const auto& [target_rni, source_id, weight] : out_edges) {
            const auto local_source_id = source_id.get_local_id();

            RelearnException::check(local_source_id < neuron_distant_out_neighborhood.size(),
                "NetworkGraph::add_edges: distant_out_neighborhood is too small: {} vs {}", source_id, neuron_distant_out_neighborhood.size());

            DistantEdges& distant_out_edges = neuron_distant_out_neighborhood[local_source_id];
            add_edge<DistantEdges, DistantEdgesKey>(distant_out_edges, target_rni, weight);
        }
    }

    /**
     * @brief Checks if the specified file contains only synapses between neurons with specified ids (only works locally).
     * @param path_synapses The path to the file in which the synapses are stored (with the global neuron ids starting at 1)
     * @param neuron_ids The neuron ids between which the synapses should be formed. Must be sorted ascendingly
     * @exception Throws an exception if the allocation of memory fails
     * @return Returns true iff the file has the correct format and only ids in neuron_ids are present
     */
    [[nodiscard]] static bool check_edges_from_file(const std::filesystem::path& path_synapses, const std::vector<size_t>& neuron_ids);

    /**
     * @brief Returns a histogram of the local neurons' connectivity
     * @param edge_direction An enum that indicates if in-edges or out-edges should be considered
     * @exception Throws an exception if the allocation of memory fails
     * @return A histogram of the connectivity, i.e., <return>[i] == c indicates that c local neurons have i edges in the requested direction
     */
    [[nodiscard]] std::vector<unsigned int> get_edges_histogram(EdgeDirection edge_direction) const {
        std::vector<unsigned int> result{};

        auto largest_number_edges = 0;

        const auto& local_neighborhood = (edge_direction == EdgeDirection::In) ? neuron_local_in_neighborhood : neuron_local_out_neighborhood;
        const auto& distant_neighborhood = (edge_direction == EdgeDirection::In) ? neuron_distant_in_neighborhood : neuron_distant_out_neighborhood;

        for (auto neuron_id = 0; neuron_id < number_local_neurons; neuron_id++) {
            auto number_of_connections = 0;

            for (const auto& [_, val] : local_neighborhood[neuron_id]) {
                if (val < 0) {
                    number_of_connections -= val;
                } else {
                    number_of_connections += val;
                }
            }

            for (const auto& [_, val] : distant_neighborhood[neuron_id]) {
                if (val < 0) {
                    number_of_connections -= val;
                } else {
                    number_of_connections += val;
                }
            }

            if (result.size() <= number_of_connections) {
                result.resize(number_of_connections * 2ULL + 1);
            }

            largest_number_edges = std::max(number_of_connections, largest_number_edges);

            result[number_of_connections]++;
        }

        result.resize(largest_number_edges + 1ULL);
        return result;
    }

    /**
     * @brief Prints all stored connections to the streams. Does not perform communication via MPI. Uses the local neuron ids and starts with 1. The formats are:
     *      <target_rank> <target_id>\t<source_rank> <source_id>\tweight
     * @param os_out_edges The out-stream to which the out-connections are printed
     * @param os_in_edges The out-stream to which the in-connections are printed
     */
    void print_with_ranks(std::ostream& os_out_edges, std::ostream& os_in_edges) const;

    /**
     * @brief Returns directly if !Config::do_debug_checks
     *      Performs a debug check on the local portion of the network graph
     *      All stored ranks must be greater or equal to zero, no weight must be equal to zero,
     *      and all purely local edges must have a matching counterpart.
     * @exception Throws a RelearnException if any of the conditions is violated
     */
    void debug_check() const;

private:
    template <typename Edges, typename NeuronId>
    // NOLINTNEXTLINE
    static void add_edge(Edges& edges, const NeuronId& other_neuron_id, const RelearnTypes::synapse_weight& weight) {
        size_t idx = 0;

        for (auto& [neuron_id, edge_weight] : edges) {
            if (neuron_id == other_neuron_id) {
                const auto new_edge_weight = edge_weight + weight;
                edge_weight = new_edge_weight;

                if (new_edge_weight == 0) {
                    const auto idx_last = edges.size() - 1;
                    std::swap(edges[idx], edges[idx_last]);
                    edges.erase(edges.cend() - 1);
                }

                return;
            }

            idx++;
        }

        edges.emplace_back(other_neuron_id, weight);
    }

    NeuronDistantInNeighborhood neuron_distant_in_neighborhood{};
    NeuronDistantOutNeighborhood neuron_distant_out_neighborhood{};

    NeuronLocalInNeighborhood neuron_local_in_neighborhood{};
    NeuronLocalOutNeighborhood neuron_local_out_neighborhood{};

    size_t number_local_neurons{ Constants::uninitialized }; // My number of neurons
    int mpi_rank{ -1 };
};
