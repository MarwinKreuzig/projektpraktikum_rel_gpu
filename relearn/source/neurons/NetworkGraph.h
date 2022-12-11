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
#include "Types.h"
#include "neurons/SignalType.h"
#include "util/MPIRank.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"

#include <filesystem>
#include <ostream>
#include <type_traits>
#include <unordered_map>
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
    using synapse_weight = RelearnTypes::synapse_weight;

    using DistantEdgesKey = RankNeuronId; // Pair of (mpi rank, local neuron id)
    using DistantEdges = std::vector<std::pair<DistantEdgesKey, synapse_weight>>;

    using NeuronDistantInNeighborhood = std::vector<DistantEdges>;
    using NeuronDistantOutNeighborhood = std::vector<DistantEdges>;

    using LocalEdges = std::vector<std::pair<NeuronID, synapse_weight>>;

    using NeuronLocalInNeighborhood = std::vector<LocalEdges>;
    using NeuronLocalOutNeighborhood = std::vector<LocalEdges>;

    using number_neurons_type = RelearnTypes::number_neurons_type;

    enum class EdgeDirection {
        In,
        Out
    };

    /**
     * @brief Constructs an object that has enough space to store the given number of neurons
     * @param number_neurons The number of neurons that the object shall handle
     * @param mpi_rank The mpi rank that handles this portion of the graph
     * @exception Throws an exception if the allocation of memory fails
     */
    [[deprecated]] NetworkGraph(const number_neurons_type number_neurons, const int mpi_rank)
        : neuron_distant_in_neighborhood(number_neurons)
        , neuron_distant_out_neighborhood(number_neurons)
        , neuron_local_in_neighborhood(number_neurons)
        , neuron_local_out_neighborhood(number_neurons)
        , number_local_neurons(number_neurons)
        , my_rank(mpi_rank) {
        RelearnException::check(my_rank.is_initialized(), "NetworkGraph::NetworkGraph: The mpi rank must be initialized");
    }

    /**
     * @brief Constructs an object that has enough space to store the given number of neurons
     * @param number_neurons The number of neurons that the object shall handle
     * @param mpi_rank The mpi rank that handles this portion of the graph, must be initialized
     * @exception Throws an exception if the allocation of memory fails
     */
    NetworkGraph(const number_neurons_type number_neurons, const MPIRank mpi_rank)
        : neuron_distant_in_neighborhood(number_neurons)
        , neuron_distant_out_neighborhood(number_neurons)
        , neuron_local_in_neighborhood(number_neurons)
        , neuron_local_out_neighborhood(number_neurons)
        , number_local_neurons(number_neurons)
        , my_rank(mpi_rank) {
        RelearnException::check(my_rank.is_initialized(), "NetworkGraph::NetworkGraph: The mpi rank must be initialized");
    }

    /**
     * @brief Resizes the network graph by adding space for more neurons. Invalidates iterators
     * @param creation_count The number of additional neurons the network graph should handle
     * @exception Throws an exception if the allocation of memory fails
     */
    void create_neurons(const number_neurons_type creation_count) {
        const auto old_size = number_local_neurons;
        const auto new_size = old_size + creation_count;

        neuron_distant_in_neighborhood.resize(new_size);
        neuron_distant_out_neighborhood.resize(new_size);

        neuron_local_in_neighborhood.resize(new_size);
        neuron_local_out_neighborhood.resize(new_size);

        number_local_neurons = new_size;
    }

    /**
     * @brief Returns a constant reference to all distant in-edges to a neuron, i.e., a view on neurons that connect to the specified one via a synapse
     *      and belong to another MPI rank
     * @param neuron_id The id of the neuron
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all distant in-edges.
     */
    [[nodiscard]] const DistantEdges& get_distant_in_edges(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < neuron_distant_in_neighborhood.size(),
            "NetworkGraph::get_distant_in_edges: Tried with a too large id of {}", neuron_id);

        return neuron_distant_in_neighborhood[local_neuron_id];
    }

    /**
     * @brief Returns a constant reference to all distant out-edges to a neuron, i.e., a view on all neurons that the specified one connectes to via a synapse
     *      and belong to another MPI rank
     * @param neuron_id The id of the neuron
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all distant out-edges.
     */
    [[nodiscard]] const DistantEdges& get_distant_out_edges(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < neuron_distant_out_neighborhood.size(),
            "NetworkGraph::get_distant_out_edges: Tried with a too large id of {}", neuron_id);

        return neuron_distant_out_neighborhood[local_neuron_id];
    }

    /**
     * @brief Returns a constant reference to all local in-edges to a neuron, i.e., a view on neurons that connect to the specified one via a synapse
     *      and belong to the current MPI rank
     * @param neuron_id The id of the neuron
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all local in-edges.
     */
    [[nodiscard]] const LocalEdges& get_local_in_edges(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < neuron_local_in_neighborhood.size(),
            "NetworkGraph::get_local_in_edges: Tried with a too large id of {}", neuron_id);

        return neuron_local_in_neighborhood[local_neuron_id];
    }

    /**
     * @brief Returns a constant reference to all local out-edges to a neuron, i.e., a view on all neurons that the specified one connectes to via a synapse
     *      and belong to the current MPI rank
     * @param neuron_id The id of the neuron
     * @exception Throws a RelearnException if neuron_id is larger or equal to the number of neurons stored
     * @return A constant view of all local out-edges.
     */
    [[nodiscard]] const LocalEdges& get_local_out_edges(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < neuron_local_out_neighborhood.size(),
            "NetworkGraph::get_local_out_edges: Tried with a too large id of {}", neuron_id);

        return neuron_local_out_neighborhood[local_neuron_id];
    }

    /**
     * @brief Returns a constant reference to all local out-edges of all neurons on this mpi rank
     * @return Vector of edges. Edges from neuron with id i are at position i
     */
    [[nodiscard]] const NeuronLocalOutNeighborhood& get_all_local_out_edges() const {
        return neuron_local_out_neighborhood;
    }

    /**
     * @brief Returns a constant reference to all distant out-edges of all neurons on this mpi rank
     * @return Vector of edges. Edges from neuron with id i are at position i
     */
    [[nodiscard]] const NeuronDistantOutNeighborhood& get_all_distant_out_edges() const {
        return neuron_distant_out_neighborhood;
    }

    /**
     * @brief Returns a constant reference to all local in-edges of all neurons on this mpi rank
     * @return Vector of edges. Edges from neuron with id i are at position i
     */
    [[nodiscard]] const NeuronLocalInNeighborhood& get_all_local_in_edges() const {
        return neuron_local_in_neighborhood;
    }

    /**
     * @brief Returns a constant reference to all distant in-edges of all neurons on this mpi rank
     * @return Vector of edges. Edges from neuron with id i are at position i
     */
    [[nodiscard]] const NeuronDistantInNeighborhood& get_all_distant_in_edges() const {
        return neuron_distant_in_neighborhood;
    }

    /**
     * @brief Returns a copy of all in-edges to a neuron, i.e., a copy of all neurons that connect to the specified one via a synapse, of a specified type.
     *      All local in-edges are added with the current MPI rank.
     * @param neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     *      Throws an exception if the allocation of memory fails
     * @return A copy of all in-edges from a certain neuron signal type
     */
    [[nodiscard]] DistantEdges get_all_in_edges(const NeuronID& neuron_id, const SignalType signal_type) const {
        const DistantEdges& all_distant_edges = get_distant_in_edges(neuron_id);
        const LocalEdges& all_local_edges = get_local_in_edges(neuron_id);

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
     * @param neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     *      Throws an exception if the allocation of memory fails
     * @return A copy of all out-edges to a certain neuron signal type
     */
    [[nodiscard]] DistantEdges get_all_out_edges(const NeuronID& neuron_id, const SignalType signal_type) const {
        const DistantEdges& all_distant_edges = get_distant_out_edges(neuron_id);
        const LocalEdges& all_local_edges = get_local_out_edges(neuron_id);

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
     * @param neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     *      Throws an exception if the allocation of memory fails
     * @return A copy of all in-edges
     */
    [[nodiscard]] DistantEdges get_all_in_edges(const NeuronID& neuron_id) const {
        const DistantEdges& all_distant_edges = get_distant_in_edges(neuron_id);
        const LocalEdges& all_local_edges = get_local_in_edges(neuron_id);

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
     * @param neuron_id The id of the neuron
     * @param signal_type The type of neurons that should be returned
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     *      Throws an exception if the allocation of memory fails
     * @return A copy of all out-edges
     */
    [[nodiscard]] DistantEdges get_all_out_edges(const NeuronID& neuron_id) const {
        const DistantEdges& all_distant_edges = get_distant_out_edges(neuron_id);
        const LocalEdges& all_local_edges = get_local_out_edges(neuron_id);

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
     * @param neuron_id The id of the neuron
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     * @return The number of incoming synapses that the specified neuron formed from excitatory neurons
     */
    [[nodiscard]] synapse_weight get_number_excitatory_in_edges(const NeuronID& neuron_id) const {
        const DistantEdges& all_distant_edges = get_distant_in_edges(neuron_id);
        const LocalEdges& all_local_edges = get_local_in_edges(neuron_id);

        synapse_weight total_num_ports = 0;

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
     * @param neuron_id The id of the neuron
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     * @return The number of incoming synapses that the specified neuron formed from inhibitory neurons
     */
    [[nodiscard]] synapse_weight get_number_inhibitory_in_edges(const NeuronID& neuron_id) const {
        const DistantEdges& all_distant_edges = get_distant_in_edges(neuron_id);
        const LocalEdges& all_local_edges = get_local_in_edges(neuron_id);

        synapse_weight total_num_ports = 0;

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
     * @param neuron_id The id of the neuron
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons stored
     * @return The number of outgoing synapses that the specified neuron formed
     */
    [[nodiscard]] synapse_weight get_number_out_edges(const NeuronID& neuron_id) const {
        const DistantEdges& all_distant_edges = get_distant_out_edges(neuron_id);
        const LocalEdges& all_local_edges = get_local_out_edges(neuron_id);

        synapse_weight total_num_ports = 0;

        for (const auto& [_, connection_strength] : all_distant_edges) {
            total_num_ports += std::abs(connection_strength);
        }

        for (const auto& [_, connection_strength] : all_local_edges) {
            total_num_ports += std::abs(connection_strength);
        }

        return total_num_ports;
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

        const auto local_target_id = target.get_neuron_id();
        const auto local_source_id = source.get_neuron_id();

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
        const auto local_target_id = target.get_neuron_id();

        const auto& [source_rank, source_id] = source_rni;

        RelearnException::check(local_target_id < number_local_neurons, "NetworkGraph::add_synapse: Distant in-synapse had a too large target: {} vs {}", target, number_local_neurons);
        RelearnException::check(source_rank != my_rank, "NetworkGraph::add_synapse: Distant in-synapse was on my rank: {}", source_rank);
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
        const auto local_source_id = source.get_neuron_id();

        const auto& [target_rank, target_id] = target_rni;

        RelearnException::check(local_source_id < number_local_neurons, "NetworkGraph::add_synapse: Distant out-synapse had a too large target: {} vs {}", source, number_local_neurons);
        RelearnException::check(target_rank != my_rank, "NetworkGraph::add_synapse: Distant out-synapse was on my rank: {}", target_rank);
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
            const auto local_target_id = target_id.get_neuron_id();
            const auto local_source_id = source_id.get_neuron_id();

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
            const auto local_target_id = target_id.get_neuron_id();

            RelearnException::check(local_target_id < neuron_distant_in_neighborhood.size(),
                "NetworkGraph::add_edges: distant_in_neighborhood is too small: {} vs {}", target_id, neuron_distant_in_neighborhood.size());

            DistantEdges& distant_in_edges = neuron_distant_in_neighborhood[local_target_id];
            add_edge<DistantEdges, DistantEdgesKey>(distant_in_edges, source_rni, weight);
        }

        for (const auto& [target_rni, source_id, weight] : out_edges) {
            const auto local_source_id = source_id.get_neuron_id();

            RelearnException::check(local_source_id < neuron_distant_out_neighborhood.size(),
                "NetworkGraph::add_edges: distant_out_neighborhood is too small: {} vs {}", source_id, neuron_distant_out_neighborhood.size());

            DistantEdges& distant_out_edges = neuron_distant_out_neighborhood[local_source_id];
            add_edge<DistantEdges, DistantEdgesKey>(distant_out_edges, target_rni, weight);
        }
    }

    /**
     * @brief Returns a histogram of the local neurons' connectivity.
     *      Casts the weights to size_t 
     * @param edge_direction An enum that indicates if in-edges or out-edges should be considered
     * @exception Throws an exception if the allocation of memory fails
     * @return A histogram of the connectivity, i.e., <return>[i] == c indicates that c local neurons have i edges in the requested direction
     */
    [[nodiscard]] std::vector<unsigned int> get_edges_histogram(EdgeDirection edge_direction) const {
        std::vector<unsigned int> result{};

        auto largest_number_edges = 0.0;

        const auto& local_neighborhood = (edge_direction == EdgeDirection::In) ? neuron_local_in_neighborhood : neuron_local_out_neighborhood;
        const auto& distant_neighborhood = (edge_direction == EdgeDirection::In) ? neuron_distant_in_neighborhood : neuron_distant_out_neighborhood;

        for (auto neuron_id = 0; neuron_id < number_local_neurons; neuron_id++) {
            auto number_of_connections = 0.0;

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
                result.resize(static_cast<size_t>(number_of_connections) * 2ULL + 1);
            }

            largest_number_edges = std::max(number_of_connections, largest_number_edges);

            result[static_cast<size_t>(number_of_connections)]++;
        }

        result.resize(static_cast<size_t>(largest_number_edges) + 1ULL);
        return result;
    }

    /**
     * @brief Prints all stored connections to the streams. Does not perform communication via MPI. Uses the local neuron ids and starts with 1. The formats are:
     *      <target_rank> <target_id>\t<source_rank> <source_id>\t<weight>\t<flag>
     * @param os_out_edges The out-stream to which the out-connections are printed
     * @param os_in_edges The out-stream to which the in-connections are printed
     * @param flag Boolean flag which will be printed at the end of the line of each connection. Usually indicates the plasticity
     */
    void print_with_ranks(std::ostream& os_out_edges, std::ostream& os_in_edges, const bool flag) const {
        for (const auto& source_id : NeuronID::range(number_local_neurons)) {
            const auto& source_local_id = source_id.get_neuron_id();

            for (const auto& [target_id, weight] : neuron_local_out_neighborhood[source_local_id]) {
                const auto& target_local_id = target_id.get_neuron_id();

                os_out_edges << my_rank.get_rank() << ' ' << (target_local_id + 1) << '\t' << my_rank.get_rank() << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << flag << '\n';
            }

            for (const auto& [target_neuron, weight] : neuron_distant_out_neighborhood[source_local_id]) {
                const auto& [target_rank, target_id] = target_neuron;
                const auto& target_local_id = target_id.get_neuron_id();

                os_out_edges << target_rank.get_rank() << ' ' << (target_local_id + 1) << '\t' << my_rank.get_rank() << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << flag << '\n';
            }
        }

        for (const auto& target_id : NeuronID::range(number_local_neurons)) {
            const auto& target_local_id = target_id.get_neuron_id();

            for (const auto& [source_id, weight] : neuron_local_in_neighborhood[target_local_id]) {
                const auto& source_local_id = source_id.get_neuron_id();

                os_in_edges << my_rank.get_rank() << ' ' << (target_local_id + 1) << '\t' << my_rank.get_rank() << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << flag << '\n';
            }

            for (const auto& [source_neuron, weight] : neuron_distant_in_neighborhood[target_local_id]) {
                const auto& [source_rank, source_id] = source_neuron;
                const auto& source_local_id = source_id.get_neuron_id();

                os_in_edges << my_rank.get_rank() << ' ' << (target_local_id + 1) << '\t' << source_rank.get_rank() << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << flag << '\n';
            }
        }
    }

    /**
     * @brief Returns directly if !Config::do_debug_checks
     *      Performs a debug check on the local portion of the network graph
     *      All stored ranks must be greater or equal to zero, no weight must be equal to zero,
     *      and all purely local edges must have a matching counterpart.
     * @exception Throws a RelearnException if any of the conditions is violated
     */
    void debug_check() const {
        if (!Config::do_debug_checks) {
            return;
        }

        struct NeuronIDPairHash {
        public:
            std::size_t operator()(const std::pair<NeuronID, NeuronID>& pair) const {
                const std::hash<NeuronID> primitive_hash{};

                const auto& [first_id, second_id] = pair;

                const auto first_hash = primitive_hash(first_id);
                const auto second_hash = primitive_hash(second_id);

                // XOR might not be the best, but this is debug code
                const auto combined_hash = first_hash ^ second_hash;
                return combined_hash;
            }
        };

        for (const auto& neuron_id : NeuronID::range(number_local_neurons)) {
            const auto& distant_out_edges = get_distant_out_edges(neuron_id);

            for (const auto& [target_id, edge_val] : distant_out_edges) {
                const auto& [target_rank, target_neuron_id] = target_id;

                RelearnException::check(edge_val != 0, "NetworkGraph::debug_check: Distant synapse value is zero (out)");
                RelearnException::check(target_rank.is_initialized(), "NetworkGraph::debug_check: Distant synapse target rank is < 0");
                RelearnException::check(target_rank != my_rank, "NetworkGraph::debug_check: Distant synapse target rank is the local rank");
            }
        }

        for (const auto& neuron_id : NeuronID::range(number_local_neurons)) {
            const auto& distant_in_edges = get_distant_in_edges(neuron_id);

            for (const auto& [source_id, edge_val] : distant_in_edges) {
                const auto& [source_rank, source_neuron_id] = source_id;

                RelearnException::check(edge_val != 0, "NetworkGraph::debug_check: Distant synapse value is zero (out)");
                RelearnException::check(source_rank.is_initialized(), "NetworkGraph::debug_check: Distant synapse source rank is < 0");
                RelearnException::check(source_rank != my_rank, "NetworkGraph::debug_check: Distant synapse source rank is the local rank");
            }
        }

        // Golden map that stores all local edges
        std::unordered_map<std::pair<NeuronID, NeuronID>, RelearnTypes::synapse_weight, NeuronIDPairHash> edges{};
        edges.reserve(number_local_neurons);

        for (const auto& neuron_id : NeuronID::range(number_local_neurons)) {
            const auto& local_out_edges = get_local_out_edges(neuron_id);

            for (const auto& [target_neuron_id, edge_val] : local_out_edges) {
                RelearnException::check(edge_val != 0, "NetworkGraph::debug_check: Value is zero (out)");
                edges[std::make_pair(neuron_id, target_neuron_id)] = edge_val;
            }
        }

        for (const auto& id : NeuronID::range(number_local_neurons)) {
            const auto& local_in_edges = get_local_in_edges(id);

            for (const auto& [source_neuron_id, edge_val] : local_in_edges) {
                RelearnException::check(edge_val != 0, "NetworkGraph::debug_check: Value is zero (out)");

                const std::pair<NeuronID, NeuronID> id_pair(source_neuron_id, id);
                const auto it = edges.find(id_pair);

                const auto found = it != edges.cend();

                RelearnException::check(found, "NetworkGraph::debug_check: Edge not found");

                const auto golden_weight = it->second;
                const auto weight_matches = golden_weight == edge_val;

                RelearnException::check(weight_matches, "NetworkGraph::debug_check: Weight doesn't match");

                edges.erase(id_pair);
            }
        }

        RelearnException::check(edges.empty(), "NetworkGraph::debug_check: Edges is not empty");
    }

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

    number_neurons_type number_local_neurons{ Constants::uninitialized }; // My number of neurons
    MPIRank my_rank{ MPIRank::uninitialized_rank() };
};
