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

#include "../util/Vec3.h"
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
  * Network graph stores in and out edges for the neuron id range
  * an MPI process is responsible for: [my_neuron_id_start, my_neuron_id_end].
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

    explicit NetworkGraph(size_t my_num_neurons);

    // Return in edges of neuron "neuron_id"
    [[nodiscard]] const Edges& get_in_edges(size_t neuron_id) const /*noexcept*/;

    // Return out edges of neuron "neuron_id"
    [[nodiscard]] const Edges& get_out_edges(size_t neuron_id) const /*noexcept*/;

    // Return in edges of neuron "neuron_id"
    [[nodiscard]] Edges get_in_edges(size_t neuron_id, SignalType signal_type) const /*noexcept*/;

    // Return out edges of neuron "neuron_id"
    [[nodiscard]] Edges get_out_edges(size_t neuron_id, SignalType signal_type) const /*noexcept*/;

    [[nodiscard]] size_t get_num_in_edges(size_t neuron_id) const;

    [[nodiscard]] size_t get_num_in_edges_ex(size_t neuron_id) const;

    [[nodiscard]] size_t get_num_in_edges_in(size_t neuron_id) const;

    [[nodiscard]] size_t get_num_out_edges(size_t neuron_id) const;

    void create_neurons(size_t creation_count);

    /**
	 * Add weight to an edge.
	 * The edge is created if it does not exist yet and parameter weight != 0.
	 * The weight is added to the current weight.
	 * The edge is deleted if its weight becomes 0.
	 */
    void add_edge_weight(size_t target_neuron_id, int target_rank,
        size_t source_neuron_id, int source_rank,
        int weight);

    // Print network using global neuron ids
    void print(std::ostream& os, const std::unique_ptr<NeuronsExtraInfo>& informations) const;

    void add_edges_from_file(const std::string& path_synapses, const std::string& path_neurons, const Partition& partition);

    /*
    * Checks if the specified file contains only synapses between neurons with ids in neuron_ids.
    * Requires neuron_ids to be sorted ascending.
    * Returns true iff the file has the correct format and only ids in neuron_ids are present
    */
    [[nodiscard]] static bool check_edges_from_file(const std::string& path_synapses, const std::vector<size_t>& neuron_ids);

    void debug_check() const;

private:
    // NOLINTNEXTLINE
    static void add_edge(Edges& edges, int rank, size_t neuron_id, int weight);

    // NOLINTNEXTLINE
    static void translate_global_to_local(const std::map<size_t, int>& id_to_rank, const Partition& partition, std::map<size_t, size_t>& global_id_to_local_id);

    // NOLINTNEXTLINE
    static void load_neuron_positions(const std::string& path_neurons, std::set<size_t>& foreing_ids, std::map<size_t, Vec3d>& id_to_pos);

    // NOLINTNEXTLINE
    static void load_synapses(const std::string& path_synapses, const Partition& partition, std::set<size_t>& foreing_ids, std::vector<std::tuple<size_t, size_t, int>>& local_synapses, std::vector<std::tuple<size_t, size_t, int>>& out_synapses, std::vector<std::tuple<size_t, size_t, int>>& in_synapses);

    void write_synapses_to_file(const std::string& filename, const Partition& partition) const;

    NeuronInNeighborhood neuron_in_neighborhood;
    NeuronOutNeighborhood neuron_out_neighborhood;

    size_t my_num_neurons; // My number of neurons
};
