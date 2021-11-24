/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NetworkGraph.h"

#include "../io/LogFiles.h"
#include "Neurons.h"

#include "spdlog/spdlog.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void NetworkGraph::add_edges(const std::vector<local_synapse>& local_edges, const std::vector<in_synapse>& in_edges, const std::vector<out_synapse>& out_edges) {
    for (const auto& [source_id, target_id, weight] : local_edges) {
        LocalEdges& in_edges = neuron_local_in_neighborhood[target_id];
        LocalEdges& out_edges = neuron_local_out_neighborhood[source_id];

        add_edge<LocalEdges, LocalEdgesKey>(in_edges, source_id, weight);
        add_edge<LocalEdges, LocalEdgesKey>(out_edges, target_id, weight);
    }

    for (const auto& [source_rni, target_id, weight] : in_edges) {
        DistantEdges& distant_in_edges = neuron_distant_in_neighborhood[target_id];
        add_edge<DistantEdges, DistantEdgesKey>(distant_in_edges, source_rni, weight);
    }

    for (const auto& [source_id, target_rni, weight] : out_edges) {
        DistantEdges& distant_out_edges = neuron_distant_out_neighborhood[source_id];
        add_edge<DistantEdges, DistantEdgesKey>(distant_out_edges, target_rni, weight);
    }
}
bool NetworkGraph::check_edges_from_file(const std::filesystem::path& path_synapses, const std::vector<size_t>& neuron_ids) {
    std::ifstream file_synapses(path_synapses, std::ios::binary | std::ios::in);

    std::set<size_t> ids_in_file{};

    for (std::string line{}; std::getline(file_synapses, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        size_t source_id = 0;
        size_t target_id = 0;
        int weight = 0;

        std::stringstream sstream(line);
        const bool success = (sstream >> source_id) && (sstream >> target_id) && (sstream >> weight);

        if (!success) {
            return false;
        }

        // The neurons start with 1
        source_id--;
        target_id--;

        ids_in_file.insert(source_id);
        ids_in_file.insert(target_id);
    }

    bool found_everything = true;

    std::for_each(ids_in_file.begin(), ids_in_file.end(), [&neuron_ids, &found_everything](size_t val) {
        const auto found = std::binary_search(neuron_ids.begin(), neuron_ids.end(), val);
        if (!found) {
            found_everything = false;
        }
    });

    return found_everything;
}

void NetworkGraph::debug_check() const {
    if (!Config::do_debug_checks) {
        return;
    }

    const auto my_rank = mpi_rank;

    // Golden map that stores all local edges
    std::map<std::pair<size_t, size_t>, EdgeWeight> edges{};

    for (size_t neuron_id = 0; neuron_id < my_num_neurons; neuron_id++) {
        const auto& local_out_edges = get_local_out_edges(neuron_id);
        const auto& distant_out_edges = get_distant_out_edges(neuron_id);

        for (const auto& [target_neuron_id, edge_val] : local_out_edges) {
            RelearnException::check(edge_val != 0, "NetworkGraph::debug_check: Value is zero (out)");
            edges[std::make_pair(neuron_id, target_neuron_id)] = edge_val;
        }
    }

    for (size_t neuron_id = 0; neuron_id < my_num_neurons; neuron_id++) {
        const auto local_in_edges = get_local_in_edges(neuron_id);
        const auto distant_in_edges = get_distant_in_edges(neuron_id);

        for (const auto& [source_neuron_id, edge_val] : local_in_edges) {
            RelearnException::check(edge_val != 0, "NetworkGraph::debug_check: Value is zero (out)");

            const std::pair<size_t, size_t> id_pair(source_neuron_id, neuron_id);
            const auto it = edges.find(id_pair);

            const bool found = it != edges.cend();

            RelearnException::check(found, "NetworkGraph::debug_check: Edge not found");

            const int golden_weight = it->second;
            const bool weight_matches = golden_weight == edge_val;

            RelearnException::check(weight_matches, "NetworkGraph::debug_check: Weight doesn't match");

            edges.erase(id_pair);
        }
    }

    RelearnException::check(edges.empty(), "NetworkGraph::debug_check: Edges is not empty");
}

void NetworkGraph::print(std::ostream& os, const std::unique_ptr<NeuronsExtraInfo>& informations) const {
    const int my_rank = mpi_rank;

    // For my neurons
    for (size_t target_neuron_id = 0; target_neuron_id < my_num_neurons; target_neuron_id++) {
        // Walk through in-edges of my neuron
        RankNeuronId rank_neuron_id{ my_rank, target_neuron_id };

        const auto global_target = informations->rank_neuron_id2glob_id(rank_neuron_id);

        for (const auto& [local_source_id, edge_val] : neuron_local_in_neighborhood[target_neuron_id]) {
            os
                << (global_target + 1) << "\t"
                << (local_source_id + 1) << "\t"
                << edge_val << "\n";
        }

        for (const auto& [distant_neuron_id, edge_val] : neuron_distant_in_neighborhood[target_neuron_id]) {
            const auto global_source = informations->rank_neuron_id2glob_id(distant_neuron_id);

            // <target neuron id>  <source neuron id>  <weight>
            os
                << (global_target + 1) << "\t"
                << (global_source + 1) << "\t"
                << edge_val << "\n";
        }
    }
}
