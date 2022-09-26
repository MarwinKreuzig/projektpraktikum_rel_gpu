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

#include "io/LogFiles.h"
#include "neurons/Neurons.h"

#include "spdlog/spdlog.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

bool NetworkGraph::check_edges_from_file(const std::filesystem::path& path_synapses, const std::vector<NeuronID::value_type>& neuron_ids) {
    std::ifstream file_synapses(path_synapses, std::ios::binary | std::ios::in);

    std::set<NeuronID::value_type> ids_in_file{};

    for (std::string line{}; std::getline(file_synapses, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        NeuronID::value_type source_id = 0;
        NeuronID::value_type target_id = 0;
        RelearnTypes::synapse_weight weight = 0;

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

    return std::ranges::all_of(ids_in_file, [&neuron_ids](NeuronID::value_type val) {
        return std::ranges::binary_search(neuron_ids, val);
    });
}

void NetworkGraph::debug_check() const {
    if (!Config::do_debug_checks) {
        return;
    }

    const auto my_rank = mpi_rank;

    // Golden map that stores all local edges
    std::map<std::pair<NeuronID, NeuronID>, RelearnTypes::synapse_weight> edges{};

    for (const auto& neuron_id : NeuronID::range(number_local_neurons)) {
        const auto& local_out_edges = get_local_out_edges(neuron_id);

        for (const auto& [target_neuron_id, edge_val] : local_out_edges) {
            RelearnException::check(edge_val != 0, "NetworkGraph::debug_check: Value is zero (out)");
            edges[std::make_pair(neuron_id, target_neuron_id)] = edge_val;
        }
    }

    for (const auto& id : NeuronID::range(number_local_neurons)) {
        const auto& local_in_edges = get_local_in_edges(id);
        const auto& distant_in_edges = get_distant_in_edges(id);

        for (const auto& [source_neuron_id, edge_val] : local_in_edges) {
            RelearnException::check(edge_val != 0, "NetworkGraph::debug_check: Value is zero (out)");

            const std::pair<NeuronID, NeuronID> id_pair(source_neuron_id, id);
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

void NetworkGraph::print_with_ranks(std::ostream& os_out_edges, std::ostream& os_in_edges) const {
    const auto my_rank = mpi_rank;

    for (const auto& source_id : NeuronID::range(number_local_neurons)) {
        const auto& source_local_id = source_id.get_neuron_id();

        for (const auto& [target_id, weight] : neuron_local_out_neighborhood[source_local_id]) {
            const auto& target_local_id = target_id.get_neuron_id();

            os_out_edges << my_rank << ' ' << (target_local_id + 1) << '\t' << my_rank << ' ' << (source_local_id + 1) << '\t' << weight << '\n';
        }

        for (const auto& [target_neuron, weight] : neuron_distant_out_neighborhood[source_local_id]) {
            const auto& [target_rank, target_id] = target_neuron;
            const auto& target_local_id = source_id.get_neuron_id();

            os_out_edges << target_rank << ' ' << (target_local_id + 1) << '\t' << my_rank << ' ' << (source_local_id + 1) << '\t' << weight << '\n';
        }
    }

    for (const auto& target_id : NeuronID::range(number_local_neurons)) {
        const auto& target_local_id = target_id.get_neuron_id();

        for (const auto& [source_id, weight] : neuron_local_in_neighborhood[target_local_id]) {
            const auto& source_local_id = source_id.get_neuron_id();

            os_in_edges << my_rank << ' ' << (target_local_id + 1) << '\t' << my_rank << ' ' << (source_local_id + 1) << '\t' << weight << '\n';
        }

        for (const auto& [source_neuron, weight] : neuron_distant_in_neighborhood[target_local_id]) {
            const auto& [source_rank, source_id] = source_neuron;
            const auto& source_local_id = source_id.get_neuron_id();

            os_in_edges << my_rank << ' ' << (target_local_id + 1) << '\t' << source_rank << ' ' << (source_local_id + 1) << '\t' << weight << '\n';
        }
    }
}
