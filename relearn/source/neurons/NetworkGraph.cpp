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
#include "Neurons.h"

#include "spdlog/spdlog.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

bool NetworkGraph::check_edges_from_file(const std::filesystem::path& path_synapses, const std::vector<size_t>& neuron_ids) {
    std::ifstream file_synapses(path_synapses, std::ios::binary | std::ios::in);

    std::set<size_t> ids_in_file{};

    for (std::string line{}; std::getline(file_synapses, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
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
    std::map<std::pair<NeuronID, NeuronID>, RelearnTypes::synapse_weight> edges{};

    for (auto neuron_id : NeuronID::range(number_local_neurons)) {
        const auto& local_out_edges = get_local_out_edges(neuron_id);
        const auto& distant_out_edges = get_distant_out_edges(neuron_id);

        for (const auto& [target_neuron_id, edge_val] : local_out_edges) {
            RelearnException::check(edge_val != 0, "NetworkGraph::debug_check: Value is zero (out)");
            edges[std::make_pair(neuron_id, target_neuron_id)] = edge_val;
        }
    }

    for (auto id : NeuronID::range(number_local_neurons)) {
        const auto local_in_edges = get_local_in_edges(id);
        const auto distant_in_edges = get_distant_in_edges(id);

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

void NetworkGraph::print(std::ostream& os, const std::shared_ptr<NeuronIdTranslator>& translator) const {
    // TODO(fabian): Write print here
}

void NetworkGraph::print_with_ranks(std::ostream& os_out_edges, std::ostream& os_in_edges) const noexcept {
    const auto my_rank = mpi_rank;

    for (const auto& source_id : NeuronID::range(number_local_neurons)) {
        const auto& source_local_id = source_id.get_local_id();

        for (const auto& [target_id, weight] : neuron_local_out_neighborhood[source_local_id]) {
            const auto& target_local_id = target_id.get_local_id();

            os_out_edges << my_rank << ' ' << (target_local_id + 1) << '\t' << my_rank << ' ' << source_local_id << '\t' << weight << '\n';
        }

        for (const auto& [target_neuron, weight] : neuron_distant_out_neighborhood[source_local_id]) {
            const auto& [target_rank, target_id] = target_neuron;
            const auto& target_local_id = source_id.get_local_id();

            os_out_edges << target_rank << ' ' << (target_local_id + 1) << '\t' << my_rank << ' ' << source_local_id << '\t' << weight << '\n';
        }
    }

    for (const auto& target_id : NeuronID::range(number_local_neurons)) {
        const auto& target_local_id = target_id.get_local_id();

        for (const auto& [source_id, weight] : neuron_local_in_neighborhood[target_local_id]) {
            const auto& source_local_id = source_id.get_local_id();

            os_in_edges << my_rank << ' ' << (target_local_id + 1) << '\t' << my_rank << ' ' << source_local_id << '\t' << weight << '\n';
        }

        for (const auto& [source_neuron, weight] : neuron_distant_in_neighborhood[target_local_id]) {
            const auto& [source_rank, source_id] = source_neuron;
            const auto& source_local_id = source_id.get_local_id();

            os_in_edges << my_rank << ' ' << (target_local_id + 1) << '\t' << source_rank << ' ' << source_local_id << '\t' << weight << '\n';
        }
    }
}
