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
#include "../structure/NeuronIdTranslator.h"
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
    std::map<std::pair<NeuronID, NeuronID>, EdgeWeight> edges{};

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
    const auto num_ranks = MPIWrapper::get_num_ranks();

    std::map<int, std::set<size_t>> required_ids{};

    for (size_t target_neuron_id = 0; target_neuron_id < number_local_neurons; target_neuron_id++) {
        const auto& in_edges = neuron_distant_in_neighborhood[target_neuron_id];

        for (const auto& [distand_neuron_id, weight] : in_edges) {
            const auto& [rank, local_neuron_id] = distand_neuron_id;
            required_ids[rank].emplace(local_neuron_id);
        }
    }

    std::vector<std::vector<size_t>> exchange_id_my_requests(num_ranks);

    for (const auto& [rank, local_neuron_ids] : required_ids) {
        auto& requests_vector = exchange_id_my_requests[rank];
        const auto& requests_set = required_ids[rank];

        requests_vector.insert(requests_vector.cend(), requests_set.begin(), requests_set.end());
    }

    const auto& exchange_ids_others_requests = MPIWrapper::exchange_values(exchange_id_my_requests);

    std::vector<std::vector<size_t>> exchange_id_my_responses(num_ranks);
    for (auto rank : MPIWrapper::get_ranks()) {
        for (const auto& local_id : exchange_ids_others_requests[rank]) {
            const auto& global_id = translator->get_global_id(NeuronID{ local_id });
            exchange_id_my_responses[rank].emplace_back(global_id);
        }
    }

    const auto& exchange_ids_other_responses = MPIWrapper::exchange_values(exchange_id_my_responses);

    // For my neurons
    for (auto target_neuron_id : NeuronID::range(number_local_neurons)) {
        const auto global_target_id = translator->get_global_id(target_neuron_id);

        for (const auto& [local_source_id, edge_val] : neuron_local_in_neighborhood[target_neuron_id.id()]) {
            const auto global_source_id = translator->get_global_id(local_source_id);

            os << (global_target_id + 1) << "\t"
               << (global_source_id + 1) << "\t"
               << edge_val << "\n";
        }

        for (const auto& [distant_neuron_id, edge_val] : neuron_distant_in_neighborhood[target_neuron_id.id()]) {
            const auto& [distant_rank, distant_local_neuron_id] = distant_neuron_id;

            const auto& request_iterator = std::find(exchange_id_my_requests[distant_rank].begin(),
                exchange_id_my_requests[distant_rank].end(), distant_local_neuron_id.id());

            const auto& distance = std::distance(exchange_id_my_requests[distant_rank].begin(), request_iterator);

            const auto global_source_id = exchange_ids_other_responses[distant_rank][distance];

            // <target neuron id>  <source neuron id>  <weight>
            os
                << (global_target_id + 1) << "\t"
                << (global_source_id + 1) << "\t"
                << edge_val << "\n";
        }
    }
}
