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
#include "../mpi/MPIWrapper.h"
#include "../structure/Partition.h"
#include "../util/RelearnException.h"
#include "Neurons.h"
#include "helper/RankNeuronId.h"
#include "spdlog/spdlog.h"

#include <algorithm>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

NetworkGraph::NetworkGraph(size_t num_neurons)
    : neuron_in_neighborhood(num_neurons)
    , neuron_out_neighborhood(num_neurons)
    , my_num_neurons(num_neurons) {
}

const NetworkGraph::Edges& NetworkGraph::get_in_edges(size_t neuron_id) const {
    RelearnException::check(neuron_id < neuron_in_neighborhood.size(), "In get_in_edges, tried with a too large id");
    return neuron_in_neighborhood[neuron_id];
}

const NetworkGraph::Edges& NetworkGraph::get_out_edges(size_t neuron_id) const {
    RelearnException::check(neuron_id < neuron_out_neighborhood.size(), "In get_out_edges, tried with a too large id");
    return neuron_out_neighborhood[neuron_id];
}

NetworkGraph::Edges NetworkGraph::get_in_edges(size_t neuron_id, SignalType signal_type) const {
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

NetworkGraph::Edges NetworkGraph::get_out_edges(size_t neuron_id, SignalType signal_type) const {
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

size_t NetworkGraph::get_num_in_edges_ex(size_t neuron_id) const {
    RelearnException::check(neuron_id < neuron_in_neighborhood.size(),
        "In get_num_in_edges, tried with a too large id: %u %u", neuron_id, my_num_neurons);

    size_t total_num_ports = 0;

    for (const auto& [_, connection_strength] : neuron_in_neighborhood[neuron_id]) {
        if (connection_strength > 0) {
            total_num_ports += connection_strength;
        }
    }

    return total_num_ports;
}

size_t NetworkGraph::get_num_in_edges_in(size_t neuron_id) const {
    RelearnException::check(neuron_id < neuron_in_neighborhood.size(),
        "In get_num_in_edges, tried with a too large id: %u %u", neuron_id, my_num_neurons);

    size_t total_num_ports = 0;

    for (const auto& [_, connection_strength] : neuron_in_neighborhood[neuron_id]) {
        if (connection_strength < 0) {
            total_num_ports += -connection_strength;
        }
    }

    return total_num_ports;
}

size_t NetworkGraph::get_num_out_edges(size_t neuron_id) const {
    RelearnException::check(neuron_id < neuron_out_neighborhood.size(),
        "In get_num_out_edges, tried with a too large id: %u %u", neuron_id, my_num_neurons);

    size_t total_num_ports = 0;

    for (const auto& [_, connection_strength] : neuron_out_neighborhood[neuron_id]) {
        total_num_ports += std::abs(connection_strength);
    }

    return total_num_ports;
}

void NetworkGraph::create_neurons(size_t creation_count) {
    const auto old_size = my_num_neurons;
    const auto new_size = old_size + creation_count;

    neuron_in_neighborhood.resize(new_size);
    neuron_out_neighborhood.resize(new_size);

    my_num_neurons = new_size;
}

void NetworkGraph::add_edge(Edges& edges, int rank, size_t neuron_id, int weight) {
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

void NetworkGraph::add_edge_weight(const RankNeuronId& target, const RankNeuronId& source, int weight) {
    if (weight == 0) {
        RelearnException::fail("weight of edge to add is zero");
        return;
    }

    const auto target_rank = target.get_rank();
    const auto target_neuron_id = target.get_neuron_id();

    const auto source_rank = source.get_rank();
    const auto source_neuron_id = source.get_neuron_id();

    const int my_rank = MPIWrapper::get_my_rank();

    // Target neuron is mine
    if (target_rank == my_rank) {
        RelearnException::check(target_neuron_id < my_num_neurons,
            "Want to add an in-edge with a too large target id: %u %u", target_neuron_id, my_num_neurons);

        Edges& in_edges = neuron_in_neighborhood[target_neuron_id];
        add_edge(in_edges, source_rank, source_neuron_id, weight);
    }

    if (source_rank == my_rank) {
        RelearnException::check(source_neuron_id < my_num_neurons,
            "Want to add an out-edge with a too large source id: ", target_neuron_id, my_num_neurons);

        Edges& out_edges = neuron_out_neighborhood[source_neuron_id];
        add_edge(out_edges, target_rank, target_neuron_id, weight);
    }

    if (target_rank != my_rank && source_rank != my_rank) {
        RelearnException::fail("In NetworkGraph::add_edge_weight, neither the target nor the source rank were for me.");
    }
}

void NetworkGraph::add_edges_from_file(const std::string& path_synapses, const std::string& path_neurons, const Partition& partition) {
    std::vector<std::tuple<size_t, size_t, int>> local_synapses{};
    std::vector<std::tuple<size_t, size_t, int>> out_synapses{};
    std::vector<std::tuple<size_t, size_t, int>> in_synapses{};

    std::set<size_t> foreing_ids{};

    std::map<size_t, int> id_to_rank{};
    std::map<size_t, Vec3d> id_to_pos{};

    const int my_rank = MPIWrapper::get_my_rank();

    load_synapses(path_synapses, partition, foreing_ids, local_synapses, out_synapses, in_synapses);
    load_neuron_positions(path_neurons, foreing_ids, id_to_pos);

    for (const auto& [id, pos] : id_to_pos) {
        const auto rank = static_cast<int>(partition.get_mpi_rank_from_pos(pos));
        id_to_rank[id] = rank;
    }

    std::map<size_t, size_t> global_id_to_local_id{};

    if (!id_to_pos.empty()) {
        translate_global_to_local(id_to_rank, partition, global_id_to_local_id);
    }

    for (const auto& [source_neuron_id, target_neuron_id, weight] : local_synapses) {
        const size_t translated_source_neuron_id = partition.get_local_id(source_neuron_id);
        const size_t translated_target_neuron_id = partition.get_local_id(target_neuron_id);

        const int source_rank = my_rank;
        const int target_rank = my_rank;

        const RankNeuronId target_id{ target_rank, translated_target_neuron_id };
        const RankNeuronId source_id{ source_rank, translated_source_neuron_id };

        add_edge_weight(target_id, source_id, weight);
    }

    for (const auto& [source_neuron_id, target_neuron_id, weight] : out_synapses) {
        const size_t translated_source_neuron_id = partition.get_local_id(source_neuron_id);

        const int source_rank = my_rank;
        const int target_rank = id_to_rank[target_neuron_id];
        const size_t translated_target_neuron_id = global_id_to_local_id[target_neuron_id];

        const RankNeuronId target_id{ target_rank, translated_target_neuron_id };
        const RankNeuronId source_id{ source_rank, translated_source_neuron_id };

        add_edge_weight(target_id, source_id, weight);
    }

    for (const auto& [source_neuron_id, target_neuron_id, weight] : in_synapses) {
        const size_t translated_target_neuron_id = partition.get_local_id(target_neuron_id);

        const int source_rank = id_to_rank[source_neuron_id];
        const int target_rank = my_rank;
        const size_t translated_source_neuron_id = global_id_to_local_id[source_neuron_id];

        const RankNeuronId target_id{ target_rank, translated_target_neuron_id };
        const RankNeuronId source_id{ source_rank, translated_source_neuron_id };

        add_edge_weight(target_id, source_id, weight);
    }

    const auto total_num_synapses = (local_synapses.size() + out_synapses.size() + in_synapses.size());

    LogFiles::write_to_file(LogFiles::EventType::Cout, true, "I'm rank: {} of {}.\n", my_rank, MPIWrapper::get_num_ranks());
    LogFiles::write_to_file(LogFiles::EventType::Cout, true, "I'v loaded: [local, out, in] {} + {} + {} = {} synapses.\n", local_synapses.size(), out_synapses.size(), in_synapses.size(), total_num_synapses);

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Loaded synapses: {}", total_num_synapses);
}

bool NetworkGraph::check_edges_from_file(const std::string& path_synapses, const std::vector<size_t>& neuron_ids) {
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

    const int my_rank = MPIWrapper::get_my_rank();

    // Golden map that stores all local edges
    std::map<std::pair<size_t, size_t>, int> edges{};

    for (size_t neuron_id = 0; neuron_id < my_num_neurons; neuron_id++) {
        const Edges& out_edges = get_out_edges(neuron_id);

        for (const auto& [out_edge_key, out_edge_val] : out_edges) {
            const int out_edge_rank = out_edge_key.first;
            const size_t out_edge_neuron_id = out_edge_key.second;

            RelearnException::check(out_edge_rank >= 0, "Rank is smaller than 0 (out)");
            RelearnException::check(out_edge_val != 0, "Value is zero (out)");

            if (out_edge_rank != my_rank) {
                continue;
            }

            edges[std::make_pair(neuron_id, out_edge_neuron_id)] = out_edge_val;
        }
    }

    for (size_t neuron_id = 0; neuron_id < my_num_neurons; neuron_id++) {
        const Edges& in_edges = get_in_edges(neuron_id);

        for (const auto& [in_edge_key, in_edge_val] : in_edges) {
            const auto [in_edge_rank, in_edge_neuron_id] = in_edge_key;

            RelearnException::check(in_edge_rank >= 0, "Rank is smaller than 0 (in)");
            RelearnException::check(in_edge_val != 0, "Value is zero (in)");

            if (in_edge_rank != my_rank) {
                continue;
            }

            const std::pair<size_t, size_t> id_pair(in_edge_neuron_id, neuron_id);
            const auto it = edges.find(id_pair);

            const bool found = it != edges.cend();

            RelearnException::check(found, "Edge not found");

            const int golden_weight = it->second;
            const bool weight_matches = golden_weight == in_edge_val;

            RelearnException::check(weight_matches, "Weight doesn't match");

            edges.erase(id_pair);
        }
    }

    RelearnException::check(edges.empty(), "Edges is not empty");
}

void NetworkGraph::translate_global_to_local(const std::map<size_t, int>& id_to_rank, const Partition& partition, std::map<size_t, size_t>& global_id_to_local_id) {

    const int num_ranks = MPIWrapper::get_num_ranks();

    std::vector<size_t> num_foreign_ids_from_ranks_send(num_ranks, 0);
    std::vector<size_t> num_foreign_ids_from_ranks(num_ranks, 0);

    std::vector<std::vector<size_t>> global_ids_to_send(num_ranks);
    std::vector<std::vector<size_t>> global_ids_to_receive(num_ranks);
    std::vector<std::vector<size_t>> global_ids_local_value(num_ranks);

    for (const auto& [id, rank] : id_to_rank) {
        num_foreign_ids_from_ranks_send[rank]++;
        global_ids_to_send[rank].emplace_back(id);
    }

    MPIWrapper::all_to_all(num_foreign_ids_from_ranks_send, num_foreign_ids_from_ranks);

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (MPIWrapper::get_my_rank() == rank) {
            RelearnException::check(global_ids_to_receive[rank].empty(), "Should receive ids from myself");
            continue;
        }

        global_ids_to_receive[rank].resize(num_foreign_ids_from_ranks[rank]);
    }

    std::vector<MPIWrapper::AsyncToken> mpi_requests(static_cast<size_t>(num_ranks) * 2 - 2);
    int request_counter = 0;

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (MPIWrapper::get_my_rank() == rank) {
            continue;
        }

        size_t* buffer = global_ids_to_receive[rank].data();
        const auto size_in_bytes = static_cast<int>(global_ids_to_receive[rank].size() * sizeof(size_t));

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (MPIWrapper::get_my_rank() == rank) {
            continue;
        }

        const size_t* buffer = global_ids_to_send[rank].data();
        const auto size_in_bytes = static_cast<int>(global_ids_to_send[rank].size() * sizeof(size_t));

        // Reserve enough space for the answer - it will be as long as the request
        global_ids_local_value[rank].resize(global_ids_to_send[rank].size());

        MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);
    for (auto& vec : global_ids_to_receive) {
        for (auto& global_id : vec) {
            global_id = partition.get_local_id(global_id);
        }
    }

    mpi_requests = std::vector<MPIWrapper::AsyncToken>(static_cast<size_t>(num_ranks) * 2 - 2);
    request_counter = 0;

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (MPIWrapper::get_my_rank() == rank) {
            continue;
        }

        size_t* buffer = global_ids_local_value[rank].data();
        const auto size_in_bytes = static_cast<int>(global_ids_local_value[rank].size() * sizeof(size_t));

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (MPIWrapper::get_my_rank() == rank) {
            continue;
        }

        const size_t* buffer = global_ids_to_receive[rank].data();
        const auto size_in_bytes = static_cast<int>(global_ids_to_receive[rank].size() * sizeof(size_t));

        MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);

    for (auto rank = 0; rank < num_ranks; rank++) {
        std::vector<size_t>& translated_ids = global_ids_local_value[rank];
        std::vector<size_t>& local_global_ids = global_ids_to_send[rank];

        RelearnException::check(translated_ids.size() == local_global_ids.size(), "The vectors have not the same size in load network");

        for (auto i = 0; i < translated_ids.size(); i++) {
            const size_t local_id = translated_ids[i];
            const size_t global_id = local_global_ids[i];

            global_id_to_local_id[global_id] = local_id;
        }
    }
}

void NetworkGraph::load_neuron_positions(const std::string& path_neurons, std::set<size_t>& foreing_ids, std::map<size_t, Vec3d>& id_to_pos) {
    std::string line;
    std::ifstream file_neurons(path_neurons, std::ios::binary | std::ios::in);

    while (std::getline(file_neurons, line)) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        size_t id{};
        double pos_x = 0.0;
        double pos_y = 0.0;
        double pos_z = 0.0;
        std::string area_name{};
        std::string type{};

        std::stringstream sstream(line);
        const bool success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> type);

        if (!success) {
            spdlog::info("Skipping line: \"{}\"", line);
            continue;
        }

        // File starts at 1
        id--;

        if (foreing_ids.find(id) != foreing_ids.end()) {
            id_to_pos[id] = { pos_x, pos_y, pos_z };
            foreing_ids.erase(id);

            if (foreing_ids.empty()) {
                break;
            }
        }
    }

    file_neurons.close();
}

void NetworkGraph::load_synapses(
    const std::string& path_synapses,
    const Partition& partition,
    std::set<size_t>& foreing_ids,
    std::vector<std::tuple<size_t, size_t, int>>& local_synapses,
    std::vector<std::tuple<size_t, size_t, int>>& out_synapses,
    std::vector<std::tuple<size_t, size_t, int>>& in_synapses) {
    enum class f_status : char {
        not_known = 0,
        local = 1,
        not_local = 2,
    };

    std::string line;

    std::vector<f_status> id_is_local(partition.get_total_num_neurons(), f_status::not_known);

    std::ifstream file_synapses(path_synapses, std::ios::binary | std::ios::in);

    while (std::getline(file_synapses, line)) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        size_t source_id = 0;
        size_t target_id = 0;
        int weight = 0;

        std::stringstream sstream(line);
        const bool success = (sstream >> source_id) && (sstream >> target_id) && (sstream >> weight);

        RelearnException::check(success, "Loading synapses was unsuccessfull!");

        // The neurons start with 1
        source_id--;
        target_id--;

        const f_status source_f = id_is_local[source_id];
        const f_status target_f = id_is_local[target_id];

        bool source_is_local = false;
        bool target_is_local = false;

        if (source_f == f_status::local) {
            source_is_local = true;
        } else if (source_f == f_status::not_local) {
            source_is_local = false;
        } else {
            source_is_local = partition.is_neuron_local(source_id);
            if (source_is_local) {
                id_is_local[source_id] = f_status::local;
            } else {
                id_is_local[source_id] = f_status::not_local;
            }
        }

        if (target_f == f_status::local) {
            target_is_local = true;
        } else if (target_f == f_status::not_local) {
            target_is_local = false;
        } else {
            target_is_local = partition.is_neuron_local(target_id);
            if (target_is_local) {
                id_is_local[target_id] = f_status::local;
            } else {
                id_is_local[target_id] = f_status::not_local;
            }
        }

        if (!source_is_local && !target_is_local) {
            continue;
        }

        if (source_is_local && target_is_local) {
            local_synapses.emplace_back(source_id, target_id, weight);
            continue;
        }

        if (source_is_local && !target_is_local) {
            out_synapses.emplace_back(source_id, target_id, weight);
            foreing_ids.emplace(target_id);
            continue;
        }

        if (!source_is_local && target_is_local) {
            in_synapses.emplace_back(source_id, target_id, weight);
            foreing_ids.emplace(source_id);
            continue;
        }

        RelearnException::fail("In loading synapses, target and source are not conform.");
    }

    file_synapses.close();
}

void NetworkGraph::print(std::ostream& os, const std::unique_ptr<NeuronsExtraInfo>& informations) const {
    const int my_rank = MPIWrapper::get_my_rank();

    // For my neurons
    for (size_t target_neuron_id = 0; target_neuron_id < my_num_neurons; target_neuron_id++) {
        // Walk through in-edges of my neuron
        const NetworkGraph::Edges& in_edges = get_in_edges(target_neuron_id);

        RankNeuronId rank_neuron_id{ my_rank, target_neuron_id };

        const auto global_target = informations->rank_neuron_id2glob_id(rank_neuron_id);

        for (auto it_in_edge = in_edges.cbegin(); it_in_edge != in_edges.cend(); ++it_in_edge) {
            RankNeuronId tmp_rank_neuron_id{ it_in_edge->first.first, it_in_edge->first.second };

            const auto global_source = informations->rank_neuron_id2glob_id(tmp_rank_neuron_id);

            // <target neuron id>  <source neuron id>  <weight>
            os
                << (global_target + 1) << "\t"
                << (global_source + 1) << "\t"
                << it_in_edge->second << "\n";
        }
    }
}

void NetworkGraph::write_synapses_to_file(const std::string& filename, [[maybe_unused]] const Partition& partition) const {
    std::ofstream ofstream(filename, std::ios::binary | std::ios::out);

    ofstream << "# <source neuron id> <target neuron id> <weight> \n";

    for (size_t source_neuron_id = 0; source_neuron_id < my_num_neurons; source_neuron_id++) {
        // Walk through in-edges of my neuron
        const NetworkGraph::Edges& out_edges = get_out_edges(source_neuron_id);

        for (const auto& out_edge : out_edges) {
            const EdgesKey& ek = out_edge.first;
            const EdgesVal& ev = out_edge.second;

            const size_t& target_neuron_id = ek.second;
            //const int& target_neuron_rank = ek.first;

            //const size_t global_source_neuron_id = partition.get_global_id(source_neuron_id);
            //const size_t global_target_neuron_id = partition.get_global_id(target_neuron_id);

            ofstream << source_neuron_id << "\t" << target_neuron_id << "\t" << ev << "\n";
        }
    }
}
