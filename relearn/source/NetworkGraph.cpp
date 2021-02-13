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

#include "MPIWrapper.h"
#include "NeuronIdMap.h"
#include "Partition.h"
#include "RelearnException.h"

#include <cmath>
#include <fstream>
#include <ostream>
#include <sstream>

NetworkGraph::NetworkGraph(size_t my_num_neurons)
    : neuron_in_neighborhood(my_num_neurons)
    , neuron_out_neighborhood(my_num_neurons)
    , my_num_neurons(my_num_neurons) {
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

size_t NetworkGraph::get_num_in_edges(size_t neuron_id) const {
    RelearnException::check(neuron_id < neuron_in_neighborhood.size(),
        "In get_num_in_edges, tried with a too large id: %u %u", neuron_id, my_num_neurons);

    return neuron_in_neighborhood[neuron_id].size();
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

void NetworkGraph::add_edge(Edges& edges, int rank, size_t neuron_id, int weight) {
    EdgesKey rank_neuron_id_pair{ rank, neuron_id };

    size_t idx = 0;

    for (auto& [key, val] : edges) {
        if (key == rank_neuron_id_pair) {
            const int sum = val + weight;
            val = sum;

            if (sum == 0) {
                break;
            }
        }

        idx++;
    }

    if (idx == edges.size()) {
        edges.emplace_back(rank_neuron_id_pair, weight);
    } else {
        const auto idx_last = edges.size() - 1;
        std::swap(edges[idx], edges[idx_last]);
        edges.erase(edges.cend() - 1);
    }
    //
    //const auto edges_it = edges.find(rank_neuron_id_pair);

    //if (edges_it == edges.end()) {
    //    edges[rank_neuron_id_pair] = weight;
    //    return;
    //}

    //// Current edge weight + additional weight
    //const int sum = edges_it->second + weight;

    //// Edge weight becomes 0, so delete edge
    //if (0 == sum) {
    //    edges.erase(edges_it);
    //    // Update edge weight
    //} else {
    //    edges_it->second = sum;
    //}
}

void NetworkGraph::add_edge_weight(size_t target_neuron_id, int target_rank, size_t source_neuron_id, int source_rank, int weight) {
    if (weight == 0) {
        RelearnException::fail("weight of edge to add is zero");
        return;
    }

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
}

void NetworkGraph::add_edge_weights(const std::string& filename) {
    std::ifstream file(filename);
    std::string line{};

    while (std::getline(file, line)) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        std::stringstream sstream(line);
        double src_x = 0.0;
        double src_y = 0.0;
        double src_z = 0.0;
        double tgt_x = 0.0;
        double tgt_y = 0.0;
        double tgt_z = 0.0;
        const bool success = (sstream >> src_x) && (sstream >> src_y) && (sstream >> src_z) && (sstream >> tgt_x) && (sstream >> tgt_y) && (sstream >> tgt_z);

        RelearnException::check(success, "success was denied");

        RankNeuronId src_id{ -1, Constants::uninitialized };
        RankNeuronId tgt_id{ -1, Constants::uninitialized };
        bool ret = false;
        std::tie(ret, src_id) = NeuronIdMap::pos2rank_neuron_id({ src_x, src_y, src_z });
        RelearnException::check(ret, "ret was false");
        std::tie(ret, tgt_id) = NeuronIdMap::pos2rank_neuron_id({ tgt_x, tgt_y, tgt_z });
        RelearnException::check(ret, "ret was false");

        add_edge_weight(tgt_id.get_neuron_id(), tgt_id.get_rank(),
            src_id.get_neuron_id(), src_id.get_rank(), 1);

        if (!success) {
            std::cerr << "Skipping line: \"" << line << "\"\n";
            continue;
        }
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
        const auto rank = static_cast<int>(partition.get_subdomain_id_from_pos(pos));
        id_to_rank[id] = rank;
    }

    std::map<size_t, size_t> global_id_to_local_id{};

    if (!id_to_pos.empty()) {
        translate_global_to_local(foreing_ids, id_to_rank, partition, global_id_to_local_id);
    }

    for (const auto& [source_neuron_id, target_neuron_id, weight] : local_synapses) {
        const size_t translated_source_neuron_id = partition.get_local_id(source_neuron_id);
        const size_t translated_target_neuron_id = partition.get_local_id(target_neuron_id);

        const int source_rank = my_rank;
        const int target_rank = my_rank;

        add_edge_weight(translated_target_neuron_id, target_rank, translated_source_neuron_id, source_rank, weight);
    }

    for (const auto& [source_neuron_id, target_neuron_id, weight] : out_synapses) {
        const size_t translated_source_neuron_id = partition.get_local_id(source_neuron_id);

        const int source_rank = my_rank;
        const int target_rank = id_to_rank[target_neuron_id];
        const size_t translated_target_neuron_id = global_id_to_local_id[target_neuron_id];

        add_edge_weight(translated_target_neuron_id, target_rank, translated_source_neuron_id, source_rank, weight);
    }

    for (const auto& [source_neuron_id, target_neuron_id, weight] : in_synapses) {
        const size_t translated_target_neuron_id = partition.get_local_id(target_neuron_id);

        const int source_rank = id_to_rank[source_neuron_id];
        const int target_rank = my_rank;
        const size_t translated_source_neuron_id = global_id_to_local_id[source_neuron_id];

        add_edge_weight(translated_target_neuron_id, target_rank, translated_source_neuron_id, source_rank, weight);
    }

    std::stringstream sstream{};

    sstream << "I'm rank: " << my_rank << " of " << MPIWrapper::get_num_ranks() << ".\n";
    sstream << "I've loaded: [local, out, int] " << local_synapses.size() << " + " << out_synapses.size() << " + " << in_synapses.size() << " = " << (local_synapses.size() + out_synapses.size() + in_synapses.size()) << " many synapses."
            << "\n";

    LogFiles::write_to_file(LogFiles::EventType::Cout, sstream.str(), true);
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

void NetworkGraph::translate_global_to_local(const std::set<size_t>& global_ids, const std::map<size_t, int>& id_to_rank, const Partition& partition, std::map<size_t, size_t>& global_id_to_local_id) {

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

    MPIWrapper::all_to_all(num_foreign_ids_from_ranks_send, num_foreign_ids_from_ranks, MPIWrapper::Scope::global);

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

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[request_counter]);
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

        MPIWrapper::async_send(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[request_counter]);
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

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[request_counter]);
        request_counter++;
    }

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (MPIWrapper::get_my_rank() == rank) {
            continue;
        }

        const size_t* buffer = global_ids_to_receive[rank].data();
        const auto size_in_bytes = static_cast<int>(global_ids_to_receive[rank].size() * sizeof(size_t));

        MPIWrapper::async_send(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[request_counter]);
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
            std::cerr << "Skipping line: \"" << line << "\"\n";
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

void NetworkGraph::print(std::ostream& os) const {
    const int my_rank = MPIWrapper::get_my_rank();

    // For my neurons
    for (size_t target_neuron_id = 0; target_neuron_id < my_num_neurons; target_neuron_id++) {
        // Walk through in-edges of my neuron
        const NetworkGraph::Edges& in_edges = get_in_edges(target_neuron_id);
        NetworkGraph::Edges::const_iterator it_in_edge;

        RankNeuronId rank_neuron_id{ my_rank, target_neuron_id };
        size_t glob_tgt = 0;

        bool ret = true;
        std::tie(ret, glob_tgt) = NeuronIdMap::rank_neuron_id2glob_id(rank_neuron_id);
        RelearnException::check(ret, "ret is false");

        for (it_in_edge = in_edges.begin(); it_in_edge != in_edges.end(); ++it_in_edge) {

            RankNeuronId tmp_rank_neuron_id{ it_in_edge->first.first, it_in_edge->first.second };
            size_t glob_src = 0;
            std::tie(ret, glob_src) = NeuronIdMap::rank_neuron_id2glob_id(tmp_rank_neuron_id);
            RelearnException::check(ret, "ret is false");

            glob_src++;
            glob_tgt++;

            // <target neuron id>  <source neuron id>  <weight>
            os
                << glob_tgt << "\t"
                << glob_src << "\t"
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
