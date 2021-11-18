/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronIdTranslator.h"

#include "../mpi/MPIWrapper.h"
#include "../structure/Partition.h"

#include "spdlog/spdlog.h"

#include <algorithm>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

std::map<size_t, RankNeuronId> NeuronIdTranslator::translate_global_ids(const std::vector<size_t>& global_ids) {
    const int mpi_rank = MPIWrapper::get_my_rank();
    const int num_ranks = MPIWrapper::get_num_ranks();

    std::vector<size_t> num_foreign_ids_from_ranks_send(num_ranks, 0);
    std::vector<size_t> num_foreign_ids_from_ranks(num_ranks, 0);

    std::vector<std::vector<size_t>> global_ids_to_send(num_ranks);
    std::vector<std::vector<size_t>> global_ids_to_receive(num_ranks);
    std::vector<std::vector<size_t>> global_ids_local_value(num_ranks);

    std::map<size_t, int> neuron_id_to_rank{};
    const auto& id_to_position = load_neuron_positions(global_ids);

    for (const auto& [neuron_id, neuron_position] : id_to_position) {
        const auto rank = partition->get_mpi_rank_from_pos(neuron_position);
        neuron_id_to_rank[neuron_id] = rank;
        global_ids_to_send[rank].emplace_back(neuron_id);
    }

    for (auto rank = 0; rank < num_ranks; rank++) {
        num_foreign_ids_from_ranks_send[rank] = global_ids_to_send[rank].size();
    }

    MPIWrapper::all_to_all(num_foreign_ids_from_ranks_send, num_foreign_ids_from_ranks);

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (mpi_rank == rank) {
            RelearnException::check(global_ids_to_receive[rank].empty(), "NetworkGraph::translate_global_to_local: Should receive ids from myself");
            continue;
        }

        global_ids_to_receive[rank].resize(num_foreign_ids_from_ranks[rank]);
    }

    std::vector<MPIWrapper::AsyncToken> mpi_requests(static_cast<size_t>(num_ranks) * 2 - 2);
    int request_counter = 0;

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (mpi_rank == rank) {
            continue;
        }

        size_t* buffer = global_ids_to_receive[rank].data();
        const auto size_in_bytes = static_cast<int>(global_ids_to_receive[rank].size() * sizeof(size_t));

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (mpi_rank == rank) {
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
            global_id = partition->get_local_id(global_id);
        }
    }

    mpi_requests = std::vector<MPIWrapper::AsyncToken>(static_cast<size_t>(num_ranks) * 2 - 2);
    request_counter = 0;

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (mpi_rank == rank) {
            continue;
        }

        size_t* buffer = global_ids_local_value[rank].data();
        const auto size_in_bytes = static_cast<int>(global_ids_local_value[rank].size() * sizeof(size_t));

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (mpi_rank == rank) {
            continue;
        }

        const size_t* buffer = global_ids_to_receive[rank].data();
        const auto size_in_bytes = static_cast<int>(global_ids_to_receive[rank].size() * sizeof(size_t));

        MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);

    std::map<size_t, RankNeuronId> final_translation_map{};

    for (auto rank = 0; rank < num_ranks; rank++) {
        std::vector<size_t>& translated_ids = global_ids_local_value[rank];
        std::vector<size_t>& local_global_ids = global_ids_to_send[rank];

        RelearnException::check(translated_ids.size() == local_global_ids.size(), "NetworkGraph::translate_global_to_local: The vectors have not the same size in load network");

        for (auto i = 0; i < translated_ids.size(); i++) {
            const size_t local_id = translated_ids[i];
            const size_t global_id = local_global_ids[i];

            const auto rank = neuron_id_to_rank[global_id];

            final_translation_map[global_id] = { rank, local_id };
        }
    }

    return final_translation_map;
}

std::map<size_t, NeuronIdTranslator::position_type> NeuronIdTranslator::load_neuron_positions(const std::vector<size_t>& global_ids) {
    std::map<size_t, position_type> translation_map{};

    std::string line{};
    std::ifstream file_neurons(path_to_neurons, std::ios::binary | std::ios::in);

    while (std::getline(file_neurons, line)) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        size_t id{};
        position_type::value_type pos_x = 0.0;
        position_type::value_type pos_y = 0.0;
        position_type::value_type pos_z = 0.0;
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

        if (std::binary_search(global_ids.cbegin(), global_ids.cend(), id)) {
            translation_map[id] = { pos_x, pos_y, pos_z };
            if (translation_map.size() == global_ids.size()) {
                break;
            }
        }
    }

    return translation_map;
}
