/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FileNeuronIdTranslator.h"

#include "../../mpi/MPIWrapper.h"
#include "../../structure/Partition.h"
#include "../../util/RelearnException.h"

#include "spdlog/spdlog.h"

#include <algorithm>
#include <istream>
#include <sstream>
#include <string>

bool FileNeuronIdTranslator::is_neuron_local(NeuronID global_id) const {
    for (const auto& global_ids : global_neuron_ids) {
        const bool found = std::binary_search(global_ids.begin(), global_ids.end(), global_id);
        if (found) {
            return true;
        }
    }

    return false;
}

NeuronID NeuronIdTranslator::get_local_id(NeuronID global_id) const {
    typename NeuronID::value_type id{ 0 };

    for (const auto& ids : global_neuron_ids) {
        const auto pos = std::lower_bound(ids.begin(), ids.end(), global_id);

        if (pos != ids.end()) {
            id += pos - ids.begin();
            return NeuronID{ id };
        }

        id += ids.size();
    }

    RelearnException::fail("Partition::is_neuron_local: Didn't find global id {}", global_id);
    return NeuronID{};
}

NeuronID NeuronIdTranslator::get_global_id(NeuronID local_id) const {
    size_t counter = 0;
    for (auto i = 0; i < partition->get_number_local_subdomains(); i++) {
        const size_t old_counter = counter;

        counter += global_neuron_ids[i].size();
        if (local_id.id() < counter) {
            const NeuronID local_local_id = local_id - old_counter;
            return global_neuron_ids[i][local_local_id.id()];
        }
    }

    return local_id;
}

std::map<NeuronID, RankNeuronId> FileNeuronIdTranslator::translate_global_ids(const std::vector<NeuronID>& global_ids) {
    if (global_ids.empty()) {
        return {};
    }

    const int mpi_rank = MPIWrapper::get_my_rank();
    const int num_ranks = MPIWrapper::get_num_ranks();

    std::vector<size_t> num_foreign_ids_from_ranks_send(num_ranks, 0);
    std::vector<size_t> num_foreign_ids_from_ranks(num_ranks, 0);

    std::vector<std::vector<NeuronID>> global_ids_to_send(num_ranks);
    std::vector<std::vector<NeuronID>> global_ids_to_receive(num_ranks);
    std::vector<std::vector<NeuronID>> global_ids_local_value(num_ranks);

    std::map<NeuronID, int> neuron_id_to_rank{};
    const auto& id_to_position = load_neuron_positions(global_ids);

    for (const auto& [neuron_id, neuron_position] : id_to_position) {
        const auto rank = partition->get_mpi_rank_from_position(neuron_position);
        neuron_id_to_rank[neuron_id] = rank;
        global_ids_to_send[rank].emplace_back(neuron_id);
    }

    for (auto rank : MPIWrapper::get_ranks()) {
        num_foreign_ids_from_ranks_send[rank] = global_ids_to_send[rank].size();
    }

    std::vector<neuron_id> num_foreign_ids_from_ranks = MPIWrapper::all_to_all(num_foreign_ids_from_ranks_send);

    for (auto rank : MPIWrapper::get_ranks()) {
        if (mpi_rank == rank) {
            RelearnException::check(global_ids_to_receive[rank].empty(), "FileNeuronIdTranslator::translate_global_to_local: Should receive ids from myself");
            continue;
        }

        global_ids_to_receive[rank].resize(num_foreign_ids_from_ranks[rank]);
    }

    std::vector<MPIWrapper::AsyncToken> mpi_requests(static_cast<size_t>(num_ranks) * 2 - 2);
    int request_counter = 0;

    for (auto rank : MPIWrapper::get_ranks_without_my_rank()) {
        MPIWrapper::async_receive(std::span{ global_ids_to_receive[rank] }, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    for (auto rank : MPIWrapper::get_ranks_without_my_rank()) {
        // Reserve enough space for the answer - it will be as long as the request
        global_ids_local_value[rank].resize(global_ids_to_send[rank].size());

        MPIWrapper::async_send(std::span{ global_ids_to_send[rank] }, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);
    for (auto& vec : global_ids_to_receive) {
        for (auto& global_id : vec) {
            global_id = get_local_id(global_id);
        }
    }

    mpi_requests = std::vector<MPIWrapper::AsyncToken>(static_cast<size_t>(num_ranks) * 2 - 2);
    request_counter = 0;

    for (auto rank : MPIWrapper::get_ranks_without_my_rank()) {
        MPIWrapper::async_receive(std::span{ global_ids_local_value[rank] }, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    for (auto rank : MPIWrapper::get_ranks_without_my_rank()) {
        MPIWrapper::async_send(std::span{ global_ids_to_receive[rank] }, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);

    std::map<NeuronID, RankNeuronId> final_translation_map{};

    for (auto rank : MPIWrapper::get_ranks()) {
        std::vector<NeuronID>& translated_ids = global_ids_local_value[rank];
        std::vector<NeuronID>& local_global_ids = global_ids_to_send[rank];

        RelearnException::check(translated_ids.size() == local_global_ids.size(), "FileNeuronIdTranslator::translate_global_to_local: The vectors have not the same size in load network");

        for (auto i = 0; i < translated_ids.size(); i++) {
            const NeuronID local_id = translated_ids[i];
            const NeuronID global_id = local_global_ids[i];

            const auto rank = neuron_id_to_rank[global_id];

            final_translation_map[global_id] = { rank, local_id };
        }
    }

    return final_translation_map;
}

NeuronID FileNeuronIdTranslator::translate_rank_neuron_id(const RankNeuronId& /*rni*/) {
    RelearnException::fail("FileNeuronIdTranslator::translate_rank_neuron_id: Should not be here");
    return {};
}

std::map<NeuronID, FileNeuronIdTranslator::position_type> FileNeuronIdTranslator::load_neuron_positions(const std::vector<NeuronID>& global_ids) {
    std::map<NeuronID, position_type> translation_map{};

    std::string line{};
    std::ifstream file_neurons(path_to_neurons, std::ios::binary | std::ios::in);

    while (std::getline(file_neurons, line)) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        size_t read_id{};
        position_type::value_type pos_x = 0.0;
        position_type::value_type pos_y = 0.0;
        position_type::value_type pos_z = 0.0;
        std::string area_name{};
        std::string type{};

        std::stringstream sstream(line);
        const bool success = (sstream >> read_id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> type);

        if (!success) {
            spdlog::info("Skipping line: \"{}\"", line);
            continue;
        }

        // File starts at 1
        --read_id;
        NeuronID id{ read_id };

        if (std::binary_search(global_ids.cbegin(), global_ids.cend(), id)) {
            translation_map[id] = { pos_x, pos_y, pos_z };
            if (translation_map.size() == global_ids.size()) {
                break;
            }
        }
    }

    return translation_map;
}

std::map<RankNeuronId, NeuronID> FileNeuronIdTranslator::translate_rank_neuron_ids(const std::vector<RankNeuronId>& /*ids*/) {
    RelearnException::fail("FileNeuronIdTranslator::translate_rank_neuron_ids: Should not be here");
    return {};
}
