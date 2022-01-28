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

NeuronIdTranslator::NeuronIdTranslator(std::shared_ptr<Partition> partition)
    : partition(std::move(partition)) {

    global_neuron_ids.resize(this->partition->get_number_local_subdomains());
}

bool NeuronIdTranslator::is_neuron_local(NeuronID global_id) const {
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

    for (auto rank = 0; rank < num_ranks; rank++) {
        num_foreign_ids_from_ranks_send[rank] = global_ids_to_send[rank].size();
    }

    MPIWrapper::all_to_all(num_foreign_ids_from_ranks_send, num_foreign_ids_from_ranks);

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (mpi_rank == rank) {
            RelearnException::check(global_ids_to_receive[rank].empty(), "FileNeuronIdTranslator::translate_global_to_local: Should receive ids from myself");
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

        auto* buffer = global_ids_to_receive[rank].data();
        const auto size_in_bytes = static_cast<int>(global_ids_to_receive[rank].size() * sizeof(size_t));

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (mpi_rank == rank) {
            continue;
        }

        const auto* buffer = global_ids_to_send[rank].data();
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
            global_id = get_local_id(global_id);
        }
    }

    mpi_requests = std::vector<MPIWrapper::AsyncToken>(static_cast<size_t>(num_ranks) * 2 - 2);
    request_counter = 0;

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (mpi_rank == rank) {
            continue;
        }

        auto* buffer = global_ids_local_value[rank].data();
        const auto size_in_bytes = static_cast<int>(global_ids_local_value[rank].size() * sizeof(size_t));

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    for (auto rank = 0; rank < num_ranks; rank++) {
        if (mpi_rank == rank) {
            continue;
        }

        const auto* buffer = global_ids_to_receive[rank].data();
        const auto size_in_bytes = static_cast<int>(global_ids_to_receive[rank].size() * sizeof(size_t));

        MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[request_counter]);
        request_counter++;
    }

    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);

    std::map<NeuronID, RankNeuronId> final_translation_map{};

    for (auto rank = 0; rank < num_ranks; rank++) {
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

RandomNeuronIdTranslator::RandomNeuronIdTranslator(std::shared_ptr<Partition> partition)
    : NeuronIdTranslator(std::move(partition))
    , number_local_neurons(this->partition->get_number_local_neurons()) {

    const auto num_ranks = MPIWrapper::get_num_ranks();

    // Gather the number of neurons of every process
    std::vector<size_t> rank_to_num_neurons(num_ranks);
    MPIWrapper::all_gather(number_local_neurons, rank_to_num_neurons);

    mpi_rank_to_local_start_id.resize(num_ranks);

    // Store global start neuron id of every rank
    mpi_rank_to_local_start_id[0] = 0;
    for (size_t i = 1; i < num_ranks; i++) {
        mpi_rank_to_local_start_id[i] = mpi_rank_to_local_start_id[i - 1] + rank_to_num_neurons[i - 1];
    }
}

std::map<NeuronID, RankNeuronId> RandomNeuronIdTranslator::translate_global_ids(const std::vector<NeuronID>& global_ids) {
    std::map<NeuronID, RankNeuronId> return_value{};

    const auto num_ranks = MPIWrapper::get_num_ranks();

    if (num_ranks == 1) {
        for (const auto& global_id : global_ids) {
            RelearnException::check(global_id.id() < number_local_neurons,
                "RandomNeuronIdTranslator::translate_global_ids: Global id ({}) is too large for the number of local neurons ({})", global_id, number_local_neurons);

            return_value.emplace(global_id, RankNeuronId{ 0, global_id });
        }

        return return_value;
    }

    for (const auto& global_id : global_ids) {
        auto last_rank = 0;
        auto current_rank = 1;
        auto current_start = mpi_rank_to_local_start_id[current_rank];

        const auto id = global_id.id();
        while (current_start <= id) {
            last_rank++;
            current_rank++;

            if (current_rank == num_ranks) {
                break;
            }

            current_start = mpi_rank_to_local_start_id[current_rank];
        }

        RelearnException::check(global_id.id() >= current_start, "RandomNeuronIdTranslator::translate_global_ids: Error in while loop");

        if (current_rank < num_ranks) {
            RelearnException::check(global_id.id() < current_rank, "RandomNeuronIdTranslator::translate_global_ids: While loop breaked too early");
        }

        const auto local_id = global_id - current_start;

        return_value.emplace(global_id, RankNeuronId{ last_rank, local_id });
    }

    return return_value;
}

NeuronID RandomNeuronIdTranslator::translate_rank_neuron_id(const RankNeuronId& rni) {
    const auto& [rank, local_neuron_id] = rni;
    RelearnException::check(rank >= 0, "RandomNeuronIdTranslator::translate_rank_neuron_ids: There was a negative MPI rank");
    RelearnException::check(rank < mpi_rank_to_local_start_id.size(), "RandomNeuronIdTranslator::translate_rank_neuron_ids: The requested MPI rank is not stored");
    RelearnException::check(local_neuron_id.is_initialized(), "RandomNeuronIdTranslator::translate_rank_neuron_ids: The requested neuron id is unitialized");

    const auto glob_id = local_neuron_id + mpi_rank_to_local_start_id[rank];

    if (local_neuron_id.id() < mpi_rank_to_local_start_id.size() - 1) {
        RelearnException::check(glob_id.id() < mpi_rank_to_local_start_id[rank + 1ULL], "RandomNeuronIdTranslator::translate_rank_neuron_ids: The translated id exceeded the starting id of the next rank");
    }

    return glob_id;
}

std::map<RankNeuronId, NeuronID> RandomNeuronIdTranslator::translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) {
    std::map<RankNeuronId, NeuronID> return_value{};

    for (const auto& id : ids) {
        const auto& [rank, local_neuron_id] = id;
        RelearnException::check(rank >= 0, "RandomNeuronIdTranslator::translate_rank_neuron_ids: There was a negative MPI rank");
        RelearnException::check(rank < mpi_rank_to_local_start_id.size(), "RandomNeuronIdTranslator::translate_rank_neuron_ids: The requested MPI rank is not stored");
        RelearnException::check(local_neuron_id.is_initialized(), "RandomNeuronIdTranslator::translate_rank_neuron_ids: The requested neuron id is unitialized");

        const auto glob_id = local_neuron_id + mpi_rank_to_local_start_id[rank];

        if (local_neuron_id.id() < mpi_rank_to_local_start_id.size() - 1) {
            RelearnException::check(glob_id.id() < mpi_rank_to_local_start_id[rank + 1ULL], "RandomNeuronIdTranslator::translate_rank_neuron_ids: The translated id exceeded the starting id of the next rank");
        }

        return_value.emplace(id, glob_id);
    }

    return return_value;
}
