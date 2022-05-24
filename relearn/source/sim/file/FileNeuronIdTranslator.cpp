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

#include "mpi/MPIWrapper.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"

#include "spdlog/spdlog.h"

#include <algorithm>
#include <istream>
#include <sstream>
#include <string>

bool FileNeuronIdTranslator::is_neuron_local(NeuronID global_id) const {
    const auto total_found = std::ranges::any_of(global_neuron_ids.begin(), global_neuron_ids.end(),
        [global_id](const auto& global_ids) { return std::binary_search(global_ids.begin(), global_ids.end(), global_id); });

    return total_found;
}

NeuronID FileNeuronIdTranslator::get_local_id(NeuronID global_id) const {
    typename NeuronID::value_type id{ 0 };

    for (const auto& ids : global_neuron_ids) {
        const auto pos = std::lower_bound(ids.begin(), ids.end(), global_id);

        if (pos != ids.end()) {
            id += pos - ids.begin();
            return NeuronID{ false, false, id };
        }

        id += ids.size();
    }

    RelearnException::fail("Partition::is_neuron_local: Didn't find global id {}", global_id);
    return NeuronID::uninitialized_id();
}

NeuronID FileNeuronIdTranslator::get_global_id(NeuronID local_id) const {
    const auto local_neuron_id = local_id.get_local_id();

    size_t counter = 0;
    for (auto i = 0; i < partition->get_number_local_subdomains(); i++) {
        const size_t old_counter = counter;

        counter += global_neuron_ids[i].size();
        if (local_neuron_id < counter) {
            return global_neuron_ids[i][local_neuron_id - old_counter];
        }
    }

    return local_id;
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
        NeuronID id{ true, false, read_id };

        if (std::binary_search(global_ids.cbegin(), global_ids.cend(), id)) {
            translation_map[id] = { pos_x, pos_y, pos_z };
            if (translation_map.size() == global_ids.size()) {
                break;
            }
        }
    }

    return translation_map;
}

void FileNeuronIdTranslator::create_neurons(size_t number_local_creations) {
    const auto num_ranks = MPIWrapper::get_num_ranks();
    RelearnException::check(num_ranks == 1, "FileNeuronIdTranslator::create_neurons: Can only create neurons for files with one mpi rank, but there were {}", num_ranks);

    const auto current_number_ids = global_neuron_ids.size();
    const auto new_ids = NeuronID::range_global(current_number_ids, current_number_ids + number_local_creations);

    for (const auto& neuron_id : new_ids) {
        global_neuron_ids[0].emplace_back(neuron_id);
    }
}
