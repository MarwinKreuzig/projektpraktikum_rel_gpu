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

#include "spdlog/spdlog.h"

#include <algorithm>
#include <istream>
#include <sstream>
#include <string>

std::map<size_t, RankNeuronId> NeuronIdTranslator::translate_global_ids(const std::vector<size_t>& global_ids, const std::filesystem::path& path_to_neurons) {
    const auto& id_to_position = load_neuron_positions(global_ids, path_to_neurons);



    return std::map<size_t, RankNeuronId>();
}

std::map<size_t, NeuronIdTranslator::position_type> NeuronIdTranslator::load_neuron_positions(const std::vector<size_t>& global_ids, const std::filesystem::path& path_neurons) {
    std::map<size_t, NeuronIdTranslator::position_type> translation_map{};

    std::string line{};
    std::ifstream file_neurons(path_neurons, std::ios::binary | std::ios::in);

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
