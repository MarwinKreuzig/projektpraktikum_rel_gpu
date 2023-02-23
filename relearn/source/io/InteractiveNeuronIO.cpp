/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "InteractiveNeuronIO.h"

#include "io/parser/StimulusParser.h"
#include "util/RelearnException.h"

#include "spdlog/spdlog.h"

#include <functional>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

std::vector<std::pair<InteractiveNeuronIO::step_type, std::vector<NeuronID>>> InteractiveNeuronIO::load_enable_interrupts(const std::filesystem::path& path_to_file, const MPIRank my_rank) {
    std::ifstream file{ path_to_file };

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "InteractiveNeuronIO::load_enable_interrupts: Opening the file was not successful");

    std::vector<std::pair<step_type, std::vector<NeuronID>>> return_value{};

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        std::stringstream sstream(line);

        step_type step{};
        char delim{};

        bool success = (sstream >> step) && (sstream >> delim);

        if (!success) {
            std::cerr << "Skipping line: \"" << line << "\"\n";
            continue;
        }

        if (delim != 'e') {
            if (delim != 'd' && delim != 'c') {
                std::cerr << "Wrong deliminator: \"" << line << "\"\n";
            }
            continue;
        }

        std::vector<NeuronID> indices{};

        for (std::string rank_neuron_string; sstream >> rank_neuron_string;) {
            const auto rank_neuron_vector = StringUtil::split_string(rank_neuron_string, ':');
            const auto rank = MPIRank{ std::stoi(rank_neuron_vector[0]) };
            const auto neuron_id = NeuronID(std::stoi(rank_neuron_vector[1]));

            if (rank == my_rank) {
                indices.emplace_back(neuron_id);
            }
        }

        return_value.emplace_back(step, std::move(indices));
    }

    return return_value;
}

std::vector<std::pair<InteractiveNeuronIO::step_type, std::vector<NeuronID>>> InteractiveNeuronIO::load_disable_interrupts(const std::filesystem::path& path_to_file, const MPIRank my_rank) {
    std::ifstream file{ path_to_file };

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "InteractiveNeuronIO::load_disable_interrupts: Opening the file was not successful");

    std::vector<std::pair<step_type, std::vector<NeuronID>>> return_value{};

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        std::stringstream sstream(line);

        step_type step{};
        char delim{};

        bool success = (sstream >> step) && (sstream >> delim);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        if (delim != 'd') {
            if (delim != 'e' && delim != 'c') {
                spdlog::info("Wrong deliminator: {}", line);
            }
            continue;
        }

        std::vector<NeuronID> indices{};

        for (std::string rank_neuron_string; sstream >> rank_neuron_string;) {
            const auto rank_neuron_vector = StringUtil::split_string(rank_neuron_string, ':');
            const auto rank = MPIRank{ std::stoi(rank_neuron_vector[0]) };
            const auto neuron_id = NeuronID(std::stoi(rank_neuron_vector[1]) - 1);

            if (rank == my_rank) {
                indices.push_back(neuron_id);
            }
        }

        return_value.emplace_back(step, std::move(indices));
    }

    return return_value;
}

std::vector<std::pair<InteractiveNeuronIO::step_type, InteractiveNeuronIO::number_neurons_type>>
InteractiveNeuronIO::load_creation_interrupts(const std::filesystem::path& path_to_file) {
    std::ifstream file{ path_to_file };

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good,
        "InteractiveNeuronIO::load_creation_interrupts: Opening the file was not successful");

    std::vector<std::pair<step_type, number_neurons_type>> return_value{};

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        std::stringstream sstream(line);

        step_type step{};
        char delim{};
        number_neurons_type count{};

        bool success = (sstream >> step) && (sstream >> delim) && (sstream >> count);

        if (!success) {
            std::cerr << "Skipping line: \"" << line << "\"\n";
            continue;
        }

        if (delim != 'c') {
            if (delim != 'e' && delim != 'd') {
                std::cerr << "Wrong deliminator: \"" << line << "\"\n";
            }
            continue;
        }

        return_value.emplace_back(step, count);
    }

    return return_value;
}

RelearnTypes::stimuli_function_type InteractiveNeuronIO::load_stimulus_interrupts(
    const std::filesystem::path& path_to_file, const MPIRank my_rank,
    std::shared_ptr<LocalAreaTranslator> local_area_translator) {
    RelearnException::check(my_rank.is_initialized(),
        "InteractiveNeuronIO::load_stimulus_interrupts: my_rank was virtual");

    std::ifstream file{ path_to_file };

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good,
        "InteractiveNeuronIO::load_stimulus_interrupts: Opening the file was not successful");

    std::vector<StimulusParser::Stimulus> stimuli{};

    const auto num_neurons = local_area_translator->get_number_neurons_in_total();

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        auto stimulus = StimulusParser::parse_line(line, my_rank.get_rank());

        if (stimulus.has_value()) {
            auto stimulus_value = stimulus.value();
            for (const auto& neuron_id : stimulus_value.matching_ids) {
                RelearnException::check(neuron_id.get_neuron_id() < num_neurons,
                    "InteractiveNeuronIO::load_stimulus_interrupts: Invalid neuron id {}",
                    neuron_id);
            }
            std::unordered_set<NeuronID> ids{};
            ids.insert(stimulus_value.matching_ids.begin(), stimulus_value.matching_ids.end());
            for (const auto& area : stimulus.value().matching_area_names) {
                if (local_area_translator->knows_area_name(area)) {
                    const auto& area_id = local_area_translator->get_area_id_for_area_name(area);
                    auto ids_in_area = local_area_translator->get_neuron_ids_in_area(area_id);
                    ids.insert(ids_in_area.begin(), ids_in_area.end());
                }
            }
            if (!ids.empty()) {
                stimuli.emplace_back(
                    StimulusParser::Stimulus{ stimulus_value.interval, stimulus_value.stimulus_intensity, ids, {} });
            }
        }
    }

    return StimulusParser::generate_stimulus_function(std::move(stimuli));
}
