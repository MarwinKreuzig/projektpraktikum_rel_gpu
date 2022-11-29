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

#include "util/RelearnException.h"
#include "util/StimulusParser.h"

#include "spdlog/spdlog.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

std::vector<std::pair<InteractiveNeuronIO::step_type, std::vector<NeuronID>>> InteractiveNeuronIO::load_enable_interrupts(const std::filesystem::path& path_to_file) {
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

        for (NeuronID::value_type id{}; sstream >> id;) {
            indices.emplace_back(id);
        }

        return_value.emplace_back(step, std::move(indices));
    }

    return return_value;
}

std::vector<std::pair<InteractiveNeuronIO::step_type, std::vector<NeuronID>>> InteractiveNeuronIO::load_disable_interrupts(const std::filesystem::path& path_to_file) {
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

        for (NeuronID::value_type id{}; sstream >> id;) {
            indices.emplace_back(id);
        }

        return_value.emplace_back(step, std::move(indices));
    }

    return return_value;
}

std::vector<std::pair<InteractiveNeuronIO::step_type, InteractiveNeuronIO::number_neurons_type>> InteractiveNeuronIO::load_creation_interrupts(const std::filesystem::path& path_to_file) {
    std::ifstream file{ path_to_file };

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "InteractiveNeuronIO::load_creation_interrupts: Opening the file was not successful");

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

std::function<double(InteractiveNeuronIO::step_type, NeuronID::value_type)> InteractiveNeuronIO::load_stimulus_interrupts(
    const std::filesystem::path& path_to_file, const int my_rank, std::shared_ptr<LocalAreaTranslator> local_area_translator) {
    std::ifstream file{ path_to_file };

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "InteractiveNeuronIO::load_stimulus_interrupts: Opening the file was not successful");

    std::vector<Stimulus> stimuli{};

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        auto stimulus = StimulusParser::parse_line(line, my_rank);
        if (stimulus.has_value()) {
            stimuli.emplace_back(std::move(stimulus.value()));
        }
    }

    return StimulusParser::generate_stimulus_function(std::move(stimuli), std::move(local_area_translator));
}
