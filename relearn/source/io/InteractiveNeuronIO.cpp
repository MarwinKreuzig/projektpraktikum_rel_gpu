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

#include "../util/RelearnException.h"
#include "spdlog/spdlog.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

std::vector<std::pair<size_t, std::vector<size_t>>> InteractiveNeuronIO::load_enable_interrups(const std::filesystem::path& path_to_file) {
    std::ifstream file{ path_to_file };

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "InteractiveNeuronIO::load_enable_interrups: Opening the file was not successful");

    std::vector<std::pair<size_t, std::vector<size_t>>> return_value;

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        std::stringstream sstream(line);

        size_t step{};
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

        std::vector<size_t> indices;

        for (size_t id{}; sstream >> id;) {
            indices.emplace_back(id);
        }

        return_value.emplace_back(step, std::move(indices));
    }

    return return_value;
}

std::vector<std::pair<size_t, std::vector<size_t>>> InteractiveNeuronIO::load_disable_interrups(const std::filesystem::path& path_to_file) {
    std::ifstream file{ path_to_file };

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "InteractiveNeuronIO::load_disable_interrups: Opening the file was not successful");

    std::vector<std::pair<size_t, std::vector<size_t>>> return_value;

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        std::stringstream sstream(line);

        size_t step{};
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

        std::vector<size_t> indices;

        for (size_t id{}; sstream >> id;) {
            indices.emplace_back(id);
        }

        return_value.emplace_back(step, std::move(indices));
    }

    return return_value;
}

std::vector<std::pair<size_t, size_t>> InteractiveNeuronIO::load_creation_interrups(const std::filesystem::path& path_to_file) {
    std::ifstream file{ path_to_file };

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "InteractiveNeuronIO::load_creation_interrups: Opening the file was not successful");

    std::vector<std::pair<size_t, size_t>> return_value;

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        std::stringstream sstream(line);

        size_t step{};
        char delim{};
        size_t count{};

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
