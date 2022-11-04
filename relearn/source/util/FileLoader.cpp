/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FileLoader.h"

#include "../util/RelearnException.h"

#include "../Config.h"

#include <fstream>
#include <iostream>
#include <sstream>


std::vector<size_t> FileLoader::load_neuron_id_list(const std::string &path_to_file) {
    std::ifstream file{path_to_file};

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good,
                            "FileLoader::load_neuron_id_list: Opening the file was not successful");

    std::vector<size_t> return_value;

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        std::stringstream sstream(line);
        size_t neuron_id;
        sstream >> neuron_id;
        return_value.push_back(neuron_id-1);

    }

    return return_value;
}

std::function<std::vector<std::pair<size_t,double>>(size_t)>
FileLoader::load_external_stimulus(const std::string &path_to_file) {
    std::ifstream file{path_to_file};

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good,
                            "FileLoader::load_external_stimulus: Opening the file was not successful");

    std::vector<std::pair<std::tuple<size_t,size_t,double>,std::vector<size_t>>> external_stimuli;

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        auto values_in_line = split_string(line,' ');
        RelearnException::check(values_in_line.size() >= 4, "FileLoader::load_external_stimulus: Invalid line");

        int step_begin = std::stoi(values_in_line[0]);
        int step_end = std::stoi(values_in_line[1]);
        double x = std::stod(values_in_line[2]);
        std::vector<size_t> neuron_ids;
        for(int i=3;i<values_in_line.size();i++) {
            size_t neuron_id = std::stoi(values_in_line[i])-1;
            neuron_ids.push_back(neuron_id);
        }
        external_stimuli.push_back(std::make_pair(std::make_tuple(step_begin,step_end,x),neuron_ids));
    }

    auto get_external_stimulus = [es = std::move(external_stimuli)] (int step) -> std::vector<std::pair<size_t,double>> {
        std::vector<std::pair<size_t,double>> neuron_stimuli;
        for(const auto& single_stimulus : es) {
            auto time_tuple = single_stimulus.first;
            size_t begin = std::get<0>(time_tuple);
            size_t end = std::get<1>(time_tuple);
            if(begin<=step && step <= end) {
                double x = std::get<2>(time_tuple);
                for(auto neuron_id:single_stimulus.second) {
                    neuron_stimuli.push_back(std::make_pair(neuron_id, x));
                }
            }
        }
        return std::move(neuron_stimuli);
    };

    return get_external_stimulus;
}

std::vector<std::string> FileLoader::split_string(const std::string &string, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(string);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}
