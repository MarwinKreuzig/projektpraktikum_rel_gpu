/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SubdomainFromFile.h"

#include "Config.h"
#include "LogFiles.h"
#include "NeuronToSubdomainAssignment.h"
#include "Partition.h"
#include "RelearnException.h"
#include "spdlog/spdlog.h"

#include <cmath>
#include <iostream>
#include <sstream>

SubdomainFromFile::SubdomainFromFile(const std::string& file_path)
    : file(file_path) {
    LogFiles::write_to_file(LogFiles::EventType::Cout, true, "Loading: " + file_path + "\n");

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "Opening the file was not successful");

    read_dimensions_from_file();
}

void SubdomainFromFile::read_dimensions_from_file() {
    Vec3d minimum(std::numeric_limits<double>::max());
    Vec3d maximum(std::numeric_limits<double>::min());

    size_t found_ex_neurons = 0;
    size_t found_in_neurons = 0;

    size_t total_number_neurons = 0;

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        size_t id{};
        double pos_x{};
        double pos_y{};
        double pos_z{};
        std::string area_name{};
        std::string signal_type{};

        std::stringstream sstream(line);
        bool success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> signal_type);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        total_number_neurons++;

        minimum.calculate_componentwise_minimum({ pos_x, pos_y, pos_z });
        maximum.calculate_componentwise_maximum({ pos_x, pos_y, pos_z });

        if (signal_type == "in") {
            found_in_neurons++;
        } else {
            found_ex_neurons++;
        }
    }

    {
        const auto new_max_x = std::nextafter(maximum.get_x(), maximum.get_x() + Constants::eps);
        const auto new_max_y = std::nextafter(maximum.get_y(), maximum.get_y() + Constants::eps);
        const auto new_max_z = std::nextafter(maximum.get_z(), maximum.get_z() + Constants::eps);

        maximum = { new_max_x, new_max_y, new_max_z };
    }

    total_num_neurons_in_file = total_number_neurons;

    const auto desired_num_neurons_ = found_ex_neurons + found_in_neurons;
    const auto desired_frac_neurons_exc_ = static_cast<double>(found_ex_neurons) / static_cast<double>(desired_num_neurons_);

    const auto simulation_box_length = maximum;

    set_desired_num_neurons(desired_num_neurons_);
    set_desired_frac_neurons_exc(desired_frac_neurons_exc_);
    set_simulation_box_length(simulation_box_length);
}

std::vector<NeuronToSubdomainAssignment::Node> SubdomainFromFile::read_nodes_from_file(const Position& min, const Position& max) {
    file.clear();
    file.seekg(0);

    double placed_ex_neurons = 0.0;
    double placed_in_neurons = 0.0;

    size_t current_num_neurons_ = 0;

    std::vector<NeuronToSubdomainAssignment::Node> nodes;

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        std::string signal_type{};

        Node node{};
        double pos_x{};
        double pos_y{};
        double pos_z{};
        std::stringstream sstream(line);
        bool success = (sstream >> node.id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> node.area_name) && (sstream >> signal_type);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        node.pos = { pos_x, pos_y, pos_z };

        // Ids start with 1
        node.id--;

        if (bool is_in_subdomain = position_in_box(node.pos, min, max); !is_in_subdomain) {
            continue;
        }

        if (signal_type == "ex") {
            node.signal_type = SignalType::EXCITATORY;
            ++placed_ex_neurons;
        } else {
            node.signal_type = SignalType::INHIBITORY;
            ++placed_in_neurons;
        }

        ++current_num_neurons_;
        nodes.emplace_back(node);
    }

    const auto current_frac_neurons_exc_ = placed_ex_neurons / static_cast<double>(current_num_neurons_);

    set_current_num_neurons(current_num_neurons_);
    set_current_frac_neurons_exc(current_frac_neurons_exc_);

    return nodes;
}

std::vector<size_t> SubdomainFromFile::neuron_global_ids(size_t subdomain_idx, [[maybe_unused]] size_t num_subdomains,
    [[maybe_unused]] size_t local_id_start, [[maybe_unused]] size_t local_id_end) const {
    const bool contains = is_loaded(subdomain_idx);
    if (!contains) {
        RelearnException::fail("Wanted to have neuron_global_ids of subdomain_idx that is not present");
        return {};
    }

    const Nodes& nodes = get_nodes(subdomain_idx);
    std::vector<size_t> global_ids;
    global_ids.reserve(nodes.size());

    for (const Node& node : nodes) {
        global_ids.push_back(node.id);
    }

    return global_ids;
}

void SubdomainFromFile::fill_subdomain(size_t subdomain_idx, [[maybe_unused]] size_t num_subdomains, const Position& min, const Position& max) {
    const bool subdomain_already_filled = is_loaded(subdomain_idx);
    if (subdomain_already_filled) {
        RelearnException::fail("Tried to fill an already filled subdomain.");
        return;
    }

    Nodes nodes{};

    auto nodes_vector = read_nodes_from_file(min, max);
    for (const auto& node : nodes_vector) {
        nodes.emplace(node);
    }

    set_nodes(subdomain_idx, std::move(nodes));
}

std::optional<std::vector<size_t>> SubdomainFromFile::read_neuron_ids_from_file(const std::string& file_path) {
    std::ifstream local_file(file_path);

    const bool file_is_good = local_file.good();
    const bool file_is_not_good = local_file.fail() || local_file.eof();

    if (!file_is_good || file_is_not_good) {
        return {};
    }

    std::vector<size_t> ids;

    for (std::string line{}; std::getline(local_file, line);) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        size_t id{};
        double pos_x{};
        double pos_y{};
        double pos_z{};
        std::string area_name{};
        std::string signal_type{};

        std::stringstream sstream(line);
        bool success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> signal_type);

        if (!success) {
            return {};
        }

        if (!ids.empty()) {
            const auto last_id = ids[ids.size() - 1];

            if (last_id >= id) {
                return {};
            }
        }

        ids.push_back(id);
    }

    return ids;
}
