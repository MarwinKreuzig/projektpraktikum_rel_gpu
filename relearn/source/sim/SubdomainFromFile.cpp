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

#include "../Config.h"
#include "../io/LogFiles.h"
#include "../sim/NeuronToSubdomainAssignment.h"
#include "../structure/Partition.h"
#include "../util/RelearnException.h"
#include "spdlog/spdlog.h"

#include <cmath>
#include <iostream>
#include <sstream>

SubdomainFromFile::SubdomainFromFile(
    const std::filesystem::path& file_path, const std::optional<std::filesystem::path>& file_path_positions, std::shared_ptr<Partition> partition)
    : NeuronToSubdomainAssignment(std::move(partition))
    , path(file_path) {
    LogFiles::write_to_file(LogFiles::EventType::Cout, false, "Loading: {} \n", file_path);

    neuron_id_translator = std::make_shared<FileNeuronIdTranslator>(this->partition, file_path);
    synapse_loader = std::make_shared<FileSynapseLoader>(this->partition, neuron_id_translator, file_path_positions);

    read_dimensions_from_file();
}

void SubdomainFromFile::read_dimensions_from_file() {
    std::ifstream file(path);

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "SubdomainFromFile::read_dimensions_from_file: Opening the file was not successful");

    box_size_type minimum(std::numeric_limits<box_size_type::value_type>::max());
    box_size_type maximum(std::numeric_limits<box_size_type::value_type>::min());

    size_t found_ex_neurons = 0;
    size_t found_in_neurons = 0;

    size_t total_number_neurons = 0;

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        size_t id{};
        box_size_type::value_type pos_x{};
        box_size_type::value_type pos_y{};
        box_size_type::value_type pos_z{};
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

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Loaded neurons: {}", total_number_neurons);

    const auto requested_number_neurons = found_ex_neurons + found_in_neurons;
    const auto requested_ratio_excitatory_neurons = static_cast<double>(found_ex_neurons) / static_cast<double>(requested_number_neurons);

    const auto simulation_box_length = maximum;

    set_requested_number_neurons(requested_number_neurons);
    set_requested_ratio_excitatory_neurons(requested_ratio_excitatory_neurons);

    partition->set_simulation_box_size({ 0, 0, 0 }, simulation_box_length);
    partition->set_total_number_neurons(total_number_neurons);
}

std::vector<NeuronToSubdomainAssignment::Node> SubdomainFromFile::read_nodes_from_file(const box_size_type& min, const box_size_type& max) {
    std::ifstream file(path);

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "SubdomainFromFile::read_nodes_from_file: Opening the file was not successful");

    double placed_ex_neurons = 0.0;
    double placed_in_neurons = 0.0;

    size_t number_placed_neurons = 0;

    std::vector<NeuronToSubdomainAssignment::Node> nodes;

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        std::string signal_type{};

        size_t id{};
        Node node{};
        box_size_type::value_type pos_x{};
        box_size_type::value_type pos_y{};
        box_size_type::value_type pos_z{};
        std::stringstream sstream(line);
        bool success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> node.area_name) && (sstream >> signal_type);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        // Ids start with 1
        --id;
        node.id = NeuronID{ id };
        node.pos = { pos_x, pos_y, pos_z };

        if (bool is_in_subdomain = node.pos.check_in_box(min, max); !is_in_subdomain) {
            continue;
        }

        if (signal_type == "ex") {
            node.signal_type = SignalType::EXCITATORY;
            ++placed_ex_neurons;
        } else {
            node.signal_type = SignalType::INHIBITORY;
            ++placed_in_neurons;
        }

        ++number_placed_neurons;
        nodes.emplace_back(node);
    }

    const auto ratio_placed_excitatory_neurons = placed_ex_neurons / static_cast<double>(number_placed_neurons);

    set_number_placed_neurons(number_placed_neurons);
    set_ratio_placed_excitatory_neurons(ratio_placed_excitatory_neurons);

    return nodes;
}

std::vector<NeuronID> SubdomainFromFile::get_neuron_global_ids_in_subdomain(const size_t subdomain_index_1d, [[maybe_unused]] const size_t total_number_subdomains) const {
    const bool contains = is_subdomain_loaded(subdomain_index_1d);
    if (!contains) {
        RelearnException::fail("SubdomainFromFile::get_neuron_global_ids_in_subdomain: Wanted to have neuron_global_ids of subdomain_index_1d that is not present");
        return {};
    }

    const Nodes& nodes = get_nodes_for_subdomain(subdomain_index_1d);
    std::vector<NeuronID> global_ids;
    global_ids.reserve(nodes.size());

    for (const Node& node : nodes) {
        global_ids.push_back(node.id);
    }

    return global_ids;
}

void SubdomainFromFile::fill_subdomain(const size_t local_subdomain_index, [[maybe_unused]] const size_t total_number_subdomains) {
    const auto subdomain_index_1d = partition->get_1d_index_of_subdomain(local_subdomain_index);
    const bool subdomain_already_filled = is_subdomain_loaded(subdomain_index_1d);
    if (subdomain_already_filled) {
        RelearnException::fail("SubdomainFromFile::fill_subdomain: Tried to fill an already filled subdomain.");
        return;
    }

    Nodes nodes{};

    const auto& [min, max] = partition->get_subdomain_boundaries(local_subdomain_index);
    auto nodes_vector = read_nodes_from_file(min, max);
    for (const auto& node : nodes_vector) {
        nodes.emplace(node);
    }

    set_nodes_for_subdomain(subdomain_index_1d, std::move(nodes));
}

std::optional<std::vector<size_t>> SubdomainFromFile::read_neuron_ids_from_file(const std::filesystem::path& file_path) {
    std::ifstream local_file(file_path);

    const bool file_is_good = local_file.good();
    const bool file_is_not_good = local_file.fail() || local_file.eof();

    if (!file_is_good || file_is_not_good) {
        return {};
    }

    std::vector<size_t> ids;

    for (std::string line{}; std::getline(local_file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        size_t id{};
        box_size_type::value_type pos_x{};
        box_size_type::value_type pos_y{};
        box_size_type::value_type pos_z{};
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
