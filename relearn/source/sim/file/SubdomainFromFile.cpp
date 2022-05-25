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
#include "io/LogFiles.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "sim/file/FileSynapseLoader.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"

#include "spdlog/spdlog.h"

#include <cmath>
#include <iostream>
#include <sstream>

SubdomainFromFile::SubdomainFromFile(const std::filesystem::path& path_to_neurons, std::optional<std::filesystem::path> path_to_synapses, std::shared_ptr<Partition> partition)
    : NeuronToSubdomainAssignment(partition) {
    RelearnException::check(partition->get_my_mpi_rank() == 0 && partition->get_number_mpi_ranks() == 1, "SubdomainFromFile::SubdomainFromFile: Can only be used for 1 MPI rank.");

    LogFiles::write_to_file(LogFiles::EventType::Cout, false, "Loading: {} \n", path_to_neurons);

    synapse_loader = std::make_shared<FileSynapseLoader>(std::move(partition), std::move(path_to_synapses));

    read_neurons_from_file(path_to_neurons);
}

void SubdomainFromFile::read_neurons_from_file(const std::filesystem::path& path_to_neurons) {
    std::ifstream file(path_to_neurons);

    const bool file_is_good = file.good();
    const bool file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "SubdomainFromFile::read_neurons_from_file: Opening the file was not successful");

    box_size_type minimum(std::numeric_limits<box_size_type::value_type>::max());
    box_size_type maximum(std::numeric_limits<box_size_type::value_type>::min());

    size_t found_ex_neurons = 0;
    size_t found_in_neurons = 0;

    std::vector<NeuronToSubdomainAssignment::Node> nodes{};

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        NeuronID::value_type id{};
        box_size_type::value_type pos_x{};
        box_size_type::value_type pos_y{};
        box_size_type::value_type pos_z{};
        std::string area_name{};
        std::string signal_type{};

        std::stringstream sstream(line);
        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> signal_type);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        RelearnException::check(pos_x >= 0, "SubdomainFromFile::read_neurons_from_file: x position of neuron {} was negative: {}", id, pos_x);
        RelearnException::check(pos_y >= 0, "SubdomainFromFile::read_neurons_from_file: y position of neuron {} was negative: {}", id, pos_y);
        RelearnException::check(pos_z >= 0, "SubdomainFromFile::read_neurons_from_file: z position of neuron {} was negative: {}", id, pos_z);

        box_size_type position{ pos_x, pos_y, pos_z };

        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);

        if (signal_type == "in") {
            found_in_neurons++;
            nodes.emplace_back(position, NeuronID{ false, id }, SignalType::Inhibitory, std::move(area_name));
        } else {
            found_ex_neurons++;
            nodes.emplace_back(position, NeuronID{ false, id }, SignalType::Excitatory, std::move(area_name));
        }
    }

    const auto new_max_x = std::nextafter(maximum.get_x(), maximum.get_x() + Constants::eps);
    const auto new_max_y = std::nextafter(maximum.get_y(), maximum.get_y() + Constants::eps);
    const auto new_max_z = std::nextafter(maximum.get_z(), maximum.get_z() + Constants::eps);

    partition->set_simulation_box_size({ 0.0, 0.0, 0.0 }, { new_max_x, new_max_y, new_max_z });

    const auto total_number_neurons = found_ex_neurons + found_in_neurons;
    set_total_number_placed_neurons(total_number_neurons);

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Loaded neurons: {}", total_number_neurons);

    const auto requested_ratio_excitatory_neurons = static_cast<double>(found_ex_neurons) / static_cast<double>(total_number_neurons);

    set_requested_number_neurons(total_number_neurons);
    set_requested_ratio_excitatory_neurons(requested_ratio_excitatory_neurons);

    const auto ratio_placed_excitatory_neurons = found_ex_neurons / static_cast<double>(total_number_neurons);

    set_number_placed_neurons(total_number_neurons);
    set_ratio_placed_excitatory_neurons(ratio_placed_excitatory_neurons);

    partition->set_total_number_neurons(total_number_neurons);

    set_nodes_for_subdomain(0, std::move(nodes));
}

std::optional<std::vector<NeuronID>> SubdomainFromFile::read_neuron_ids_from_file(const std::filesystem::path& file_path) {
    std::ifstream local_file(file_path);

    const bool file_is_good = local_file.good();
    const bool file_is_not_good = local_file.fail() || local_file.eof();

    if (!file_is_good || file_is_not_good) {
        return {};
    }

    std::vector<NeuronID> ids{};

    for (std::string line{}; std::getline(local_file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        NeuronID::value_type id{};
        box_size_type::value_type pos_x{};
        box_size_type::value_type pos_y{};
        box_size_type::value_type pos_z{};
        std::string area_name{};
        std::string signal_type{};

        std::stringstream sstream(line);
        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> signal_type);

        if (!success) {
            return {};
        }

        if (!ids.empty()) {
            const auto last_id = ids[ids.size() - 1].get_local_id();

            if (last_id >= id) {
                return {};
            }
        }

        ids.emplace_back(false, id);
    }

    return ids;
}
