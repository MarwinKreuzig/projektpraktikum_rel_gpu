/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FileSynapseLoader.h"

#include "structure/Partition.h"
#include "util/RelearnException.h"

#include <fstream>
#include <set>
#include <sstream>
#include <string>

FileSynapseLoader::FileSynapseLoader(std::shared_ptr<Partition> partition, std::optional<std::filesystem::path> path_to_synapses)
    : SynapseLoader(std::move(partition))
    , optional_path_to_file(std::move(path_to_synapses)) {
    RelearnException::check(this->partition->get_number_mpi_ranks() == 1 && this->partition->get_my_mpi_rank() == 0,
        "FileSynapseLoader::FileSynapseLoader: Can only use this class with 1 MPI rank.");
}

FileSynapseLoader::synapses_tuple_type FileSynapseLoader::internal_load_synapses() {
    if (!optional_path_to_file.has_value()) {
        return {};
    }

    const auto total_number_neurons = partition->get_total_number_neurons();

    LocalSynapses local_synapses{};

    const auto& path_to_file = optional_path_to_file.value();
    std::ifstream file_synapses(path_to_file, std::ios::binary | std::ios::in);

    for (std::string line{}; std::getline(file_synapses, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        NeuronID::value_type read_source_id = 0;
        NeuronID::value_type read_target_id = 0;
        RelearnTypes::synapse_weight weight = 0;

        std::stringstream sstream(line);
        const bool success = (sstream >> read_source_id) && (sstream >> read_target_id) && (sstream >> weight);

        RelearnException::check(success, "FileSynapseLoader::internal_load_synapses: Loading synapses was unsuccessfull!");

        RelearnException::check(read_source_id > 0 && read_source_id <= total_number_neurons, "FileSynapseLoader::internal_load_synapses: source_id was not from [1, {}]: {}", total_number_neurons, read_source_id);
        RelearnException::check(read_target_id > 0 && read_target_id <= total_number_neurons, "FileSynapseLoader::internal_load_synapses: target_id was not from [1, {}]: {}", total_number_neurons, read_target_id);
        
        RelearnException::check(weight != 0, "FileSynapseLoader::internal_load_synapses: weight was 0");

        // The neurons start with 1
        --read_source_id;
        --read_target_id;
        auto source_id = NeuronID{ false, read_source_id };
        auto target_id = NeuronID{ false, read_target_id };
        
        local_synapses.emplace_back(source_id, target_id, weight);
    }

    auto return_synapses = std::make_tuple(std::move(local_synapses), DistantInSynapses{}, DistantOutSynapses{});
    return return_synapses;
}
