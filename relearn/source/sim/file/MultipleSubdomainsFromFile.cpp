/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MultipleSubdomainsFromFile.h"

#include "Config.h"
#include "io/LogFiles.h"
#include "io/NeuronIO.h"
#include "sim/file/MultipleFilesSynapseLoader.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"

#include "spdlog/spdlog.h"

MultipleSubdomainsFromFile::MultipleSubdomainsFromFile(const std::filesystem::path& path_to_neurons, 
	std::optional<std::filesystem::path> path_to_synapses, std::shared_ptr<Partition> partition) 
: NeuronToSubdomainAssignment(partition) {
    RelearnException::check(partition->get_number_mpi_ranks() > 1, "MultipleSubdomainsFromFile::MultipleSubdomainsFromFile: There was only one MPI rank.");
    const auto my_position_filename = "rank_" + std::to_string(partition->get_my_mpi_rank()) + "_positions.txt";
    const auto& path_to_file = path_to_neurons / my_position_filename;

    RelearnException::check(std::filesystem::exists(path_to_file), "MultipleSubdomainsFromFile::MultipleSubdomainsFromFile: position file {} does not exist.", path_to_file.string());

    synapse_loader = std::make_shared<MultipleFilesSynapseLoader>(std::move(partition), std::move(path_to_synapses));

    read_neurons_from_file(path_to_file);
}

void MultipleSubdomainsFromFile::read_neurons_from_file(const std::filesystem::path& path_to_neurons) {
    const auto& comments = NeuronIO::read_comments(path_to_neurons);

    auto search = [&comments](const char specifier[13]) -> double {
        double value{};

        for (const auto& comment : comments) {
            const auto position = comment.find(specifier);
            if (position == std::string::npos) {
                continue;
            }

            return std::stod(comment.substr(position + 13));
        }

        RelearnException::fail("MultipleSubdomainsFromFile::read_neurons_from_file: Did not find comment containing {}", specifier);
    };

    auto check = [](double value) -> bool {
        const auto min = MPIWrapper::reduce(value, MPIWrapper::ReduceFunction::Min, value);
        const auto max = MPIWrapper::reduce(value, MPIWrapper::ReduceFunction::Max, value);

        return min != max;
    };

    const auto min_x = search("# Minimum x:");
    const auto min_y = search("# Minimum y:");
    const auto min_z = search("# Minimum z:");
    const auto max_x = search("# Maximum x:");
    const auto max_y = search("# Maximum y:");
    const auto max_z = search("# Maximum z:");

    const auto all_same_min_x = check(min_x);
    const auto all_same_min_y = check(min_y);
    const auto all_same_min_z = check(min_z);
    const auto all_same_max_x = check(max_x);
    const auto all_same_max_y = check(max_y);
    const auto all_same_max_z = check(max_z);

    RelearnException::check(all_same_min_x, "MultipleSubdomainsFromFile::read_neurons_from_file: min_x is different across the ranks!");
    RelearnException::check(all_same_min_y, "MultipleSubdomainsFromFile::read_neurons_from_file: min_y is different across the ranks!");
    RelearnException::check(all_same_min_z, "MultipleSubdomainsFromFile::read_neurons_from_file: min_z is different across the ranks!");
    RelearnException::check(all_same_max_x, "MultipleSubdomainsFromFile::read_neurons_from_file: max_x is different across the ranks!");
    RelearnException::check(all_same_max_y, "MultipleSubdomainsFromFile::read_neurons_from_file: max_y is different across the ranks!");
    RelearnException::check(all_same_max_z, "MultipleSubdomainsFromFile::read_neurons_from_file: max_z is different across the ranks!");

    RelearnTypes::box_size_type minimum{ min_x, min_y, min_z };
    RelearnTypes::box_size_type maximum{ max_x, max_y, max_z };

    auto [nodes, additional_infos] = NeuronIO::read_neurons(path_to_neurons);
    const auto& [_1, _2, loaded_ex_neurons, loaded_in_neurons] = additional_infos;

    // TODO: Let partition calculate the local portion and then check if all neurons are in it

    partition->set_simulation_box_size(minimum, maximum);

    const auto total_number_neurons = loaded_ex_neurons + loaded_in_neurons;
    set_total_number_placed_neurons(total_number_neurons);
    set_requested_number_neurons(total_number_neurons);
    set_number_placed_neurons(total_number_neurons);

    const auto ratio_excitatory_neurons = static_cast<double>(loaded_ex_neurons) / static_cast<double>(total_number_neurons);

    set_requested_ratio_excitatory_neurons(ratio_excitatory_neurons);
    set_ratio_placed_excitatory_neurons(ratio_excitatory_neurons);

    partition->set_total_number_neurons(total_number_neurons);

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Loaded neurons: {}", total_number_neurons);

    set_loaded_nodes(std::move(nodes));
}
