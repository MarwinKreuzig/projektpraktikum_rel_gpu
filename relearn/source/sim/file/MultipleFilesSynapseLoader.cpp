/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MultipleFilesSynapseLoader.h"

#include "io/NeuronIO.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"

#include <filesystem>

MultipleFilesSynapseLoader::MultipleFilesSynapseLoader(std::shared_ptr<Partition> partition, std::optional<std::filesystem::path> path_to_synapses)
    : SynapseLoader(std::move(partition))
    , optional_path_to_file(std::move(path_to_synapses)) {
    RelearnException::check(this->partition->get_number_mpi_ranks() > 1, "MultipleFilesSynapseLoader::MultipleFilesSynapseLoader: Can only use this class with >1 MPI ranks.");
    if (optional_path_to_file.has_value()) {
        const auto& actual_path = optional_path_to_file.value();
        RelearnException::check(std::filesystem::is_directory(actual_path), "MultipleFilesSynapseLoader::MultipleFilesSynapseLoader: Path {} is no directory.", actual_path);
    }
}

MultipleFilesSynapseLoader::synapses_tuple_type MultipleFilesSynapseLoader::internal_load_synapses() {
    if (!optional_path_to_file.has_value()) {
        return synapses_tuple_type{};
    }

    const auto number_local_neurons = partition->get_number_local_neurons();
    const auto my_rank = partition->get_my_mpi_rank();
    const auto number_ranks = partition->get_number_mpi_ranks();

    const auto& actual_path = optional_path_to_file.value();

    const auto expected_in_name = "rank_" + std::to_string(my_rank) + "_in_network.txt";
    const auto expected_out_name = "rank_" + std::to_string(my_rank) + "_out_network.txt";

    auto [read_local_in_synapses, read_distant_in_synapses] = NeuronIO::read_in_synapses(actual_path / expected_in_name, number_local_neurons, my_rank, number_ranks);
    auto [read_local_out_synapses, read_distant_out_synapses] = NeuronIO::read_out_synapses(actual_path / expected_out_name, number_local_neurons, my_rank, number_ranks);

    return { std::move(read_local_in_synapses), std::move(read_distant_in_synapses), std::move(read_distant_out_synapses) };
}
