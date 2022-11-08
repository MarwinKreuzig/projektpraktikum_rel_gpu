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

#include "io/NeuronIO.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"

#include <filesystem>

FileSynapseLoader::FileSynapseLoader(std::shared_ptr<Partition> partition, std::optional<std::filesystem::path> path_to_synapses)
    : SynapseLoader(std::move(partition))
    , optional_path_to_file(std::move(path_to_synapses)) {
    RelearnException::check(this->partition->get_number_mpi_ranks() == 1 && this->partition->get_my_mpi_rank() == 0,
        "FileSynapseLoader::FileSynapseLoader: Can only use this class with 1 MPI rank.");
}

FileSynapseLoader::synapses_pair_type FileSynapseLoader::internal_load_synapses() {
    if (!optional_path_to_file.has_value()) {
        return synapses_pair_type{};
    }

    const auto& actual_path = optional_path_to_file.value();

    const auto expected_in_name = "rank_0_in_network.txt";
    const auto expected_out_name = "rank_0_out_network.txt";

    const auto number_local_neurons = partition->get_number_local_neurons();

    auto [read_local_in_synapses, read_distant_in_synapses] = NeuronIO::read_in_synapses(actual_path / expected_in_name, number_local_neurons, 0, 1);
    auto [read_local_out_synapses, read_distant_out_synapses] = NeuronIO::read_out_synapses(actual_path / expected_out_name, number_local_neurons, 0, 1);

    auto return_synapses_plastic = std::make_tuple(std::move(read_local_in_synapses), std::move(read_distant_in_synapses), std::move(read_distant_out_synapses));
    return synapses_pair_type({}, std::move(return_synapses_plastic));
}
