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
#include "util/StringUtil.h"

#include <filesystem>

FileSynapseLoader::FileSynapseLoader(std::shared_ptr<Partition> partition, std::optional<std::filesystem::path> path_to_synapses)
    : SynapseLoader(std::move(partition))
    , optional_path_to_file(std::move(path_to_synapses)) {
    RelearnException::check(this->partition->get_number_mpi_ranks() == 1 && this->partition->get_my_mpi_rank() == MPIRank::root_rank(),
        "FileSynapseLoader::FileSynapseLoader: Can only use this class with 1 MPI rank.");
}

FileSynapseLoader::synapses_pair_type FileSynapseLoader::internal_load_synapses() {
    if (!optional_path_to_file.has_value()) {
        return synapses_pair_type{};
    }

    const auto& actual_path = optional_path_to_file.value();

    const std::filesystem::path path_to_in_file = StringUtil::find_file_for_rank(actual_path, MPIRank::root_rank().get_rank(), "rank_", "_in_network.txt", 5);
    const std::filesystem::path path_to_out_file = StringUtil::find_file_for_rank(actual_path, MPIRank::root_rank().get_rank(), "rank_", "_out_network.txt", 5);

    const auto number_local_neurons = partition->get_number_local_neurons();

    auto [in_synapses_static, in_synapses_plastic] = NeuronIO::read_in_synapses(path_to_in_file, number_local_neurons, MPIRank::root_rank(), 1);
    auto [read_local_in_synapses_static, read_distant_in_synapses_static] = in_synapses_static;
    auto [read_local_in_synapses_plastic, read_distant_in_synapses_plastic] = in_synapses_plastic;

    auto [out_synapses_static, out_synapses_plastic] = NeuronIO::read_out_synapses(path_to_out_file, number_local_neurons, MPIRank::root_rank(), 1);
    auto [read_local_out_synapses_static, read_distant_out_synapses_static] = out_synapses_static;
    auto [read_local_out_synapses_plastic, read_distant_out_synapses_plastic] = out_synapses_plastic;

    auto return_synapses_plastic = std::make_tuple(std::move(read_local_in_synapses_plastic), std::move(read_distant_in_synapses_plastic), std::move(read_distant_out_synapses_plastic));
    auto return_synapses_static = std::make_tuple(std::move(read_local_in_synapses_static), std::move(read_distant_in_synapses_static), std::move(read_distant_out_synapses_static));
    
    return { std::move(return_synapses_static), std::move(return_synapses_plastic) };
}
