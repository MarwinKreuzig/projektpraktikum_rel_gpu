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

FileSynapseLoader::FileSynapseLoader(std::shared_ptr<Partition> partition, std::optional<std::filesystem::path> path_to_synapses)
        : SynapseLoader(std::move(partition))
        , optional_path_to_file(std::move(path_to_synapses)) {
    RelearnException::check(this->partition->get_number_mpi_ranks() == 1 && this->partition->get_my_mpi_rank() == 0,
                            "FileSynapseLoader::FileSynapseLoader: Can only use this class with 1 MPI rank.");
}

FileSynapseLoader::synapses_pair_type FileSynapseLoader::internal_load_synapses() {
    if (!optional_path_to_file.has_value()) {
        return {};
    }

    const auto total_number_neurons = partition->get_total_number_neurons();

    const auto& path_to_file = optional_path_to_file.value();
    auto [local_synapses_static, local_synapses_plastic] = NeuronIO::read_local_synapses(path_to_file, total_number_neurons);

    auto return_synapses_static = std::make_tuple(std::move(local_synapses_static), DistantInSynapses{}, DistantOutSynapses{});
    auto return_synapses_plastic = std::make_tuple(std::move(local_synapses_plastic), DistantInSynapses{}, DistantOutSynapses{});
    return std::make_pair(return_synapses_static, return_synapses_plastic);
}