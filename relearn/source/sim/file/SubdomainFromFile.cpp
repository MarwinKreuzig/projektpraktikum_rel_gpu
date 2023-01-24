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
#include "io/NeuronIO.h"
#include "sim/Essentials.h"
#include "sim/file/FileSynapseLoader.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"

#include "fmt/std.h"

SubdomainFromFile::SubdomainFromFile(const std::filesystem::path& path_to_neurons, std::optional<std::filesystem::path> path_to_synapses, std::shared_ptr<Partition> partition)
    : NeuronToSubdomainAssignment(partition) {
    RelearnException::check(partition->get_my_mpi_rank() == MPIRank::root_rank() && partition->get_number_mpi_ranks() == 1, "SubdomainFromFile::SubdomainFromFile: Can only be used for 1 MPI rank.");

    LogFiles::write_to_file(LogFiles::EventType::Cout, false, "Loading: {} \n", path_to_neurons);

    synapse_loader = std::make_shared<FileSynapseLoader>(std::move(partition), std::move(path_to_synapses));

    read_neurons_from_file(path_to_neurons);
}

void SubdomainFromFile::print_essentials(const std::unique_ptr<Essentials>& essentials) {
    essentials->insert("Neurons-Loaded", get_total_number_placed_neurons());
}

void SubdomainFromFile::read_neurons_from_file(const std::filesystem::path& path_to_neurons) {
    auto [loaded_neurons, area_id_vs_area_name, additional_infos] = NeuronIO::read_neurons(path_to_neurons);
    this->set_area_id_to_area_name(area_id_vs_area_name);
    const auto& [min_position, max_position, loaded_ex_neurons, loaded_in_neurons] = additional_infos;

    partition->set_simulation_box_size({ 0.0, 0.0, 0.0 }, max_position);

    const auto total_number_neurons = loaded_ex_neurons + loaded_in_neurons;
    set_total_number_placed_neurons(total_number_neurons);
    set_requested_number_neurons(total_number_neurons);
    set_number_placed_neurons(total_number_neurons);

    const auto ratio_excitatory_neurons = static_cast<double>(loaded_ex_neurons) / static_cast<double>(total_number_neurons);

    set_requested_ratio_excitatory_neurons(ratio_excitatory_neurons);
    set_ratio_placed_excitatory_neurons(ratio_excitatory_neurons);

    partition->set_total_number_neurons(total_number_neurons);

    set_loaded_nodes(std::move(loaded_neurons));
    create_local_area_translator(total_number_neurons);
}
