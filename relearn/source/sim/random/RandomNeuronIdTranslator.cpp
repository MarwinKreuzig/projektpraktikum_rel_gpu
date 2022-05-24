/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RandomNeuronIdTranslator.h"

#include "../../mpi/MPIWrapper.h"
#include "../../structure/Partition.h"
#include "../../util/RelearnException.h"

#include "spdlog/spdlog.h"

#include <algorithm>

RandomNeuronIdTranslator::RandomNeuronIdTranslator(std::shared_ptr<Partition> partition)
    : RandomNeuronIdTranslator(std::move(partition), MPIWrapper::all_gather<size_t>) { }

RandomNeuronIdTranslator::RandomNeuronIdTranslator(std::shared_ptr<Partition> partition, const std::function<std::vector<size_t>(size_t)>& gather_function)
    : partition(std::move(partition)) {
    number_local_neurons = this->partition->get_number_local_neurons();

    const auto num_ranks = this->partition->get_number_mpi_ranks();
    mpi_rank_to_local_start_id.resize(num_ranks);

    // Gather the number of neurons of every process
    const auto rank_to_num_neurons = gather_function(number_local_neurons);

    // Store global start neuron id of every rank
    mpi_rank_to_local_start_id[0] = 0;
    for (size_t i = 1; i < num_ranks; i++) {
        mpi_rank_to_local_start_id[i] = mpi_rank_to_local_start_id[i - 1] + rank_to_num_neurons[i - 1];
    }
}

bool RandomNeuronIdTranslator::is_neuron_local(NeuronID global_id) const {
    const auto global_neuron_id = global_id.get_global_id();
    const auto my_rank = partition->get_my_mpi_rank();

    const auto my_global_ids_start = mpi_rank_to_local_start_id[my_rank];
    const auto my_global_ids_end = my_global_ids_start + number_local_neurons;

    const auto is_local = my_global_ids_start <= global_neuron_id && global_neuron_id < my_global_ids_end;

    return is_local;
}

NeuronID RandomNeuronIdTranslator::get_local_id(NeuronID global_id) const {
    const auto global_neuron_id = global_id.get_global_id();
    const auto my_rank = partition->get_my_mpi_rank();

    const auto my_global_ids_start = mpi_rank_to_local_start_id[my_rank];
    const auto my_global_ids_end = my_global_ids_start + number_local_neurons;

    const auto is_local = my_global_ids_start <= global_neuron_id && global_neuron_id < my_global_ids_end;

    RelearnException::check(is_local, "RandomNeuronIdTranslator::get_local_id: The global id is not local! {} vs [{}, {})", global_id, my_global_ids_start, my_global_ids_end);

    return NeuronID{ false, false, global_neuron_id - my_global_ids_start };
}

NeuronID RandomNeuronIdTranslator::get_global_id(NeuronID local_id) const {
    const auto local_neuron_id = local_id.get_local_id();
    RelearnException::check(local_neuron_id < number_local_neurons, "RandomNeuronIdTranslator::get_local_id: The local id is too large! {} vs {}", local_id, number_local_neurons);

    const auto my_rank = partition->get_my_mpi_rank();
    const auto my_global_ids_start = mpi_rank_to_local_start_id[my_rank];

    return NeuronID{ true, false, local_neuron_id + my_global_ids_start };
}

void RandomNeuronIdTranslator::create_neurons(size_t number_local_creations) {
    const auto num_ranks = MPIWrapper::get_num_ranks();
    RelearnException::check(num_ranks == 1, "RandomNeuronIdTranslator::create_neurons: Can only create neurons for files with one mpi rank, but there were {}", num_ranks);

    number_local_neurons += number_local_creations;
}
