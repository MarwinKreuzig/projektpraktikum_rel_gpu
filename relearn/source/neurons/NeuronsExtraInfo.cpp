/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronsExtraInfo.h"

#include "../mpi/MPIWrapper.h"

void NeuronsExtraInfo::init(size_t number_neurons) noexcept {
    RelearnException::check(size == 0, "NeuronsExtraInfo initialized two times");
    size = number_neurons;

    const int num_ranks = MPIWrapper::get_num_ranks();

    // Gather the number of neurons of every process
    std::vector<size_t> rank_to_num_neurons(num_ranks);

    MPIWrapper::all_gather(number_neurons, rank_to_num_neurons, MPIWrapper::Scope::global);
    mpi_rank_to_local_start_id.resize(num_ranks);

    // Store global start neuron id of every rank
    mpi_rank_to_local_start_id[0] = 0;
    for (size_t i = 1; i < num_ranks; i++) {
        mpi_rank_to_local_start_id[i] = mpi_rank_to_local_start_id[i - 1] + rank_to_num_neurons[i - 1];
    }
}
