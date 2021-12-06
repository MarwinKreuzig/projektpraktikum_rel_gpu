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
#include "../util/Random.h"

void NeuronsExtraInfo::init(const size_t number_neurons) {
    RelearnException::check(size == 0, "NeuronsExtraInfo::init: NeuronsExtraInfo initialized two times");
    size = number_neurons;

    const int num_ranks = MPIWrapper::get_num_ranks();

    // Gather the number of neurons of every process
    std::vector<size_t> rank_to_num_neurons(num_ranks);

    MPIWrapper::all_gather(number_neurons, rank_to_num_neurons);
    mpi_rank_to_local_start_id.resize(num_ranks);

    // Store global start neuron id of every rank
    mpi_rank_to_local_start_id[0] = 0;
    for (size_t i = 1; i < num_ranks; i++) {
        mpi_rank_to_local_start_id[i] = mpi_rank_to_local_start_id[i - 1] + rank_to_num_neurons[i - 1];
    }
}

void NeuronsExtraInfo::create_neurons(const size_t creation_count) {
    RelearnException::check(creation_count != 0, "Cannot add 0 neurons");

    RelearnException::check(!positions.empty(), "NeuronsExtraInfo::create_neurons: positions must not be empty");

    const auto num_ranks = MPIWrapper::get_num_ranks();

    RelearnException::check(num_ranks == 1, "NeuronsExtraInfo::create_neurons: Cannot create neurons if more than 1 MPI rank is computing");

    const auto current_size = size;
    const auto new_size = current_size + creation_count;

    area_names.resize(new_size, "UNKNOWN (inserted by creation");

    positions.resize(new_size);

    for (size_t i = current_size; i < new_size; i++) {
        const auto x_it = RandomHolder::get_random_uniform_double(RandomHolderKey::NeuronsExtraInformation, 0.0, 1.0);
        const auto y_it = RandomHolder::get_random_uniform_double(RandomHolderKey::NeuronsExtraInformation, 0.0, 1.0);
        const auto z_it = RandomHolder::get_random_uniform_double(RandomHolderKey::NeuronsExtraInformation, 0.0, 1.0);

        const auto x_pos = positions[static_cast<size_t>(x_it * current_size)].get_x();
        const auto y_pos = positions[static_cast<size_t>(y_it * current_size)].get_y();
        const auto z_pos = positions[static_cast<size_t>(z_it * current_size)].get_z();

        positions[i] = { x_pos, y_pos, z_pos };
    }

    size = new_size;
}
