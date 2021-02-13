/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronIdMap.h"

#include "Config.h"
#include "MPIWrapper.h"
#include "RelearnException.h"

#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

void NeuronIdMap::init(size_t my_num_neurons, const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z) {
    const int num_ranks = MPIWrapper::get_num_ranks();

    // Gather the number of neurons of every process
    std::vector<size_t> rank_to_num_neurons(num_ranks);

    MPIWrapper::all_gather(my_num_neurons, rank_to_num_neurons, MPIWrapper::Scope::global);
    rank_to_start_neuron_id.resize(num_ranks);

    // Store global start neuron id of every rank
    rank_to_start_neuron_id[0] = 0;
    for (size_t i = 1; i < num_ranks; i++) {
        rank_to_start_neuron_id[i] = rank_to_start_neuron_id[i - 1] + rank_to_num_neurons[i - 1];
    }
}

std::optional<size_t> NeuronIdMap::rank_neuron_id2glob_id(const RankNeuronId& rank_neuron_id) /*noexcept*/ {
    // Rank is not valid
    if (rank_neuron_id.get_rank() < 0 || rank_neuron_id.get_rank() > (rank_to_start_neuron_id.size() - 1)) {
        return {};
    }

    const auto glob_id = rank_to_start_neuron_id[rank_neuron_id.get_rank()] + rank_neuron_id.get_neuron_id();
    return glob_id;
}
