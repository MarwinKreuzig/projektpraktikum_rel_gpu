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

    create_rank_to_start_neuron_id_mapping(rank_to_num_neurons);

    create_pos_to_rank_neuron_id_mapping(
        rank_to_num_neurons,
        my_num_neurons,
        x, y, z);
}

std::tuple<bool, size_t> NeuronIdMap::rank_neuron_id2glob_id(const RankNeuronId& rank_neuron_id) /*noexcept*/ {
    // Rank is not valid
    if (rank_neuron_id.get_rank() < 0 || rank_neuron_id.get_rank() > (rank_to_start_neuron_id.size() - 1)) {
        return std::make_tuple(false, Constants::uninitialized);
    }

    size_t glob_id = rank_to_start_neuron_id[rank_neuron_id.get_rank()] + rank_neuron_id.get_neuron_id();
    return std::make_tuple(true, glob_id);
}

std::tuple<bool, RankNeuronId> NeuronIdMap::pos2rank_neuron_id(const Vec3d& pos) {
    auto it = pos_to_rank_neuron_id.find(pos);

    // Neuron position not found
    if (it == pos_to_rank_neuron_id.end()) {
        return std::make_tuple(false, RankNeuronId{ -1, Constants::uninitialized });
    }

    // Return rank and neuron id
    RankNeuronId result = it->second;
    return std::make_tuple(true, result);
}

void NeuronIdMap::create_rank_to_start_neuron_id_mapping(const std::vector<size_t>& rank_to_num_neurons) {
    const size_t num_ranks = rank_to_num_neurons.size();
    rank_to_start_neuron_id.resize(num_ranks);

    // Store global start neuron id of every rank
    rank_to_start_neuron_id[0] = 0;
    for (size_t i = 1; i < num_ranks; i++) {
        rank_to_start_neuron_id[i] = rank_to_start_neuron_id[i - 1] + rank_to_num_neurons[i - 1];
    }
}

void NeuronIdMap::create_pos_to_rank_neuron_id_mapping(
    const std::vector<size_t>& rank_to_num_neurons,
    size_t my_num_neurons,
    const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z) {

    const auto num_ranks = MPIWrapper::get_num_ranks();
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto total_num_neurons = rank_to_start_neuron_id[num_ranks - 1] + rank_to_num_neurons[num_ranks - 1];
    std::vector<double> xyz_pos(total_num_neurons * 3);

    // Copy my neuron positions as xyz-triple into the send buffer
    for (size_t i = 0; i < my_num_neurons; i++) {
        auto idx = (rank_to_start_neuron_id[my_rank] + i) * 3;
        RelearnException::check(idx < total_num_neurons * 3, "idx is too large in neuronidmap");

        xyz_pos[idx] = x[i];
        xyz_pos[idx + 1] = y[i];
        xyz_pos[idx + 2] = z[i];
    }

    std::vector<int> recvcounts(num_ranks);
    std::vector<int> displs(num_ranks);
    for (size_t i = 0; i < num_ranks; i++) {
        RelearnException::check(rank_to_num_neurons[i] <= std::numeric_limits<int>::max(), "rank to neuron is too large in neuronidmap");
        recvcounts[i] = static_cast<int>(rank_to_num_neurons[i]);

        RelearnException::check(rank_to_start_neuron_id[i] <= std::numeric_limits<int>::max(), "rank to start is too large in neuronidmap");
        displs[i] = static_cast<int>(rank_to_start_neuron_id[i]);
    }

    MPIWrapper::all_gather_v(total_num_neurons, xyz_pos, recvcounts, displs);

    // Map every neuron position to one (rank, neuron_id) pair
    size_t glob_neuron_id = 0;
    for (int rank = 0; rank < num_ranks; rank++) {
        for (size_t neuron_id = 0; neuron_id < rank_to_num_neurons[rank]; neuron_id++) {
            RelearnException::check(glob_neuron_id < total_num_neurons, "global id is too large in neuronidmap");

            const auto idx = glob_neuron_id * 3;

            RelearnException::check(idx < xyz_pos.size(), "idx is too large in neuronidmap");
            Vec3d key{ xyz_pos[idx], xyz_pos[idx + 1], xyz_pos[idx + 2] };

            RankNeuronId val{ rank, neuron_id };
            auto ret = pos_to_rank_neuron_id.insert(std::make_pair(key, val));
            RelearnException::check(ret.second, "there is a duplicate in neuronidmap"); // New element was inserted, otherwise duplicates exist

            glob_neuron_id++;
        }
    }
}
