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

#include "RelearnException.h"

#include <mpi.h>

#include <limits>
#include <map>
#include <utility>
#include <vector>

NeuronIdMap::NeuronIdMap(size_t my_num_neurons,
	const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
	MPI_Comm mpi_comm) {
	int num_ranks;
	MPI_Comm_size(mpi_comm, &num_ranks);

	// Gather the number of neurons of every process
	std::vector<size_t> rank_to_num_neurons(num_ranks);
	MPI_Allgather(&my_num_neurons, sizeof(size_t), MPI_CHAR, rank_to_num_neurons.data(),
		sizeof(size_t), MPI_CHAR, mpi_comm);

	create_rank_to_start_neuron_id_mapping(rank_to_num_neurons, rank_to_start_neuron_id);

	create_pos_to_rank_neuron_id_mapping(
		rank_to_num_neurons,
		rank_to_start_neuron_id,
		my_num_neurons,
		x, y, z,
		mpi_comm,
		pos_to_rank_neuron_id);
}

bool NeuronIdMap::rank_neuron_id2glob_id(const RankNeuronId&
	rank_neuron_id, size_t& glob_id) const /*noexcept*/ {
	// Rank is not valid
	if (rank_neuron_id.rank < 0 ||
		rank_neuron_id.rank > (rank_to_start_neuron_id.size() - 1))
		return false;

	glob_id = rank_to_start_neuron_id[rank_neuron_id.rank] + rank_neuron_id.neuron_id;
	return true;
}

bool NeuronIdMap::pos2rank_neuron_id(const Vec3d& pos, RankNeuronId& result) const {
	auto it = pos_to_rank_neuron_id.find(pos);

	// Neuron position not found
	if (it == pos_to_rank_neuron_id.end()) {
		return false;
	}

	// Return rank and neuron id
	result = it->second;
	return true;
}

void NeuronIdMap::create_rank_to_start_neuron_id_mapping(
	const std::vector<size_t>& rank_to_num_neurons,
	std::vector<size_t>& rank_to_start_neuron_id) {
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
	const std::vector<size_t>& rank_to_start_neuron_id,
	size_t my_num_neurons,
	const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
	MPI_Comm mpi_comm,
	std::map<Vec3d, RankNeuronId>& pos_to_rank_neuron_id) {
	int num_ranks, my_rank;
	MPI_Comm_size(mpi_comm, &num_ranks);
	MPI_Comm_rank(mpi_comm, &my_rank);

	const size_t total_num_neurons = rank_to_start_neuron_id[static_cast<long long>(num_ranks) - 1] + rank_to_num_neurons[static_cast<long long>(num_ranks) - 1];
	std::vector<double> xyz_pos(total_num_neurons * 3);

	// Copy my neuron positions as xyz-triple into the send buffer
	for (size_t i = 0; i < my_num_neurons; i++) {
		auto idx = (rank_to_start_neuron_id[my_rank] + i) * 3;
		RelearnException::check(idx < total_num_neurons * 3);

		xyz_pos[idx] = x[i];
		xyz_pos[idx + 1] = y[i];
		xyz_pos[idx + 2] = z[i];
	}

	// Create MPI data type for three doubles
	MPI_Datatype type;
	MPI_Type_contiguous(3, MPI_DOUBLE, &type);
	MPI_Type_commit(&type);

	std::vector<int> recvcounts(num_ranks);
	std::vector<int> displs(num_ranks);
	for (int i = 0; i < num_ranks; i++) {
		RelearnException::check(rank_to_num_neurons[i] <= std::numeric_limits<int>::max());
		recvcounts[i] = static_cast<int>(rank_to_num_neurons[i]);

		RelearnException::check(rank_to_start_neuron_id[i] <= std::numeric_limits<int>::max());
		displs[i] = static_cast<int>(rank_to_start_neuron_id[i]);
	}

	// Receive all neuron positions as xyz-triples
	MPI_Allgatherv(MPI_IN_PLACE, static_cast<int>(total_num_neurons), type,
		xyz_pos.data(), recvcounts.data(),
		displs.data(), type, mpi_comm);

	MPI_Type_free(&type);

	// Map every neuron position to one (rank, neuron_id) pair
	size_t glob_neuron_id = 0;
	for (int rank = 0; rank < num_ranks; rank++) {
		RankNeuronId val{ 0, 0 };
		val.rank = rank;
		for (size_t neuron_id = 0; neuron_id < rank_to_num_neurons[rank]; neuron_id++) {
			RelearnException::check(glob_neuron_id < total_num_neurons);

			const auto idx = glob_neuron_id * 3;

			RelearnException::check(idx < xyz_pos.size());
			Vec3d key{ xyz_pos[idx], xyz_pos[idx + 1], xyz_pos[idx + 2] };
			val.neuron_id = neuron_id;

			auto ret = pos_to_rank_neuron_id.insert(std::make_pair(key, val));
			RelearnException::check(ret.second); // New element was inserted, otherwise duplicates exist

			glob_neuron_id++;
		}
	}
}
