/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once 

#include "Vec3.h"

#include <map>
#include <tuple>
#include <vector>

class NeuronIdMap {
public:
	// Rank and local neuron id
	struct RankNeuronId {
		size_t rank;
		size_t neuron_id;
	};

	NeuronIdMap(size_t my_num_neurons, const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z);

	[[nodiscard]] std::tuple<bool, size_t> rank_neuron_id2glob_id(const RankNeuronId& rank_neuron_id) const /*noexcept*/;

	[[nodiscard]] std::tuple<bool, RankNeuronId> pos2rank_neuron_id(const Vec3d& pos) const;

private:
	void create_rank_to_start_neuron_id_mapping(
		const std::vector<size_t>& rank_to_num_neurons,
		std::vector<size_t>& rank_to_start_neuron_id);

	void create_pos_to_rank_neuron_id_mapping(
		const std::vector<size_t>& rank_to_num_neurons,
		const std::vector<size_t>& rank_to_start_neuron_id,
		size_t my_num_neurons,
		const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
		std::map<Vec3d, RankNeuronId>& pos_to_rank_neuron_id);

	std::vector<size_t> rank_to_start_neuron_id;  // Global neuron id of every rank's first local neuron
	std::map<Vec3d, RankNeuronId> pos_to_rank_neuron_id;
};
