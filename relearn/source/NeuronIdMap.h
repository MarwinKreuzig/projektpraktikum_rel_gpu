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

#include "RankNeuronId.h"
#include "Vec3.h"

#include <map>
#include <tuple>
#include <vector>

class NeuronIdMap {
public:
    static void init(size_t my_num_neurons, const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z);

    [[nodiscard]] static std::tuple<bool, size_t> rank_neuron_id2glob_id(const RankNeuronId& rank_neuron_id) /*noexcept*/;

    [[nodiscard]] static std::tuple<bool, RankNeuronId> pos2rank_neuron_id(const Vec3d& pos);

private:
    static void create_rank_to_start_neuron_id_mapping(const std::vector<size_t>& rank_to_num_neuronsd);

    static void create_pos_to_rank_neuron_id_mapping(
        const std::vector<size_t>& rank_to_num_neurons,
        size_t my_num_neurons,
        const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z);

    static inline std::vector<size_t> rank_to_start_neuron_id{}; // Global neuron id of every rank's first local neuron
    static inline std::map<Vec3d, RankNeuronId> pos_to_rank_neuron_id{};
};
