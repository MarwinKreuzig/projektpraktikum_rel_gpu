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

#include <optional>
#include <tuple>
#include <vector>

class NeuronIdMap {
public:
    static void init(size_t my_num_neurons);

    [[nodiscard]] static std::optional<size_t> rank_neuron_id2glob_id(const RankNeuronId& rank_neuron_id) /*noexcept*/;

private:
    static inline std::vector<size_t> rank_to_start_neuron_id{}; // Global neuron id of every rank's first local neuron
};
