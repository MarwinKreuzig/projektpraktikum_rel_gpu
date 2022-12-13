#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RandomAdapter.h"
#include "mpi/mpi_rank_adapter.h"
#include "tagged_id/tagged_id_adapter.h"

#include "neurons/helper/RankNeuronId.h"

#include <random>

class RankNeuronIdAdapter {
public:
    static RankNeuronId generate_random_rank_neuron_id(std::mt19937& mt) {
        const auto rank = MPIRankAdapter::get_random_mpi_rank(mt);
        const auto neuron_id = TaggedIdAdapter::get_random_neuron_id(mt);

        return { rank, neuron_id };
    }
};
