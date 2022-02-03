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

#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/Synapse.h"
#include "util/Vec3.h"

#include <vector>

enum class AlgorithmEnum {
    Naive,
    BarnesHut,
    FastMultipoleMethods,
};

namespace RelearnTypes {
// In the future, these might become different types
using box_size_type = Vec3d;
using position_type = Vec3d;

using neuron_id = size_t;
using synapse_weight = int;

} // namespace RelearnTypes

using LocalSynapse = Synapse<RelearnTypes::neuron_id, RelearnTypes::neuron_id, RelearnTypes::synapse_weight>;
using DistantInSynapse = Synapse<RelearnTypes::neuron_id, RankNeuronId, RelearnTypes::synapse_weight>;
using DistantOutSynapse = Synapse<RankNeuronId, RelearnTypes::neuron_id, RelearnTypes::synapse_weight>;
using DistantSynapse = Synapse<RankNeuronId, RankNeuronId, RelearnTypes::synapse_weight>;

using LocalSynapses = std::vector<LocalSynapse>;
using DistantInSynapses = std::vector<DistantInSynapse>;
using DistantOutSynapses = std::vector<DistantOutSynapse>;
using DistantSynapses = std::vector<DistantSynapse>;
