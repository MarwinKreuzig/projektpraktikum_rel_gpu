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

#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/Synapse.h"
#include "util/MPIRank.h"
#include "util/TaggedID.h"
#include "util/Vec3.h"

#include <cstdint>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

enum class NeuronModelEnum {
    Poisson,
    Izhikevich,
    AEIF,
    FitzHughNagumo
};
inline std::string string(const NeuronModelEnum& neuron_model_enum) {
    if (neuron_model_enum == NeuronModelEnum::Poisson) {
        return "Poisson";
    }

    if (neuron_model_enum == NeuronModelEnum::Izhikevich) {
        return "Izhikevich";
    }

    if (neuron_model_enum == NeuronModelEnum::AEIF) {
        return "AEIF";
    }

    if (neuron_model_enum == NeuronModelEnum::FitzHughNagumo) {
        return "FitzHughNagumo";
    }

    return "";
}
/**
 * @brief Pretty-prints the algorithm to the chosen stream
 * @param out The stream to which to print the algorithm
 * @param neuron_model_enum The algorithm to print
 * @return The argument out, now altered with the algorithm
 */
inline std::ostream& operator<<(std::ostream& out, const NeuronModelEnum& neuron_model_enum) {
    return out << string(neuron_model_enum);
}
template <>
struct fmt::formatter<NeuronModelEnum> : ostream_formatter { };

namespace RelearnTypes {
// In the future, these might become different types
using box_size_type = Vec3d;
using position_type = Vec3d;

using synapse_weight = double;
using neuron_id = size_t;

using counter_type = unsigned int;

using step_type = std::uint32_t;
using number_neurons_type = std::uint64_t;

using area_name = std::string;
using area_id = size_t;

using stimuli_list_type = std::vector<std::pair<std::unordered_set<NeuronID>, double>>;
using stimuli_function_type = std::function<stimuli_list_type(step_type)>;

} // namespace RelearnTypes

using LocalSynapse = Synapse<NeuronID, NeuronID, RelearnTypes::synapse_weight>;
using DistantInSynapse = Synapse<NeuronID, RankNeuronId, RelearnTypes::synapse_weight>;
using DistantOutSynapse = Synapse<RankNeuronId, NeuronID, RelearnTypes::synapse_weight>;
using DistantSynapse = Synapse<RankNeuronId, RankNeuronId, RelearnTypes::synapse_weight>;

using LocalSynapses = std::vector<LocalSynapse>;
using DistantInSynapses = std::vector<DistantInSynapse>;
using DistantOutSynapses = std::vector<DistantOutSynapse>;
using DistantSynapses = std::vector<DistantSynapse>;
