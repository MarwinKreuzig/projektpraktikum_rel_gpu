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

#include "../neurons/helper/RankNeuronId.h"
#include "../util/Vec3.h"

#include <filesystem>
#include <map>
#include <vector>

class NeuronIdTranslator {
    using position_type = RelearnTypes::position_type;

public:
    std::map<size_t, RankNeuronId> translate_global_ids(const std::vector<size_t>& global_ids, const std::filesystem::path& path_to_neurons);

private:
    std::map<size_t, position_type> load_neuron_positions(const std::vector<size_t>& global_ids, const std::filesystem::path& path_neurons);
};