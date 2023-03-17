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

#include "adapter/random/RandomAdapter.h"

#include "adapter/simulation/SimulationAdapter.h"

#include "util/TaggedID.h"
#include "util/Vec3.h"

#include <random>
#include <tuple>
#include <vector>

class NeuronsAdapter {
public:
    static std::vector<std::tuple<Vec3d, NeuronID>> generate_random_neurons(const Vec3d& min, const Vec3d& max, size_t count, size_t max_id, std::mt19937& mt) {
        std::vector<NeuronID> ids(max_id);
        for (auto i = 0; i < max_id; i++) {
            ids[i] = NeuronID(i);
        }
        RandomAdapter::shuffle(ids.begin(), ids.end(), mt);

        std::vector<std::tuple<Vec3d, NeuronID>> return_value(count);
        for (auto i = 0; i < count; i++) {
            return_value[i] = { SimulationAdapter::get_random_position_in_box(min, max, mt), ids[i] };
        }

        return return_value;
    }
};
