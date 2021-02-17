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

#include "Config.h"
#include "RelearnException.h"

#include <map>
#include <random>

namespace randomNumberSeeds {
extern unsigned int partition;
extern unsigned int octree;
} // namespace randomNumberSeeds

enum class RandomHolderKey : char {
    Octree = 0,
    Partition = 1,
    SubdomainFromNeuronDensity = 2,
    ModelA = 3,
    Neurons = 4,
    NeuronModels = 5,
};

class RandomHolder {
    RandomHolder() = default;

    static std::map<RandomHolderKey, std::mt19937> random_number_generators;

public:
    static std::mt19937& get_random_generator(RandomHolderKey key) {
        return random_number_generators[key];
    }

    static double get_random_uniform_double(RandomHolderKey key, double lower_inclusive, double upper_exclusive) {
        RelearnException::check(lower_inclusive < upper_exclusive, "Random number from invalid interval");
        std::uniform_real_distribution<double> urd(lower_inclusive, upper_exclusive);

        return urd(random_number_generators[key]);
    }

    static double get_random_normal_double(RandomHolderKey key, double mean, double sigma) {
        RelearnException::check(0.0 < sigma, "Random number from invalid deviation");
        std::normal_distribution<double> nd(mean, sigma);

        return nd(random_number_generators[key]);
    }

    static void seed(RandomHolderKey key, unsigned int seed) {
        random_number_generators[key].seed(seed);
    }
};
