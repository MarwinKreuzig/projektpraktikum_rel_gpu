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

#include <cstdint>
#include <map>
#include <random>

namespace randomNumberSeeds {
extern unsigned int partition;
extern unsigned int octree;
} // namespace randomNumberSeeds

class RandomHolder {
    RandomHolder() = default;

    std::map<char, std::mt19937> random_number_generators;

public:

    static RandomHolder& get_instance() noexcept {
        static RandomHolder instance;
        return instance;
    }

    std::mt19937& get_random_generator(char key) {
        return random_number_generators[key];
    }

    void seed(char key, unsigned int seed) {
        random_number_generators[key].seed(seed);
    }

    constexpr static char OCTREE = 0;
    constexpr static char PARTITION = 1;
    constexpr static char SubdomainFromNeuronDensity = 2;
    constexpr static char ModelA = 3;
    constexpr static char Neurons = 4;
    constexpr static char NeuronModel = 5;
};
