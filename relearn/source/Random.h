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
#include <random>

namespace randomNumberSeeds {
extern int64_t partition;
extern int64_t octree;
} // namespace randomNumberSeeds

template <typename T>
class RandomHolder {
public:
    static std::mt19937& get_random_generator() noexcept {
        // NOLINTNEXTLINE
        static std::mt19937 random_generator;
        return random_generator;
    }
};
