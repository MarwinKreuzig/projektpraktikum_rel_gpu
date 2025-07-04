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

#include "../source/gpu/utils/Macros.h"

#include <cmath>
#include <filesystem>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <gtest/gtest-typed-test.h>

/**
 * @brief Get the path to relearn/relearn
 *
 * @return std::filesystem::path path
 */
[[nodiscard]] std::filesystem::path get_relearn_path();

class RelearnGPUTest : public ::testing::Test {
protected:
    RelearnGPUTest();

    virtual ~RelearnGPUTest();

    void SetUp() override; // Called immediately after the constructor for each test

    void TearDown() override; // Called immediately before the destructor for each test

    size_t round_to_next_exponent(size_t numToRound, size_t exponent) {
        auto log = std::log(static_cast<double>(numToRound)) / std::log(static_cast<double>(exponent));
        auto rounded_exp = std::ceil(log);
        auto new_val = std::pow(static_cast<double>(exponent), rounded_exp);
        return static_cast<size_t>(new_val);
    }

    constexpr static int number_neurons_out_of_scope = 100;

    std::mt19937 mt;

    static int iterations;
    static double eps;

private:
    static bool use_predetermined_seed;
    static unsigned int predetermined_seed;
};
