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

#include "gtest/gtest.h"
#include "gtest/gtest-typed-test.h"

#include "mpi/MPIWrapper.h"
#include "util/MemoryHolder.h"

#include <cmath>
#include <random>
#include <vector>

/**
 * @brief Get the path to relearn/relearn
 *
 * @return std::filesystem::path path
 */
[[nodiscard]] std::filesystem::path get_relearn_path();

class RelearnTest : public ::testing::Test {
protected:
    static void init();

protected:
    static void SetUpTestCaseTemplate();

    static void SetUpTestSuite();

    static void TearDownTestSuite();

    void SetUp() override;

    void TearDown() override;

    template <typename AdditionalCellAttributes>
    void make_mpi_mem_available() {
        MemoryHolder<AdditionalCellAttributes>::make_all_available();
    }

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

class RelearnTestWithAdditionalCellAttribute : public RelearnTest {
protected:
    template <typename AdditionalCellAttributes>
    static void init() {
        RelearnTest::init();
        MPIWrapper::init_buffer_octree<AdditionalCellAttributes>();
    }

protected:
    template <typename AdditionalCellAttributes>
    static void SetUpTestCaseTemplate() {
        RelearnTest::SetUpTestCaseTemplate();
        init<AdditionalCellAttributes>();
    }
};
