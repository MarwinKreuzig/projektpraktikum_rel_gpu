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

#include "../../../source/mpi/MPIWrapper.h"
#include "../../../source/io/LogFiles.h"
#include "../../../source/util/RelearnException.h"

#include <chrono>
#include <random>

class RelearnTest : public ::testing::Test {
private:
    static void init() {
        static bool initialized = false;

        if (initialized) {
            return;
        }

        initialized = true;

        char* argument = (char*)"./runTests";
        MPIWrapper::init(1, &argument);
        MPIWrapper::init_buffer_octree(1);
    }

protected:
    std::mt19937 mt;

    static void SetUpTestCase() {
        RelearnException::hide_messages = true;
        LogFiles::disable = true;

        init();
    }

    static void TearDownTestCase() {
        RelearnException::hide_messages = false;
        LogFiles::disable = false;
    }

    void SetUp() override {
        if constexpr (use_predetermined_seed) {
            std::cerr << "Using predetermined seed: " << predetermined_seed << '\n';
            mt.seed(predetermined_seed);
        } else {
            const auto now = std::chrono::high_resolution_clock::now();
            const auto time_since_epoch = now.time_since_epoch();
            const auto time_since_epoch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch).count();

            const auto seed = static_cast<unsigned int>(time_since_epoch_ns);

            std::cerr << "Test seed: " << seed << '\n';
            mt.seed(seed);
        }
    }

    void TearDown() override {
        std::cerr << "Test finished\n";
    }

    constexpr static int iterations = 10;
    constexpr static size_t num_neurons_test = 1000;
    constexpr static double eps = 0.00001;

    constexpr static bool use_predetermined_seed = true;
    constexpr static unsigned int predetermined_seed = 983859766;
};

class NetworkGraphTest : public RelearnTest {
};

class NeuronAssignmentTest : public RelearnTest {
};

class NeuronModelsTest : public RelearnTest {
};

class NeuronsTest : public RelearnTest {
};

class OctreeTest : public RelearnTest {
};

class PartitionTest : public RelearnTest {

};

class SynapticElementsTest : public RelearnTest {

};

class VectorTest : public RelearnTest {

};
