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

#include "../../../source/algorithm/BarnesHutCell.h"
#include "../../../source/structure/OctreeNode.h"
#include "../../../source/mpi/MPIWrapper.h"
#include "../../../source/io/LogFiles.h"
#include "../../../source/util/MemoryHolder.h"
#include "../../../source/util/RelearnException.h"

#include <chrono>
#include <map>
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
        MPIWrapper::init_buffer_octree<BarnesHutCell>();
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
        if (use_predetermined_seed) {
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

    void make_mpi_mem_available() {
        MemoryHolder<BarnesHutCell>::make_all_available();
    }

    static int iterations;
    static size_t num_neurons_test;
    static double eps;

    static bool use_predetermined_seed;
    static unsigned int predetermined_seed;
};

class NetworkGraphTest : public RelearnTest {
protected:
    static size_t upper_bound_num_neurons;
    static int bound_synapse_weight;
    static int num_ranks;
    static int num_synapses_per_neuron;

    template <typename T>
    void erase_empty(std::map<T, int>& edges) {
        for (auto iterator = edges.begin(); iterator != edges.end();) {
            if (iterator->second == 0) {
                iterator = edges.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }

    template <typename T>
    void erase_empties(std::map<T, std::map<T, int>>& edges) {
        for (auto iterator = edges.begin(); iterator != edges.end();) {
            erase_empty<T>(iterator->second);

            if (iterator->second.empty()) {
                iterator = edges.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }
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

class SpaceFillingCurveTest : public RelearnTest {
};
