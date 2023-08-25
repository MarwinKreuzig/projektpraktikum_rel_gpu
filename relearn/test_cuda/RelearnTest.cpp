/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"


#include <chrono>
#include <iostream>

//TODO
//Test CudaVector
//Free everything between tests
//Problem reusing __constant__ in Tests

int RelearnGPUTest::iterations = 10;
double RelearnGPUTest::eps = 0.001;

bool RelearnGPUTest::use_predetermined_seed = false;
unsigned int RelearnGPUTest::predetermined_seed = 2818124801;

RelearnGPUTest::RelearnGPUTest() {
}

RelearnGPUTest::~RelearnGPUTest() {
}

void RelearnGPUTest::SetUp() {
    //CudaHelper::set_use_cuda(false);
    
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

void RelearnGPUTest::TearDown() {
    // Remove tmp files
    for (auto const& entry : std::filesystem::recursive_directory_iterator("./")) {
        if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".tmp") {
            std::filesystem::remove(entry);
            std::cerr << "REMOVED " << entry.path() << std::endl;
        }
    }

    std::cerr << "Test finished\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    const auto tests_return_code = RUN_ALL_TESTS();

    return tests_return_code;
}
