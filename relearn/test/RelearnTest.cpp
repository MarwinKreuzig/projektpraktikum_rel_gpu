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

#include "io/LogFiles.h"
#include "util/RelearnException.h"

#include <chrono>
#include <iostream>

int RelearnTest::iterations = 10;
double RelearnTest::eps = 0.001;

bool RelearnTest::use_predetermined_seed = false;
unsigned int RelearnTest::predetermined_seed = 2818124801;

bool initialized = false;

void RelearnTest::init() {

    static bool template_initialized = false;

    if (template_initialized) {
        return;
    }

    if (!initialized) {
        initialized = true;

        char* argument = (char*)"./runTests";
        MPIWrapper::init(1, &argument);
    }
    template_initialized = true;
}

void RelearnTest::SetUpTestCaseTemplate() {
    RelearnException::hide_messages = true;
    LogFiles::disable = true;

    init();
}

void RelearnTest::TearDownTestSuite() {
    RelearnException::hide_messages = false;
    LogFiles::disable = false;
}

void RelearnTest::SetUp() {
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
    // Remove tmp files
    for (auto const& entry : std::filesystem::recursive_directory_iterator("./")) {
        if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".tmp") {
            std::filesystem::remove(entry);
            std::cerr << "REMOVED " << entry.path() << std::endl;
        }
    }
}

void RelearnTest::TearDown() {
    std::cerr << "Test finished\n";
}
