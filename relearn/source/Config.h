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

#include <cstddef>

#ifdef _OPENMP
constexpr bool OPENMPAVAILABLE = true;
#else
constexpr bool OPENMPAVAILABLE = false;
#endif

// This exists for easier switching of compilation modes
// NOLINTNEXTLINE
#define RELEARN_MPI_FOUND MPI_FOUND

class Constants {
public:
    constexpr static size_t number_oct = 8;
    constexpr static size_t uninitialized = 1111222233334444;

    constexpr static size_t max_lvl_subdomains = 20;

    constexpr static size_t num_items_per_request = 6;

    constexpr static double eps = 0.00001;

    constexpr static size_t print_width = 22;
    constexpr static size_t print_precision = 8;

    // Update connectivity every 100 ms
    inline static size_t plasticity_update_step = 100;

    // Print details every 100 ms
    inline static size_t logfile_update_step = 100;

    // Print to cout every 100 ms
    inline static size_t console_update_step = 100;

    // Capture individual neuron informations ever 100 ms
    inline static size_t monitor_step = 100;

    // Capture the global statistics every 100 ms
    inline static size_t statistics_step = 100;

    // Capture the calcium values every 10000 ms
    inline static size_t calcium_step = 10000;

    constexpr static size_t mpi_alloc_mem = 1024 * 1024 * 300;

    //Constants for Fast Gauss
    constexpr static int p = 4;
    constexpr static int p3 = p * p * p;
    constexpr static int max_neurons_in_target = 70; //cutoff for target box
    constexpr static int max_neurons_in_source = 70; //cutoff for source box
};

namespace Config {
constexpr bool do_debug_checks = false;
} // namespace Config
