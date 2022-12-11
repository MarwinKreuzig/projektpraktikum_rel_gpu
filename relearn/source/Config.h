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

#include <cstddef>
#include <cstdint>

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

    constexpr static size_t number_prealloc_space = 30;

    constexpr static size_t max_lvl_subdomains = 20;

    constexpr static double eps = 0.00001;

    constexpr static size_t print_width = 22;
    constexpr static size_t print_precision = 8;

    constexpr static size_t mpi_alloc_mem = 1024 * 1024 * 300;

    // Constants for Fast Gauss
    constexpr static unsigned int p = 4;
    constexpr static unsigned int p3 = p * p * p;
    constexpr static unsigned int max_neurons_in_target = 70; // cutoff for target box
    constexpr static unsigned int max_neurons_in_source = 70; // cutoff for source box

    constexpr static unsigned int unpacking = 0; // indicates how many levels a node is unpacked to give the synaptic elements more choice to connect
    // only used when FMM is selected. When unpacking == 0 normal FMM is used.

    constexpr static double bh_default_theta{ 0.3 };
    constexpr static double bh_max_theta{ 0.5 };
};

class Config {
public:
    constexpr static bool do_debug_checks = false;

    // Update connectivity every <plasticity_update_step> ms
    inline static std::uint32_t plasticity_update_step = 100; // NOLINT

    // Update connectivity starting at <first_plasticity_update> ms
    inline static std::uint32_t first_plasticity_update = 0; // NOLINT

    // End the connectivity updates at <last_plascitiy_update> ms
    inline static std::uint32_t last_plasticity_update = -1; // NOLINT

    // Print details every <logfile_update_step> ms
    inline static std::uint32_t logfile_update_step = 100; // NOLINT

    // Print to cout every <console_update_step> ms
    inline static std::uint32_t console_update_step = 100; // NOLINT

    // Capture individual neuron informations ever <monitor_step> ms
    inline static std::uint32_t monitor_step = 100; // NOLINT

    // Capture ensemble informations ever <monitor_area_step> ms
    inline static std::uint32_t monitor_area_step = 100; // NOLINT

    // Capture the global statistics every <statistics_step> ms
    inline static std::uint32_t statistics_step = 100; // NOLINT

    // Capture the calcium values every <calcium_log_step> ms
    inline static std::uint32_t calcium_log_step = 1000000; // NOLINT

    // Capture the network every <network_log_step> ms
    inline static std::uint32_t network_log_step = 10000; // NOLINT

    // Capture the syanptic input every <synaptic_input_log_step> ms
    inline static std::uint32_t synaptic_input_log_step = 10000; // NOLINT

    // Capture the average local euclidean distance every <distance_step> ms
    inline static std::uint32_t distance_step = 10000; // NOLINT

    // Flush the neuron monitors every <flush_monitor_step> ms
    inline static std::uint32_t flush_monitor_step = 30000; // NOLINT
};
