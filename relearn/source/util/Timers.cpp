/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Timers.h"

#include "../Config.h"
#include "../io/LogFiles.h"
#include "../mpi/MPIWrapper.h"

#include <array>
#include <iomanip>

void Timers::print_timer(std::stringstream& sstream, const TimerRegion timer_index, const std::array<double, size_t(3) * NUM_TIMERS>& timers) {
    sstream
        << std::setw(Constants::print_width) << timers[3 * static_cast<size_t>(timer_index)] << " | "
        << std::setw(Constants::print_width) << timers[3 * static_cast<size_t>(timer_index) + 1] << " | "
        << std::setw(Constants::print_width) << timers[3 * static_cast<size_t>(timer_index) + 2] << "\n";
}

void Timers::print() {
    /**
	 * Print timers and memory usage
	 */
    constexpr size_t expected_num_timers = size_t(3) * NUM_TIMERS;

    std::array<double, expected_num_timers> timers_local{};

    for (size_t i = 0; i < NUM_TIMERS; ++i) {
        const auto timer = static_cast<TimerRegion>(i);
        const double elapsed = get_elapsed(timer);

        for (int j = 0; j < 3; ++j) {
            const size_t idx = 3 * i + j;
            // NOLINTNEXTLINE
            timers_local[idx] = elapsed;
        }
    }

    std::array<double, expected_num_timers> timers_global = MPIWrapper::reduce(timers_local, MPIWrapper::ReduceFunction::minsummax, 0);
    std::stringstream sstring{};

    // Divide second entry of (min, sum, max), i.e., sum, by the number of ranks
    // so that sum becomes average
    for (size_t i = 0; i < NUM_TIMERS; i++) {
        const size_t idx = 3 * i + 1;
        // NOLINTNEXTLINE
        timers_global[idx] /= MPIWrapper::get_num_ranks();
    }

    if (0 != MPIWrapper::get_my_rank()) {
        return;
    }

    // Set precision for aligned double output
    const auto old_precision = sstring.precision();
    sstring.precision(Constants::print_precision);

    sstring << "\n======== TIMERS GLOBAL OVER ALL RANKS ========\n";
    sstring << "                                                ("
            << std::setw(Constants::print_width) << "    min | "
            << std::setw(Constants::print_width) << "    avg | "
            << std::setw(Constants::print_width) << "    max) sec.\n";
    sstring << "TIMERS: main()\n";

    sstring << "  Initialization                               : ";
    print_timer(sstring, TimerRegion::INITIALIZATION, timers_global);

    sstring << "  Simulation loop                              : ";
    print_timer(sstring, TimerRegion::SIMULATION_LOOP, timers_global);

    sstring << "    Update electrical activity                 : ";
    print_timer(sstring, TimerRegion::UPDATE_ELECTRICAL_ACTIVITY, timers_global);

    sstring << "      Prepare sending spikes                   : ";
    print_timer(sstring, TimerRegion::PREPARE_SENDING_SPIKES, timers_global);

    sstring << "      Prepare num neuron ids                   : ";
    print_timer(sstring, TimerRegion::PREPARE_NUM_NEURON_IDS, timers_global);

    sstring << "      All to all                               : ";
    print_timer(sstring, TimerRegion::ALL_TO_ALL, timers_global);

    sstring << "      Alloc mem for neuron ids                 : ";
    print_timer(sstring, TimerRegion::ALLOC_MEM_FOR_NEURON_IDS, timers_global);

    sstring << "      Exchange neuron ids                      : ";
    print_timer(sstring, TimerRegion::EXCHANGE_NEURON_IDS, timers_global);

    sstring << "      Calculate serial activity setup          : ";
    print_timer(sstring, TimerRegion::CALC_SERIAL_ACTIVITY, timers_global);

    sstring << "      Calculate synaptic background            : ";
    print_timer(sstring, TimerRegion::CALC_SYNAPTIC_BACKGROUND, timers_global);

    sstring << "      Calculate synaptic input                 : ";
    print_timer(sstring, TimerRegion::CALC_SYNAPTIC_INPUT, timers_global);

    sstring << "      Calculate activity                       : ";
    print_timer(sstring, TimerRegion::CALC_ACTIVITY, timers_global);

    sstring << "    Update #synaptic elements delta            : ";
    print_timer(sstring, TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA, timers_global);

    sstring << "    Connectivity update                        : ";
    print_timer(sstring, TimerRegion::UPDATE_CONNECTIVITY, timers_global);

    sstring << "      Update #synaptic elements + del synapses : ";
    print_timer(sstring, TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES, timers_global);

    sstring << "      Update leaf nodes                        : ";
    print_timer(sstring, TimerRegion::UPDATE_LEAF_NODES, timers_global);

    sstring << "      Update local trees                       : ";
    print_timer(sstring, TimerRegion::UPDATE_LOCAL_TREES, timers_global);

    sstring << "      Exchange branch nodes (w/ Allgather)     : ";
    print_timer(sstring, TimerRegion::EXCHANGE_BRANCH_NODES, timers_global);

    sstring << "      Insert branch nodes into global tree     : ";
    print_timer(sstring, TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE, timers_global);

    sstring << "      Update global tree                       : ";
    print_timer(sstring, TimerRegion::UPDATE_GLOBAL_TREE, timers_global);

    sstring << "      Find target neurons (w/ RMA)             : ";
    print_timer(sstring, TimerRegion::FIND_TARGET_NEURONS, timers_global);

    sstring << "      Empty remote nodes cache                 : ";
    print_timer(sstring, TimerRegion::EMPTY_REMOTE_NODES_CACHE, timers_global);

    sstring << "      Create synapses (w/ Alltoall)            : ";
    print_timer(sstring, TimerRegion::CREATE_SYNAPSES, timers_global);

    sstring << "\n\n";

    LogFiles::write_to_file(LogFiles::EventType::Timers, true, sstring.str());

    const auto avg_time = timers_global[3 * static_cast<size_t>(TimerRegion::SIMULATION_LOOP) + 1];

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Simulation time [sec]: {}", avg_time);
}
