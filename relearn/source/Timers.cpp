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

#include "Config.h"
#include "LogFiles.h"
#include "MPIWrapper.h"

#include <array>
#include <iomanip>
#include <sstream>

namespace GlobalTimers {
Timers timers(TimerRegion::NUM_TIMER_REGIONS);
} // namespace GlobalTimers

void Timers::print() {
    /**
	 * Print timers and memory usage
	 */
    RelearnException::check(3 * TimerRegion::NUM_TIMER_REGIONS == 66, "Number of timers are unfitting");

    std::array<double, 66> timers_local{};

    for (int i = 0; i < TimerRegion::NUM_TIMER_REGIONS; ++i) {
        const double elapsed = GlobalTimers::timers.get_elapsed(i);

        for (int j = 0; j < 3; ++j) {
            timers_local[3 * i + j] = elapsed;
        }
    }

    std::array<double, 66> timers_global{};

    MPIWrapper::reduce(timers_local, timers_global, MPIWrapper::ReduceFunction::minsummax, 0, MPIWrapper::Scope::global);
    std::stringstream sstring;

    // Divide second entry of (min, sum, max), i.e., sum, by the number of ranks
    // so that sum becomes average
    for (int i = 0; i < TimerRegion::NUM_TIMER_REGIONS; i++) {
        timers_global[3 * i + 1] /= MPIWrapper::get_num_ranks();
    }

    if (0 == MPIWrapper::get_my_rank()) {
        // Set precision for aligned double output
        const auto old_precision = sstring.precision();
        sstring.precision(6);

        sstring << "\n======== TIMERS GLOBAL OVER ALL RANKS ========"
                << "\n";
        sstring << "                                                (" << std::setw(Constants::print_width) << "    min"
                << " | " << std::setw(Constants::print_width) << "    avg"
                << " | " << std::setw(Constants::print_width) << "    max"
                << ") sec."
                << "\n";
        sstring << "TIMERS: main()"
                << "\n";

        sstring << "  Initialization                               : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INITIALIZATION] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INITIALIZATION + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INITIALIZATION + 2] << "\n";

        sstring << "  Simulation loop                              : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::SIMULATION_LOOP] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::SIMULATION_LOOP + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::SIMULATION_LOOP + 2] << "\n";

        sstring << "    Update electrical activity                 : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_ELECTRICAL_ACTIVITY] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_ELECTRICAL_ACTIVITY + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_ELECTRICAL_ACTIVITY + 2] << "\n";

        sstring << "      Prepare sending spikes                   : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_SENDING_SPIKES] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_SENDING_SPIKES + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_SENDING_SPIKES + 2] << "\n";

        sstring << "      Prepare num neuron ids                   : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_NUM_NEURON_IDS] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_NUM_NEURON_IDS + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_NUM_NEURON_IDS + 2] << "\n";

        sstring << "      All to all                               : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALL_TO_ALL] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALL_TO_ALL + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALL_TO_ALL + 2] << "\n";

        sstring << "      Alloc mem for neuron ids                 : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALLOC_MEM_FOR_NEURON_IDS] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALLOC_MEM_FOR_NEURON_IDS + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALLOC_MEM_FOR_NEURON_IDS + 2] << "\n";

        sstring << "      Exchange neuron ids                      : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_NEURON_IDS] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_NEURON_IDS + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_NEURON_IDS + 2] << "\n";

        sstring << "      Calculate serial activity setup          : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SERIAL_ACTIVITY] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SERIAL_ACTIVITY + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SERIAL_ACTIVITY + 2] << "\n";

        sstring << "      Calculate synaptic background            : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_BACKGROUND] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_BACKGROUND + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_BACKGROUND + 2] << "\n";

        sstring << "      Calculate synaptic input                 : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_INPUT] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_INPUT + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_INPUT + 2] << "\n";

        sstring << "      Calculate activity                       : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_ACTIVITY] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_ACTIVITY + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_ACTIVITY + 2] << "\n";

        sstring << "    Update #synaptic elements delta            : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA + 2] << "\n";

        sstring << "    Connectivity update                        : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_CONNECTIVITY] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_CONNECTIVITY + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_CONNECTIVITY + 2] << "\n";

        sstring << "      Update #synaptic elements + del synapses : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES + 2] << "\n";

        sstring << "      Update local trees                       : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_LOCAL_TREES] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_LOCAL_TREES + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_LOCAL_TREES + 2] << "\n";

        sstring << "      Exchange branch nodes (w/ Allgather)     : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_BRANCH_NODES] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_BRANCH_NODES + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_BRANCH_NODES + 2] << "\n";

        sstring << "      Insert branch nodes into global tree     : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE + 2] << "\n";

        sstring << "      Update global tree                       : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_GLOBAL_TREE] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_GLOBAL_TREE + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_GLOBAL_TREE + 2] << "\n";

        sstring << "      Find target neurons (w/ RMA)             : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::FIND_TARGET_NEURONS] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::FIND_TARGET_NEURONS + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::FIND_TARGET_NEURONS + 2] << "\n";

        sstring << "      Empty remote nodes cache                 : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EMPTY_REMOTE_NODES_CACHE] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EMPTY_REMOTE_NODES_CACHE + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EMPTY_REMOTE_NODES_CACHE + 2] << "\n";

        sstring << "      Create synapses (w/ Alltoall)            : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CREATE_SYNAPSES] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CREATE_SYNAPSES + 1] << " | "
                << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CREATE_SYNAPSES + 2] << "\n";

        // Restore old precision
        sstring.precision(old_precision);

        //cout << "\n======== TIMERS RANK 0 ========" << "\n";
        //neurons.print_timers();

        //cout << "\n======== MEMORY USAGE RANK 0 ========" << "\n";

        sstring << "\n======== RMA MEMORY ALLOCATOR RANK 0 ========"
                << "\n";
        sstring << "Min num objects available: " << MPIWrapper::get_num_avail_objects() << "\n";
    }

    LogFiles::write_to_file(LogFiles::EventType::Timers, sstring.str(), true);
}
