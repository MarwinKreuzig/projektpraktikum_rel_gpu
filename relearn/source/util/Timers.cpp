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
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"

#include <array>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <time.h>

std::string Timers::wall_clock_time() {
#ifdef __linux__
    time_t rawtime = 0;
    time(&rawtime);
    // NOLINTNEXTLINE
    struct tm* timeinfo = localtime(&rawtime);
    // NOLINTNEXTLINE
    char* string = asctime(timeinfo);

    // Only copy 24 characters
    return std::string(string, 24);
#else
    time_t rawtime = 0;
    struct tm timeinfo;
    char char_buff[30];

    time(&rawtime);
    localtime_s(&timeinfo, &rawtime);
    asctime_s(char_buff, &timeinfo);

    // Remove linebreak in string
    // NOLINTNEXTLINE
    char_buff[24] = '\0';

    return std::string(char_buff);
#endif
}

void Timers::print() {
    /**
     * Print timers and memory usage
     */
    constexpr size_t expected_num_timers = size_t(3) * NUMBER_TIMERS;

    std::array<double, expected_num_timers> timers_local{};

    std::stringstream local_timer_output{};

    for (size_t i = 0; i < NUMBER_TIMERS; ++i) {
        const auto timer = static_cast<TimerRegion>(i);
        const auto elapsed = get_elapsed(timer);

        local_timer_output << elapsed << '\n';

        for (auto j = 0U; j < 3; ++j) {
            const auto idx = 3 * i + j;
            const auto counted = elapsed.count();
            const auto seconds = static_cast<double>(counted) * 1e-9;

            // NOLINTNEXTLINE
            timers_local[idx] = seconds;
        }
    }

    LogFiles::write_to_file(LogFiles::EventType::TimersLocal, false, local_timer_output.str());

    auto timers_global = MPIWrapper::reduce(timers_local, MPIWrapper::ReduceFunction::MinSumMax, 0);
    if (0 != MPIWrapper::get_my_rank()) {
        return;
    }

    std::stringstream sstring{};

    auto print_timer = [&timers_global, &sstring](auto message, const TimerRegion timer_index) {
        const auto min_time = timers_global[3 * static_cast<size_t>(timer_index)];
        const auto avg_time = timers_global[3 * static_cast<size_t>(timer_index) + 1];
        const auto max_time = timers_global[3 * static_cast<size_t>(timer_index) + 2];

        sstring << message
                << std::setw(Constants::print_width) << min_time << " | "
                << std::setw(Constants::print_width) << avg_time << " | "
                << std::setw(Constants::print_width) << max_time << '\n';
    };

    // Divide second entry of (min, sum, max), i.e., sum, by the number of ranks
    // so that sum becomes average
    for (size_t i = 0; i < NUMBER_TIMERS; i++) {
        const size_t idx = 3 * i + 1;
        // NOLINTNEXTLINE
        timers_global[idx] /= MPIWrapper::get_num_ranks();
    }

    // Set precision for aligned double output
    sstring.precision(Constants::print_precision);

    sstring << "\n======== TIMERS GLOBAL OVER ALL RANKS ========\n";
    sstring << "                                                ("
            << std::setw(Constants::print_width) << " min"
            << " | "
            << std::setw(Constants::print_width) << " avg"
            << " | "
            << std::setw(Constants::print_width) << " max"
            << ") sec.\n";
    sstring << "TIMERS: main()\n";

    print_timer("  Initialization                               : ", TimerRegion::INITIALIZATION);
    print_timer("    Load Synapses                              : ", TimerRegion::LOAD_SYNAPSES);
    print_timer("    Translate Global IDs                       : ", TimerRegion::TRANSLATE_GLOBAL_IDS);
    print_timer("    Initialize Network Graph                   : ", TimerRegion::INITIALIZE_NETWORK_GRAPH);
    print_timer("  Simulation loop                              : ", TimerRegion::SIMULATION_LOOP);
    print_timer("    Update electrical activity                 : ", TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);
    print_timer("      Prepare sending spikes                   : ", TimerRegion::PREPARE_SENDING_SPIKES);
    print_timer("      Exchange neuron ids                      : ", TimerRegion::EXCHANGE_NEURON_IDS);
    print_timer("      Calculate serial activity setup          : ", TimerRegion::CALC_SERIAL_ACTIVITY);
    print_timer("      Calculate synaptic background            : ", TimerRegion::CALC_SYNAPTIC_BACKGROUND);
    print_timer("      Calculate synaptic input                 : ", TimerRegion::CALC_SYNAPTIC_INPUT);
    print_timer("      Calculate activity                       : ", TimerRegion::CALC_ACTIVITY);
    print_timer("    Update #synaptic elements delta            : ", TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);
    print_timer("    Connectivity update                        : ", TimerRegion::UPDATE_CONNECTIVITY);
    print_timer("      Delete synapses                          : ", TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES);
    print_timer("        Commit #synaptic elements              : ", TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS);
    print_timer("        Find synapses to delete                : ", TimerRegion::FIND_SYNAPSES_TO_DELETE);
    print_timer("        Exchange deletions (w/ alltoall)       : ", TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL);
    print_timer("        Process deletion requests              : ", TimerRegion::PROCESS_DELETE_REQUESTS);
    print_timer("      Update leaf nodes                        : ", TimerRegion::UPDATE_LEAF_NODES);
    print_timer("      Update local trees                       : ", TimerRegion::UPDATE_LOCAL_TREES);
    print_timer("      Exchange branch nodes (w/ Allgather)     : ", TimerRegion::EXCHANGE_BRANCH_NODES);
    print_timer("      Insert branch nodes into global tree     : ", TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE);
    print_timer("      Update global tree                       : ", TimerRegion::UPDATE_GLOBAL_TREE);
    print_timer("      Find target neurons (w/ RMA)             : ", TimerRegion::FIND_TARGET_NEURONS);
    print_timer("        FMM: Calculate Taylor Coefficients     : ", TimerRegion::CALC_TAYLOR_COEFFICIENTS);
    print_timer("        FMM: Calculate Hermite Coefficients    : ", TimerRegion::CALC_HERMITE_COEFFICIENTS);
    print_timer("      Empty remote nodes cache                 : ", TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    print_timer("      Create synapses                          : ", TimerRegion::CREATE_SYNAPSES);
    print_timer("        Create synapses Exchange Requests      : ", TimerRegion::CREATE_SYNAPSES_EXCHANGE_REQUESTS);
    print_timer("        Create synapses Process Requests       : ", TimerRegion::CREATE_SYNAPSES_PROCESS_REQUESTS);
    print_timer("        Create synapses Exchange Responses     : ", TimerRegion::CREATE_SYNAPSES_EXCHANGE_RESPONSES);
    print_timer("        Create synapses Process Responses      : ", TimerRegion::CREATE_SYNAPSES_PROCESS_RESPONSES);
    print_timer("      Add synapses in local network graphs     : ", TimerRegion::ADD_SYNAPSES_TO_NETWORKGRAPH);

    sstring << "\n\n";

    LogFiles::write_to_file(LogFiles::EventType::Timers, true, sstring.str());

    const auto average_simulation_time = timers_global[3 * static_cast<size_t>(TimerRegion::SIMULATION_LOOP) + 1];
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Simulation time [sec]: {}", average_simulation_time);
}
