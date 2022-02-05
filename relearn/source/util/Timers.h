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

#include "../util/RelearnException.h"

#include <array>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

/**
 * This type allows type-safe specification of a specific timer
 */
enum class TimerRegion : int {
    INITIALIZATION = 0,
    SIMULATION_LOOP = 1,
    UPDATE_ELECTRICAL_ACTIVITY = 2,
    PREPARE_SENDING_SPIKES = 3,
    PREPARE_NUM_NEURON_IDS = 4,
    ALL_TO_ALL = 5,
    ALLOC_MEM_FOR_NEURON_IDS = 6,
    EXCHANGE_NEURON_IDS = 7,
    CALC_SYNAPTIC_BACKGROUND = 8,
    CALC_SERIAL_ACTIVITY = 9,
    CALC_SYNAPTIC_INPUT = 10,
    CALC_ACTIVITY = 11,
    UPDATE_SYNAPTIC_ELEMENTS_DELTA = 12,
    UPDATE_CONNECTIVITY = 13,
    UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES = 14,
    UPDATE_LOCAL_TREES = 15,
    EXCHANGE_BRANCH_NODES = 16,
    INSERT_BRANCH_NODES_INTO_GLOBAL_TREE = 17,
    UPDATE_GLOBAL_TREE = 18,
    FIND_TARGET_NEURONS = 19,
    EMPTY_REMOTE_NODES_CACHE = 20,
    CREATE_SYNAPSES = 21,
    UPDATE_LEAF_NODES = 22,
    CALC_TAYLOR_COEFFICIENTS = 23,
    CALC_HERMITE_COEFFICIENTS = 24,
    LOAD_SYNAPSES = 25,
    TRANSLATE_GLOBAL_IDS = 26,
    INITIALIZE_NETWORK_GRAPH = 27,
    ADD_SYNAPSES_TO_NETWORKGRAPH = 28,
    DELETE_SYNAPSES_ALL_TO_ALL = 29,
    FIND_SYNAPSES_TO_DELETE = 30,
    PROCESS_DELETE_REQUESTS = 31,
    COMMIT_NUM_SYNAPTIC_ELEMENTS = 32,
};

/**
 * This number is used as a shortcut to count the number of values valid for TimerRegion
 */
constexpr size_t NUM_TIMERS = 33;

/**
 * This class is used to collect all sorts of different timers (see TimerRegion).
 * It provides an interface to start, stop, and print the timers
 */
class Timers {
public:
    /**
     * @brief Starts the respective timer
     * @param timer The timer to start
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUM_TIMERS
     */
    static void start(const TimerRegion timer) {
        const auto timer_id = static_cast<size_t>(timer);
        RelearnException::check(timer_id < NUM_TIMERS, "Timers::start: timer_id was {}", timer_id);
        time_start[timer_id] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Stops the respective timer
     * @param timer The timer to stops
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUM_TIMERS
     */
    static void stop(const TimerRegion timer) {
        const auto timer_id = static_cast<size_t>(timer);
        RelearnException::check(timer_id < NUM_TIMERS, "Timers::stop: timer_id was: {}", timer_id);
        time_stop[timer_id] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Stops the respective timer and adds the elapsed time
     * @param timer The timer to stops
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUM_TIMERS
     */
    static void stop_and_add(const TimerRegion timer) {
        stop(timer);
        add_start_stop_diff_to_elapsed(timer);
    }

    /**
     * @brief Adds the difference between the current start and stop time points to the elapsed time
     * @param timer The timer for which to add the difference
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUM_TIMERS
     */
    static void add_start_stop_diff_to_elapsed(const TimerRegion timer) {
        const auto timer_id = static_cast<size_t>(timer);
        RelearnException::check(timer_id < NUM_TIMERS, "Timers::add_start_stop_diff_to_elapsed: timer_id was: {}", timer_id);
        time_elapsed[timer_id] += std::chrono::duration_cast<std::chrono::duration<double>>(time_stop[timer_id] - time_start[timer_id]);
    }

    /**
     * @brief Resets the elapsed time for the timer
     * @param timer The timer for which to reset the elapsed time
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUM_TIMERS
     */
    static void reset_elapsed(const TimerRegion timer) {
        const auto timer_id = static_cast<size_t>(timer);
        RelearnException::check(timer_id < NUM_TIMERS, "Timers::reset_elapsed: timer_id was: {}", timer_id);
        time_elapsed[timer_id] = std::chrono::duration<double>::zero();
    }

    /**
     * @brief Returns the elapsed time for the respecive timer
     * @param timer The timer for which to return the elapsed time
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUM_TIMERS
     * @return The elapsed time
     */
    [[nodiscard]] static double get_elapsed(const TimerRegion timer) {
        const auto timer_id = static_cast<size_t>(timer);
        RelearnException::check(timer_id < NUM_TIMERS, "Timers::get_elapsed: timer_id was: {}", timer_id);
        return time_elapsed[timer_id].count();
    }

    /**
     * @brief Prints all timers with min, max, and sum across all MPI ranks to LogFiles::EventType::Timers.
     * Performs MPI communication.
     */
    static void print();

    /**
	 * @brief Returns the current time as a string
     * @return The current time as a string
	 */
    [[nodiscard]] static std::string wall_clock_time() {
#ifdef __linux__
        time_t rawtime = 0;
        time(&rawtime);
        struct tm* timeinfo = localtime(&rawtime);
        char* string = asctime(timeinfo);

        // Remove linebreak in string
        // NOLINTNEXTLINE
        string[24] = '\0';

        return std::string(string);
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

private:
    static void print_timer(std::stringstream& sstream, TimerRegion timer_index, const std::array<double, size_t(3) * NUM_TIMERS>& timers);

    // NOLINTNEXTLINE
    static inline std::vector<std::chrono::high_resolution_clock::time_point> time_start{ NUM_TIMERS };
    // NOLINTNEXTLINE
    static inline std::vector<std::chrono::high_resolution_clock::time_point> time_stop{ NUM_TIMERS };

    // NOLINTNEXTLINE
    static inline std::vector<std::chrono::duration<double>> time_elapsed{ NUM_TIMERS };
};