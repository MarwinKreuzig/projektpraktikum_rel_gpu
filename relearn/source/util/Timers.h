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

#include "RelearnException.h"

#include <chrono>
#include <iosfwd>
#include <string>
#include <vector>

/**
 * This type allows type-safe specification of a specific timer
 */
enum class TimerRegion : unsigned int {
    INITIALIZATION = 0,
    LOAD_SYNAPSES,
    TRANSLATE_GLOBAL_IDS,
    INITIALIZE_NETWORK_GRAPH,

    SIMULATION_LOOP,
    UPDATE_ELECTRICAL_ACTIVITY,
    PREPARE_SENDING_SPIKES,
    EXCHANGE_NEURON_IDS,
    CALC_SERIAL_ACTIVITY,
    CALC_SYNAPTIC_BACKGROUND,
    CALC_SYNAPTIC_INPUT,
    CALC_ACTIVITY,

    UPDATE_CALCIUM,
    UPDATE_TARGET_CALCIUM,

    UPDATE_SYNAPTIC_ELEMENTS_DELTA,

    UPDATE_CONNECTIVITY,

    UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES,
    COMMIT_NUM_SYNAPTIC_ELEMENTS,
    FIND_SYNAPSES_TO_DELETE,
    DELETE_SYNAPSES_ALL_TO_ALL,
    PROCESS_DELETE_REQUESTS,

    UPDATE_LEAF_NODES,
    UPDATE_LOCAL_TREES,
    EXCHANGE_BRANCH_NODES,
    INSERT_BRANCH_NODES_INTO_GLOBAL_TREE,
    UPDATE_GLOBAL_TREE,

    FIND_TARGET_NEURONS,
    CALC_TAYLOR_COEFFICIENTS,
    CALC_HERMITE_COEFFICIENTS,

    EMPTY_REMOTE_NODES_CACHE,

    CREATE_SYNAPSES,
    CREATE_SYNAPSES_EXCHANGE_REQUESTS,
    CREATE_SYNAPSES_PROCESS_REQUESTS,
    CREATE_SYNAPSES_EXCHANGE_RESPONSES,
    CREATE_SYNAPSES_PROCESS_RESPONSES,

    ADD_SYNAPSES_TO_NETWORKGRAPH,
};

/**
 * This number is used as a shortcut to count the number of values valid for TimerRegion
 */
constexpr size_t NUMBER_TIMERS = 36;

/**
 * This class is used to collect all sorts of different timers (see TimerRegion).
 * It provides an interface to start, stop, and print the timers
 */
class Timers {
    using time_point = std::chrono::high_resolution_clock::time_point;

public:
    /**
     * @brief Starts the respective timer
     * @param timer The timer to start
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUMBER_TIMERS
     */
    static void start(const TimerRegion timer) {
        const auto timer_id = static_cast<size_t>(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::start: timer_id was {}", timer_id);
        time_start[timer_id] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Stops the respective timer
     * @param timer The timer to stops
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUMBER_TIMERS
     */
    static void stop(const TimerRegion timer) {
        const auto timer_id = static_cast<size_t>(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::stop: timer_id was: {}", timer_id);
        time_stop[timer_id] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Stops the respective timer and adds the elapsed time
     * @param timer The timer to stops
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUMBER_TIMERS
     */
    static void stop_and_add(const TimerRegion timer) {
        stop(timer);
        add_start_stop_diff_to_elapsed(timer);
    }

    /**
     * @brief Adds the difference between the current start and stop time points to the elapsed time
     * @param timer The timer for which to add the difference
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUMBER_TIMERS
     */
    static void add_start_stop_diff_to_elapsed(const TimerRegion timer) {
        const auto timer_id = static_cast<size_t>(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::add_start_stop_diff_to_elapsed: timer_id was: {}", timer_id);
        time_elapsed[timer_id] += (time_stop[timer_id] - time_start[timer_id]);
    }

    /**
     * @brief Resets the elapsed time for the timer
     * @param timer The timer for which to reset the elapsed time
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUMBER_TIMERS
     */
    static void reset_elapsed(const TimerRegion timer) {
        const auto timer_id = static_cast<size_t>(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::reset_elapsed: timer_id was: {}", timer_id);
        time_elapsed[timer_id] = std::chrono::nanoseconds(0);
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
    [[nodiscard]] static std::string wall_clock_time();

private:
    /**
     * @brief Returns the elapsed time for the respecive timer
     * @param timer The timer for which to return the elapsed time
     * @exception Throws a RelearnException if the timer casts to a size_t that is >= NUMBER_TIMERS
     * @return The elapsed time
     */
    [[nodiscard]] static std::chrono::nanoseconds get_elapsed(const TimerRegion timer) {
        const auto timer_id = static_cast<size_t>(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::get_elapsed: timer_id was: {}", timer_id);
        return time_elapsed[timer_id];
    }

    // NOLINTNEXTLINE
    static inline std::vector<time_point> time_start{ NUMBER_TIMERS };
    // NOLINTNEXTLINE
    static inline std::vector<time_point> time_stop{ NUMBER_TIMERS };

    // NOLINTNEXTLINE
    static inline std::vector<std::chrono::nanoseconds> time_elapsed{ NUMBER_TIMERS };
};