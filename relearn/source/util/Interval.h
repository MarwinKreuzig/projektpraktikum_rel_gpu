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

#include "Types.h"

#include <vector>

/**
 * This struct represents an interval, i.e., it has a begin step, an end step, and a frequency
 * with which during [begin, end] the function shall be executed, i.e., at steps:
 * begin, begin + frequency, begin + 2*frequency, ...
 */
struct Interval {
    using step_type = RelearnTypes::step_type;

    step_type begin{};
    step_type end{};
    step_type frequency{};

    bool operator==(const Interval& other) const noexcept = default;

    /**
     * @brief Checks if a given step is hit by the interval, i.e., if current_step \in {begin, begin + frequency, begin + 2*frequency, ..., end}
     * @param current_step The current step
     * @return True if the interval hits current_step
     */
    [[nodiscard]] bool hits_step(const step_type current_step) const noexcept {
        if (current_step < begin) {
            return false;
        }

        if (current_step > end) {
            return false;
        }

        const auto relative_offset = current_step - begin;
        return relative_offset % frequency == 0;
    }

    /**
     * @brief Checks if two intervals intersect (ignoring the frequencies)
     * @param first The first interval
     * @param second The second interval
     * @return True iff the intervals intersect
     */
    [[nodiscard]] bool check_for_intersection(const Interval& other) const noexcept {
        return begin <= other.end && other.begin <= end;
    }

    /**
     * @brief Checks if any two intervals intersect
     * @param intervals All intervals
     * @return True iff any two intervals intersect
     */
    [[nodiscard]] static bool check_intervals_for_intersection(const std::vector<Interval>& intervals) noexcept {
        for (auto i = 0; i < intervals.size(); i++) {
            for (auto j = i + 1; j < intervals.size(); j++) {
                const auto intervals_intersect = intervals[i].check_for_intersection(intervals[j]);
                if (intervals_intersect) {
                    return true;
                }
            }
        }

        return false;
    }
};
