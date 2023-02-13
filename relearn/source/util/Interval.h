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
#include "io/LogFiles.h"

#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>
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

    /**
     * @brief Parses an interval from a description. The description must have the form
     *      <begin>-<end>:<frequency> with <begin> <= <end>
     * @param description The description of the interval
     * @return An optional which is empty if no interval could be parsed, and contains the interval if the parsing succeeded
     */
    [[nodiscard]] static std::optional<Interval> parse_interval(std::string_view description) {
        const auto hyphen_position = description.find('-');
        const auto colon_position = description.find(':');

        if (hyphen_position == std::string::npos || colon_position == std::string::npos) {
            return {};
        }

        const auto& begin_string = description.substr(0, hyphen_position);
        const auto& end_string = description.substr(hyphen_position + 1, colon_position - hyphen_position - 1);
        const auto& frequency_string = description.substr(colon_position + 1, description.size() - colon_position);

        step_type begin{};
        const auto& [begin_ptr, begin_err] = std::from_chars(begin_string.data(), begin_string.data() + begin_string.size(), begin);

        step_type end{};
        const auto& [end_ptr, end_err] = std::from_chars(end_string.data(), end_string.data() + end_string.size(), end);

        step_type frequency{};
        const auto& [frequency_ptr, frequency_err] = std::from_chars(frequency_string.data(), frequency_string.data() + frequency_string.size(), frequency);

        const auto begin_ok = (begin_err == std::errc{}) && (begin_ptr == begin_string.data() + begin_string.size());
        const auto end_ok = (end_err == std::errc{}) && (end_ptr == end_string.data() + end_string.size());
        const auto frequency_ok = (frequency_err == std::errc{}) && (frequency_ptr == frequency_string.data() + frequency_string.size());

        if (begin_ok && end_ok && frequency_ok) {
            if (end < begin) {
                LogFiles::print_message_rank(MPIRank::root_rank(), "Parsed interval description has end before beginning : {}", description);
                return {};
            }

            return Interval{ begin, end, frequency };
        }

        LogFiles::print_message_rank(MPIRank::root_rank(), "Failed to parse string to match the pattern <uint64>-<uint64>:<uint64> : {}", description);
        return {};
    }

    /**
     * @brief Parses multiple intervals from the description. Each interval must have the form
     *      <begin>-<end>:<frequency> with ; separating the intervals
     * @param description The description of the intervals
     * @return A vector with all successfully parsed intervals
     */
    [[nodiscard]] static std::vector<Interval> parse_description_as_intervals(const std::string& description) {
        std::vector<Interval> intervals{};
        std::string::size_type current_position = 0;

        while (true) {
            auto semicolon_position = description.find(';', current_position);
            if (semicolon_position == std::string_view::npos) {
                semicolon_position = description.size();
            }

            std::string_view substring{ description.data() + current_position, description.data() + semicolon_position };
            const auto& opt_interval = parse_interval(substring);

            if (opt_interval.has_value()) {
                intervals.emplace_back(opt_interval.value());
            }

            if (semicolon_position == description.size()) {
                break;
            }

            current_position = semicolon_position + 1;
        }

        return intervals;
    }
};
