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

#include "Config.h"
#include "RelearnException.h"
#include "io/LogFiles.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * This class parses string that describe at which step a function shall be executed, and returns an std::function which encapsulates this logic.
 * It uses intervals of the form [begin, end] (in predetermined step sizes) to check
 */
class StepParser {
protected:
    /**
     * This struct represents an interval, i.e., it has a begin step, an end step, and a frequency
     * with which during [begin, end] the function shall be executed, i.e., at steps:
     * begin, begin + frequency, begin + 2*frequency, ...
     */
    struct Interval {
        using step_type = std::uint64_t;

        step_type begin{};
        step_type end{};
        step_type frequency{};

        bool operator==(const Interval& other) const noexcept = default;
    };

    /**
     * @brief Checks if two intervals intersect (ignoring the frequences)
     * @param first The first interval
     * @param second The second interval
     * @return True iff the intervals intersect
     */
    [[nodiscard]] static bool check_intervals_for_intersection(const Interval& first, const Interval& second) noexcept {
        return first.begin <= second.end && second.begin <= first.end;
    }

    /**
     * @brief Checks if any two intervals intersect
     * @param intervals All intervals
     * @return True iff any two intervals intersect
     */
    [[nodiscard]] static bool check_intervals_for_intersection(const std::vector<Interval>& intervals) noexcept {
        for (auto i = 0; i < intervals.size(); i++) {
            for (auto j = i + 1; j < intervals.size(); j++) {
                const auto intervals_intersect = check_intervals_for_intersection(intervals[i], intervals[j]);
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

        std::uint64_t begin{};
        const auto& [begin_ptr, begin_err] = std::from_chars(begin_string.data(), begin_string.data() + begin_string.size(), begin);

        std::uint64_t end{};
        const auto& [end_ptr, end_err] = std::from_chars(end_string.data(), end_string.data() + end_string.size(), end);

        std::uint64_t frequency{};
        const auto& [frequency_ptr, frequency_err] = std::from_chars(frequency_string.data(), frequency_string.data() + frequency_string.size(), frequency);

        const auto begin_ok = (begin_err == std::errc{}) && (begin_ptr == begin_string.data() + begin_string.size());
        const auto end_ok = (end_err == std::errc{}) && (end_ptr == end_string.data() + end_string.size());
        const auto frequency_ok = (frequency_err == std::errc{}) && (frequency_ptr == frequency_string.data() + frequency_string.size());

        if (begin_ok && end_ok && frequency_ok) {
            if (end < begin) {
                LogFiles::print_message_rank(0, "Parsed interval description has end before beginning : {}", description);
                return {};
            }

            return Interval{ begin, end, frequency };
        }

        LogFiles::print_message_rank(0, "Failed to parse string to match the pattern <uint64>-<uint64>:<uint64> : {}", description);
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

    /**
     * @brief Converts a vector of intervals to a function which maps the current simulation step to
     *      whether or not it is matched by the intervals. If the intervals intersect themselves, the empty std::function is returned
     * @param intervals The intervals that specify if an event shall occur
     * @return A std::function object that maps the current step to true or false, indicating if the event shall occur
     */
    [[nodiscard]] static std::function<bool(std::uint64_t)> generate_step_check_function(std::vector<Interval> intervals) noexcept {
        const auto intervals_intersect = check_intervals_for_intersection(intervals);
        if (intervals_intersect) {
            return {};
        }

        auto comparison = [](const Interval& first, const Interval& second) -> bool {
            return first.begin < second.begin;
        };

        std::ranges::sort(intervals, comparison);

        auto step_check_function = [intervals = std::move(intervals)](std::uint64_t step) noexcept -> bool {
            for (const auto& [begin, end, frequency] : intervals) {
                if (step < begin) {
                    return false;
                }

                if (step > end) {
                    continue;
                }

                const auto relative_offset = step - begin;
                return relative_offset % frequency == 0;
            }

            return false;
        };

        return step_check_function;
    }

public:
    /**
     * @brief Parses a description of intervals to a std::function which returns true whenever the current simulation step
     *      falls into one of the intervals. The format must be: <begin>-<end>:<frequency> with ; separating the intervals
     * @param description The description of the intervals
     * @return The function indicating if the event shall occur
     */
    [[nodiscard]] static std::function<bool(std::uint64_t)> generate_step_check_function(const std::string& description) {
        auto intervals = parse_description_as_intervals(description);
        auto function = generate_step_check_function(std::move(intervals));
        return function;
    }
};
