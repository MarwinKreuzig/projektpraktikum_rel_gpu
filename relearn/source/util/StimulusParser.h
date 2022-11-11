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
#include "Types.h"
#include "neurons/NeuronsExtraInfo.h"
#include "util/Interval.h"
#include "util/TaggedID.h"
#include "util/Helper.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

struct Stimulus {
    Interval interval{};
    double stimulus_intensity{};
    std::unordered_set<NeuronID::value_type> matching_ids{};
    std::unordered_set<RelearnTypes::area_name> matching_area_names;
};

class StimulusParser {
public:
    using step_type = RelearnTypes::step_type;

    /**
     * @brief Parses one line into a stimulus. Ignores stimuli for other ranks than the current.
     * The line must have the format:
     *      <interval_description> <stimulus intensity> <neuron_id>*
     *      A neuron id must have the format: <rank>:<local_neuron_id> or must be an area name
     * @param line The line to parse
     * @param my_rank The mpi rank of the current process
     * @return Returns an optional Stimulus which is empty if parsing failed
     */
    [[nodiscard]] static std::optional<Stimulus> parse_line(const std::string& line, const int my_rank) {
        std::stringstream ss{ line };

        std::string interval_description{};
        ss >> interval_description;

        if (!ss) {
            return {};
        }

        const auto& parsed_interval = Interval::parse_interval(interval_description);
        if (!parsed_interval.has_value()) {
            return {};
        }

        double intensity{};
        ss >> intensity;

        if (!ss) {
            return {};
        }

        std::unordered_set<NeuronID::value_type> ids{};
        ids.reserve(line.size());
        std::unordered_set<RelearnTypes::area_name> area_names{};
        area_names.reserve(line.size());

        for (std::string current_value{}; ss >> current_value;) {
            const auto& rank_neuron_id_vector = Helper::split_string(current_value, ':');
            if (rank_neuron_id_vector.size() == 2) {
                // Neuron has format <rank>:<neuron_id>
                const int rank = std::stoi(rank_neuron_id_vector[0]);
                if (rank != my_rank) {
                    continue;
                }
                const auto neuron_id = std::stoul(rank_neuron_id_vector[1]);
                ids.insert({ neuron_id });
            } else {
                RelearnException::check(!Helper::is_number(current_value), "StimulusParser::parseLine:: Illegal neuron id {} in stimulus files. Must have the format <rank>:<neuron_id> or be an area name", current_value);
                // Neuron descriptor is an area name
                area_names.insert(current_value);
            }
        }

        return { Stimulus{ parsed_interval.value(), intensity, ids, area_names } };
    }

    /**
     * @brief Parses all lines as stimuli and discards those that fail or are for another mpi rank
     * @param lines The lines to parse
     * @param my_rank The mpi rank of the current propcess
     * @return All successfully parsed stimuli
     */
    [[nodiscard]] static std::vector<Stimulus> parse_lines(const std::vector<std::string>& lines, const int my_rank) {
        std::vector<Stimulus> stimuli{};
        stimuli.reserve(lines.size());

        for (const auto& line : lines) {
            const auto& optional_stimulus = parse_line(line, my_rank);
            if (optional_stimulus.has_value()) {
                stimuli.emplace_back(optional_stimulus.value());
            }
        }

        return stimuli;
    }

    /**
     * @brief Converts the given stimuli to a function that allows easy checking of the current step and neuron id.
     *      If a the combination of step and neuron id hits a stimulus, it returns the intensity. Otherwise, returns 0.0.
     * @param stimuli The given stimuli, should not intersect.
     * @param neuron_id_vs_area_name A vector which assigns each neuron id to an area name
     * @return The check function. Empty if the stimuli intersect
     */
    [[nodiscard]] static std::function<double(step_type, NeuronID::value_type)> generate_stimulus_function(std::vector<Stimulus> stimuli, const std::vector<RelearnTypes::area_name>& neuron_id_vs_area_name) {
        std::vector<Interval> intervals{};
        intervals.reserve(stimuli.size());

        for (const auto& [interval, _1, _2, _3] : stimuli) {
            intervals.emplace_back(interval);
        }

        if (Interval::check_intervals_for_intersection(intervals)) {
            LogFiles::print_message_rank(0, "The intervals for the stimulus parser intersected, discarding all.");
            return {};
        }

        auto comparison = [](const Stimulus& first, const Stimulus& second) -> bool {
            return first.interval.begin < second.interval.begin;
        };

        std::ranges::sort(stimuli, comparison);

        auto step_checker_function = [stimuli = std::move(stimuli), neuron_id_vs_area_name](step_type current_step, NeuronID::value_type neuron_id) noexcept -> double {
            for (const auto& [interval, intensity, ids, areas] : stimuli) {
                if (interval.hits_step(current_step) && (ids.contains(neuron_id) || areas.contains(neuron_id_vs_area_name[neuron_id]))) {
                    return intensity;
                }
            }

            return 0.0;
        };

        return step_checker_function;
    }
};
