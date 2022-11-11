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

#include "neurons/helper/RankNeuronId.h"
#include "util/FileLoader.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"

#include <algorithm>
#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

/**
 * This class provides an interface to parse the neuron ids that shall be monitored from a std::string.
 * It also provides the functionality to sort them and remove duplicates.
 */
class MonitorParser {
public:
    /**
     * @brief Parses a RankNeuronId from a description. Format must be:
     *      <mpi_rank>:<neuron_id>
     *      with a non-negative MPI rank. However, if -1 is parsed as the MPI rank, default_rank is used instead.
     * @param description The description to parse
     * @param default_rank The default MPI rank, must be non-negative
     * @return An optional that contains the parsed RankNeuronId. Is empty if parsing failed or default_rank < 0.
     */
    [[nodiscard]] static std::optional<RankNeuronId> parse_description(const std::string_view description, const int default_rank) {
        if (default_rank < 0) {
            return {};
        }

        const auto colon_position = description.find(':');
        if (colon_position == std::string::npos) {
            return {};
        }

        const auto& mpi_rank_string = description.substr(0, colon_position);
        const auto& neuron_id_string = description.substr(colon_position + 1, description.size() - colon_position);

        int mpi_rank{};
        const auto& [mpi_rank_ptr, mpi_rank_err] = std::from_chars(mpi_rank_string.data(), mpi_rank_string.data() + mpi_rank_string.size(), mpi_rank);

        NeuronID::value_type neuron_id{};
        const auto& [neuron_id_ptr, neuron_id_err] = std::from_chars(neuron_id_string.data(), neuron_id_string.data() + neuron_id_string.size(), neuron_id);

        if (mpi_rank == -1) {
            mpi_rank = default_rank;
        }

        const auto mpi_rank_ok = (mpi_rank_err == std::errc{}) && (mpi_rank_ptr == mpi_rank_string.data() + mpi_rank_string.size()) && mpi_rank >= 0;
        const auto neuron_id_ok = (neuron_id_err == std::errc{}) && (neuron_id_ptr == neuron_id_string.data() + neuron_id_string.size());

        if (mpi_rank_ok && neuron_id_ok) {
            return RankNeuronId{ mpi_rank, NeuronID(neuron_id) };
        }

        LogFiles::print_message_rank(0, "Failed to parse string to match the pattern <mpi_rank>:<neuron_id> : {}", description);
        return {};
    }

    /**
     * @brief Parses multiple RankNeuronIds from a description. Format is:
     *      <mpi_rank>:<neuron_id> with ; separating the RankNeuronIds
     *      with a non-negative MPI rank. However, if -1 is parsed as the MPI rank, default_rank is used instead.
     * @param description The description of the RankNeuronIds
     * @param default_rank The default MPI rank, must be non-negative
     * @exception Throws a RelearnException if default_rank < 0
     * @return A vector with all successfully parsed RankNeuronIds
     */
    [[nodiscard]] static std::vector<RankNeuronId> parse_multiple_description(const std::string& description, const int default_rank) {
        RelearnException::check(default_rank >= 0, "MonitorParser::parse_multiple_description: default_rank {} is < 0.", default_rank);

        std::vector<RankNeuronId> parsed_ids{};
        // The first description is at least 3 chars long, the following at least 4
        parsed_ids.reserve((description.size() >> 2U) + 1U);

        std::string::size_type current_position = 0;

        while (true) {
            auto semicolon_position = description.find(';', current_position);
            if (semicolon_position == std::string_view::npos) {
                semicolon_position = description.size();
            }

            std::string_view substring{ description.data() + current_position, description.data() + semicolon_position };
            const auto& opt_rank_neuron_id = parse_description(substring, default_rank);

            if (opt_rank_neuron_id.has_value()) {
                parsed_ids.emplace_back(opt_rank_neuron_id.value());
            }

            if (semicolon_position == description.size()) {
                break;
            }

            current_position = semicolon_position + 1;
        }

        return parsed_ids;
    }

    [[nodiscard]] static std::vector<RelearnTypes::area_name> parse_area_names(const std::string& description) {
        const auto& vector = FileLoader::split_string(description, ';');
        std::vector<RelearnTypes::area_name> area_names{};
        for (const auto& desc : vector) {
            if (desc.find(':') != std::string::npos || FileLoader::is_number(desc)) {
                // Description has the format of a neuron id. Skip it
                continue;
            }
            area_names.emplace_back(desc);
        }
        return std::move(area_names);
    }

    /**
     * @brief Extracts all NeuronIDs from the RankNeuronIds that belong to the given rank
     * @param rank_neuron_ids The rank neuron ids
     * @param my_rank The current MPI rank, must be non-negative
     * @exception Throws a RelearnException if my_rank < 0
     * @return A vector with all successfully parsed RankNeuronIds
     */
    [[nodiscard]] static std::vector<NeuronID> extract_my_ids(const std::vector<RankNeuronId>& rank_neuron_ids, const int my_rank) {
        RelearnException::check(my_rank >= 0, "MonitorParser::extract_my_ids: my_rank {} is < 0.", my_rank);

        std::vector<NeuronID> my_parsed_ids{};
        my_parsed_ids.reserve(rank_neuron_ids.size());

        for (const auto& [rank, neuron_id] : rank_neuron_ids) {
            if (rank == my_rank) {
                my_parsed_ids.emplace_back(neuron_id);
            }
        }

        return my_parsed_ids;
    }

    /**
     * @brief Removes duplicate NeuronID from the parameter and sorts the result.
     *      Requires that all NeuronIDs are local ids, i.e., neither virtual nor uninitialized.
     * @param neuron_ids NeuronIDs to check for duplicates
     * @exception Throws a RelearnException if a NeuronID was virtual or uninitialized
     * @return The unique and sorted NeuronIDs
     */
    [[nodiscard]] static std::vector<NeuronID> remove_duplicates_and_sort(std::vector<NeuronID> neuron_ids) {
        std::unordered_set<NeuronID> duplicate_checker{};
        duplicate_checker.reserve(neuron_ids.size());

        for (const auto& neuron_id : neuron_ids) {
            duplicate_checker.emplace(neuron_id);
        }

        neuron_ids.assign(duplicate_checker.begin(), duplicate_checker.end());

        auto comparison = [](const NeuronID& first, const NeuronID& second) -> bool {
            return first.get_neuron_id() < second.get_neuron_id();
        };

        std::ranges::sort(neuron_ids, comparison);

        return neuron_ids;
    }

    [[nodiscard]] static std::vector<NeuronID> get_neuron_ids_in_area(const std::vector<RelearnTypes::area_name>& neuron_vs_area, const std::vector<RelearnTypes::area_name>& area_names) {
        std::vector<NeuronID> neurons_in_area{};
        for (const auto& neuron_id : NeuronID::range(0, neuron_vs_area.size())) {
            if (std::find(area_names.begin(), area_names.end(), neuron_vs_area[neuron_id.get_neuron_id()]) != area_names.end()) {
                neurons_in_area.emplace_back(neuron_id);
            }
        }
        return neurons_in_area;
    }

    /**
     * @brief Extracts all to be monitored NeuronIDs that belong to the current rank. Format is:
     *      <mpi_rank>:<neuron_id> with ; separating the RankNeuronIds
     *      with a non-negative MPI rank. However, if -1 is parsed as the MPI rank, default_rank is used instead.
     * @param description The description of the RankNeuronIds
     * @param default_rank The default MPI rank, must be non-negative
     * @param my_rank The current MPI rank, must be non-negative
     * @exception Throws a RelearnException if default_rank < 0 or my_rank < 0
     * @return A vector with all NeuronIDs that shall be monitored at the current rank, sorted and unique
     */
    [[nodiscard]] static std::vector<NeuronID> parse_my_ids(const std::string& description, const int default_rank, const int my_rank, const std::vector<RelearnTypes::area_name>& neuron_vs_area) {
        const auto& rank_neuron_ids = parse_multiple_description(description, default_rank);
        const auto& area_names = parse_area_names(description);
        auto neuron_ids = extract_my_ids(rank_neuron_ids, my_rank);
        auto neurons_in_areas = get_neuron_ids_in_area(neuron_vs_area, area_names);
        neuron_ids.insert(neuron_ids.end(), neurons_in_areas.begin(), neurons_in_areas.end());
        return remove_duplicates_and_sort(std::move(neuron_ids));
    }
};
