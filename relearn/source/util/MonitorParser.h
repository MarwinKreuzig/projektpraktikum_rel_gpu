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
#include "neurons/LocalAreaTranslator.h"
#include "util/StringUtil.h"
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
     *      with a non-negative MPI rank. However, if -1 is parsed as the MPI rank, my_rank is used instead.
     *      <neuron_id> is in input format, i.e., "+1".
     * @param description The description to parse
     * @param my_rank The default MPI rank
     * @exception Throws a RelearnException if a <neuron_id> is 0
     * @return An optional that contains the parsed RankNeuronId. Is empty if parsing failed or my_rank is not initialized
     */
    [[nodiscard]] static std::optional<RankNeuronId> parse_description(const std::string_view description, const MPIRank my_rank) {
        if (!my_rank.is_initialized()) {
            return {};
        }

        const auto colon_position = description.find(':');
        if (colon_position == std::string::npos) {
            return {};
        }

        const auto& mpi_rank_string = description.substr(0, colon_position);
        const auto& neuron_id_string = description.substr(colon_position + 1, description.size() - colon_position);

        int parsed_mpi_rank{};
        const auto& [mpi_rank_ptr, mpi_rank_err] = std::from_chars(mpi_rank_string.data(), mpi_rank_string.data() + mpi_rank_string.size(), parsed_mpi_rank);

        if (parsed_mpi_rank == -1) {
            parsed_mpi_rank = my_rank.get_rank();
        }

        NeuronID::value_type neuron_id{};
        const auto& [neuron_id_ptr, neuron_id_err] = std::from_chars(neuron_id_string.data(), neuron_id_string.data() + neuron_id_string.size(), neuron_id);

        const auto mpi_rank_ok = (mpi_rank_err == std::errc{}) && (mpi_rank_ptr == mpi_rank_string.data() + mpi_rank_string.size()) && parsed_mpi_rank >= 0;
        const auto neuron_id_ok = (neuron_id_err == std::errc{}) && (neuron_id_ptr == neuron_id_string.data() + neuron_id_string.size());

        if (mpi_rank_ok && neuron_id_ok) {
            // Check here so we can use the previous error codes correctly
            RelearnException::check(neuron_id > 0, "MonitorParser::parse_description: A parsed NeuronID is 0, but the input is 1-based: {}", description);

            return RankNeuronId{ MPIRank(parsed_mpi_rank), NeuronID(neuron_id - 1) };
        }

        LogFiles::print_message_rank(0, "Failed to parse string to match the pattern <mpi_rank>:<neuron_id> : {}", description);
        return {};
    }

    /**
     * @brief Parses multiple RankNeuronIds from a description. Format is:
     *      <mpi_rank>:<neuron_id> with ; separating the RankNeuronIds
     *      with a non-negative MPI rank. However, if -1 is parsed as the MPI rank, my_rank is used instead.
     *      <neuron_id> is in input format, i.e., "+1".
     * @param description The description of the RankNeuronIds
     * @param my_rank The default MPI rank, must be initialized
     * @exception Throws a RelearnException if my_rank is not initialized or a <neuron_id> is 0
     * @return A vector with all successfully parsed RankNeuronIds
     */
    [[nodiscard]] static std::vector<RankNeuronId> parse_multiple_description(const std::string& description, const MPIRank my_rank) {
        RelearnException::check(my_rank.is_initialized(), "MonitorParser::parse_multiple_description: my_rank is not initialized.", my_rank);

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
            const auto& opt_rank_neuron_id = parse_description(substring, my_rank);

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

    /**
     * @brief Parses a descriptor string for the neuron monitors. If it contains an area name (a string without ':' and not only containing digits),
     *      uses this to get the associated neuron ids from the local_area_translator (discards those that are not present)
     * @param description The string that will be parsed
     * @param local_area_translator Translates between the local area id on the current mpi rank and its area name
     * @return List of area ids found in the string
     */
    [[nodiscard]] static std::vector<RelearnTypes::area_id> parse_area_names(const std::string& description,
        const std::shared_ptr<LocalAreaTranslator>& local_area_translator) {
        const auto& vector = StringUtil::split_string(description, ';');
        std::vector<RelearnTypes::area_name> parsed_area_names{};
        for (auto& desc : vector) {
            if (desc.find(':') != std::string::npos || StringUtil::is_number(desc)) {
                // Description has the format of a neuron id. Skip it
                continue;
            }
            parsed_area_names.emplace_back(std::move(desc));
        }

        std::vector<RelearnTypes::area_id> area_ids{};
        area_ids.reserve(parsed_area_names.size());

        const auto& known_area_names = local_area_translator->get_all_area_names();
        for (const auto& parsed_area_name : parsed_area_names) {
            if (const auto it = std::find(known_area_names.begin(), known_area_names.end(), parsed_area_name); it != known_area_names.end()) {
                area_ids.emplace_back(std::distance(known_area_names.begin(), it));
            }
        }

        return area_ids;
    }

    /**
     * @brief Extracts all NeuronIDs from the RankNeuronIds that belong to the given rank.
     *      The ids in the RankNeuronIds are in input format, i.e., "+1"; this method will subtract one when converting them to NeuronID.
     * @param rank_neuron_ids The rank neuron ids
     * @param my_rank The current MPI rank, must be initialized
     * @exception Throws a RelearnException if my_rank is not initialized
     * @return A vector with all successfully parsed RankNeuronIds
     */
    [[nodiscard]] static std::vector<NeuronID> extract_my_ids(const std::vector<RankNeuronId>& rank_neuron_ids, const MPIRank my_rank) {
        RelearnException::check(my_rank.is_initialized(), "MonitorParser::extract_my_ids: my_rank is not initialized.", my_rank);

        std::vector<NeuronID> my_parsed_ids{};
        my_parsed_ids.reserve(rank_neuron_ids.size());

        for (const auto& [rank, neuron_id] : rank_neuron_ids) {
            if (rank == my_rank) {
                my_parsed_ids.emplace_back(neuron_id.get_neuron_id() - 1);
            }
        }

        return my_parsed_ids;
    }

    /**
     * @brief Removes duplicate NeuronID from the parameter and sorts the result.
     *      Requires that all NeuronIDs are actual ids, i.e., neither virtual nor uninitialized.
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

    /**
     * @brief Extracts all to be monitored NeuronIDs that belong to the current rank. Format is:
     *      <mpi_rank>:<neuron_id> with ; separating the RankNeuronIds
     *      with a non-negative MPI rank. However, if -1 is parsed as the MPI rank, my_rank is used instead.
     *      Alternatively, it can also contain
     *      <area_name>
     *      which then translates to all NeuronIDs within the areas
     * @param description The description of the RankNeuronIds
     * @param my_rank The current MPI rank, must be initialized
     * @param local_area_translator Translates the area names to the associated NeuronIDs
     * @exception Throws a RelearnException if my_rank is not initialized
     * @return A vector with all NeuronIDs that shall be monitored at the current rank, sorted and unique
     */
    [[nodiscard]] static std::vector<NeuronID> parse_my_ids(const std::string& description, const MPIRank my_rank,
        const std::shared_ptr<LocalAreaTranslator>& local_area_translator) {
        const auto& rank_neuron_ids = parse_multiple_description(description, my_rank);
        auto neuron_ids = extract_my_ids(rank_neuron_ids, my_rank);

        const auto& area_ids = parse_area_names(description, local_area_translator);
        const auto& neurons_in_areas = local_area_translator->get_neuron_ids_in_areas(area_ids);

        neuron_ids.insert(neuron_ids.end(), neurons_in_areas.begin(), neurons_in_areas.end());
        return remove_duplicates_and_sort(std::move(neuron_ids));
    }
};
