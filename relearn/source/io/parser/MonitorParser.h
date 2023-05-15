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
#include "io/parser/NeuronIdParser.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/LocalAreaTranslator.h"
#include "util/StringUtil.h"
#include "util/RelearnException.h"
#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"

#include <range/v3/view/cache1.hpp>
#include <string>
#include <string_view>
#include <vector>

#include <range/v3/action/insert.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/cache1.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

/**
 * This class provides an interface to parse neuron ids with descriptions that can contain area names.
 */
class MonitorParser {
public:
    /**
     * @brief Parses a descriptor string for the neuron monitors. If it contains an area name (a string without ':' and not only containing digits),
     *      uses this to get the associated neuron ids from the local_area_translator (discards those that are not present)
     * @param description The string that will be parsed
     * @param local_area_translator Translates between the local area id on the current mpi rank and its area name
     * @return List of area ids found in the string
     */
    [[nodiscard]] static std::vector<RelearnTypes::area_id> parse_area_names(const std::string_view description,
        const std::shared_ptr<LocalAreaTranslator>& local_area_translator) {
        const auto& vector = StringUtil::split_string(std::string(description), ';');
        const auto& known_area_names = local_area_translator->get_all_area_names();

        return vector | ranges::views::filter([](const auto& desc) {
            return !(desc.find(':') != std::string::npos || StringUtil::is_number(desc));
        }) | ranges::views::transform([&known_area_names](const auto& parsed_area_name) {
            return ranges::find(known_area_names, parsed_area_name);
        }) | ranges::views::cache1
            | ranges::views::filter(not_equal_to(known_area_names.end())) | ranges::views::transform([&known_area_names](const auto& parsed_area_name_iter) -> RelearnTypes::area_id {
                  return ranges::distance(known_area_names.begin(),
                      parsed_area_name_iter);
              })
            | ranges::to_vector;
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
    [[nodiscard]] static std::vector<NeuronID> parse_my_ids(const std::string_view description, const MPIRank my_rank,
        const std::shared_ptr<LocalAreaTranslator>& local_area_translator) {
        const auto& rank_neuron_ids = NeuronIdParser::parse_multiple_description(description, my_rank);
        auto neuron_ids = NeuronIdParser::extract_my_ids(rank_neuron_ids, my_rank);

        const auto& area_ids = parse_area_names(description, local_area_translator);
        const auto& neurons_in_areas = local_area_translator->get_neuron_ids_in_areas(area_ids);

        ranges::insert(neuron_ids, neuron_ids.end(), neurons_in_areas);
        return NeuronIdParser::remove_duplicates_and_sort(std::move(neuron_ids));
    }
};
