/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapseLoader.h"

#include "NeuronIdTranslator.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <fstream>
#include <set>
#include <sstream>
#include <string>

std::tuple<LocalSynapses, DistantInSynapses, DistantOutSynapses> SynapseLoader::load_synapses() {
    Timers::start(TimerRegion::LOAD_SYNAPSES);
    const auto& [synapses, global_ids] = internal_load_synapses();
    const auto& [local_synapses, in_synapses, out_synapses] = synapses;
    Timers::stop_and_add(TimerRegion::LOAD_SYNAPSES);

    LocalSynapses return_local_synapses{};
    return_local_synapses.reserve(local_synapses.size());

    DistantInSynapses return_in_synapses{};
    return_in_synapses.reserve(in_synapses.size());

    DistantOutSynapses return_out_synapses{};
    return_out_synapses.reserve(out_synapses.size());

    Timers::start(TimerRegion::TRANSLATE_GLOBAL_IDS);
    const auto& translated_ids = nit->translate_global_ids(global_ids);
    Timers::stop_and_add(TimerRegion::TRANSLATE_GLOBAL_IDS);

    auto total_local_weight = 0;
    for (const auto& [source_id, target_id, weight] : local_synapses) {
        const auto local_source_id = nit->get_local_id(source_id);
        const auto local_target_id = nit->get_local_id(target_id);

        return_local_synapses.emplace_back(local_target_id, local_source_id, weight);
        total_local_weight += weight;
    }

    auto total_in_weight = 0;
    for (const auto& [source_id, target_id, weight] : in_synapses) {
        const auto local_target_id = nit->get_local_id(target_id);

        return_in_synapses.emplace_back(local_target_id, translated_ids.at(source_id), weight);
        total_in_weight += weight;
    }

    auto total_out_weight = 0;
    for (const auto& [source_id, target_id, weight] : out_synapses) {
        const auto local_source_id = nit->get_local_id(source_id);

        return_out_synapses.emplace_back(translated_ids.at(target_id), local_source_id, weight);
        total_out_weight += weight;
    }

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Loaded local synapses: {}", local_synapses.size());
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "The local synapses had a weight of: {}", total_local_weight);
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Loaded in synapses: {}", in_synapses.size());
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "The in synapses had a weight of: {}", total_in_weight);
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Loaded out synapses: {}", out_synapses.size());
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "The out synapses had a weight of: {}", total_out_weight);

    return { return_local_synapses, return_in_synapses, return_out_synapses };
}
