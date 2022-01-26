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

#include "../structure/NeuronIdTranslator.h"
#include "../util/RelearnException.h"
#include "../util/Timers.h"
#include "Partition.h"

#include <fstream>
#include <set>
#include <sstream>
#include <string>

std::pair<FileSynapseLoader::synapses_tuple_type, std::vector<NeuronID>>
FileSynapseLoader::internal_load_synapses() {

    if (!optional_path_to_file.has_value()) {
        return {};
    }

    const auto& path_to_file = optional_path_to_file.value();

    local_synapses_type local_synapses{};
    in_synapses_type in_synapses{};
    out_synapses_type out_synapses{};

    enum class f_status : char {
        not_known = 0,
        local = 1,
        not_local = 2,
    };

    std::string line{};

    std::vector<f_status> id_is_local(partition->get_total_number_neurons(), f_status::not_known);

    std::ifstream file_synapses(path_to_file, std::ios::binary | std::ios::in);

    std::set<NeuronID> foreign_ids{};

    while (std::getline(file_synapses, line)) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        size_t read_source_id = 0;
        size_t read_target_id = 0;
        synapse_weight weight = 0;

        std::stringstream sstream(line);
        const bool success = (sstream >> read_source_id) && (sstream >> read_target_id) && (sstream >> weight);

        RelearnException::check(success, "FileSynapseLoader::internal_load_synapses: Loading synapses was unsuccessfull!");

        RelearnException::check(read_source_id > 0, "FileSynapseLoader::internal_load_synapses: source_id was 0");
        RelearnException::check(read_target_id > 0, "FileSynapseLoader::internal_load_synapses: target_id was 0");
        RelearnException::check(weight != 0, "FileSynapseLoader::internal_load_synapses: weight was 0");

        // The neurons start with 1
        --read_source_id;
        --read_target_id;
        auto source_id = NeuronID{ read_source_id };
        auto target_id = NeuronID{ read_target_id };

        const f_status source_f = id_is_local[source_id.id()];
        const f_status target_f = id_is_local[target_id.id()];

        bool source_is_local = false;
        bool target_is_local = false;

        if (source_f == f_status::local) {
            source_is_local = true;
        } else if (source_f == f_status::not_local) {
            source_is_local = false;
        } else {
            source_is_local = nit->is_neuron_local(source_id);
            if (source_is_local) {
                id_is_local[source_id.id()] = f_status::local;
            } else {
                id_is_local[source_id.id()] = f_status::not_local;
            }
        }

        if (target_f == f_status::local) {
            target_is_local = true;
        } else if (target_f == f_status::not_local) {
            target_is_local = false;
        } else {
            target_is_local = nit->is_neuron_local(target_id);
            if (target_is_local) {
                id_is_local[target_id.id()] = f_status::local;
            } else {
                id_is_local[target_id.id()] = f_status::not_local;
            }
        }

        if (!source_is_local && !target_is_local) {
            continue;
        }

        if (source_is_local && target_is_local) {
            local_synapses.emplace_back(source_id, target_id, weight);
            continue;
        }

        if (source_is_local && !target_is_local) {
            out_synapses.emplace_back(source_id, target_id, weight);
            foreign_ids.emplace(target_id);
            continue;
        }

        if (!source_is_local && target_is_local) {
            in_synapses.emplace_back(source_id, target_id, weight);
            foreign_ids.emplace(source_id);
            continue;
        }

        RelearnException::fail("FileSynapseLoader::internal_load_synapses: In loading synapses, target and source are not conform.");
    }

    std::vector<NeuronID> global_ids{};
    global_ids.reserve(foreign_ids.size());

    for (const auto& foreign_id : foreign_ids) {
        global_ids.emplace_back(foreign_id);
    }

    auto return_synapses = std::make_tuple(std::move(local_synapses), std::move(in_synapses), std::move(out_synapses));
    auto return_value = std::make_pair(std::move(return_synapses), std::move(global_ids));

    return return_value;
}

std::tuple<SynapseLoader::LocalSynapses, SynapseLoader::InSynapses, SynapseLoader::OutSynapses> SynapseLoader::load_synapses() {
    Timers::start(TimerRegion::LOAD_SYNAPSES);
    const auto& [synapses, global_ids] = internal_load_synapses();
    const auto& [local_synapses, in_synapses, out_synapses] = synapses;
    Timers::stop_and_add(TimerRegion::LOAD_SYNAPSES);

    LocalSynapses return_local_synapses{};
    return_local_synapses.reserve(local_synapses.size());

    InSynapses return_in_synapses{};
    return_in_synapses.reserve(in_synapses.size());

    OutSynapses return_out_synapses{};
    return_out_synapses.reserve(out_synapses.size());

    Timers::start(TimerRegion::TRANSLATE_GLOBAL_IDS);
    const auto& translated_ids = nit->translate_global_ids(global_ids);
    Timers::stop_and_add(TimerRegion::TRANSLATE_GLOBAL_IDS);

    auto total_local_weight = 0;
    for (const auto& [source_id, target_id, weight] : local_synapses) {
        const auto local_source_id = nit->get_local_id(source_id);
        const auto local_target_id = nit->get_local_id(target_id);

        return_local_synapses.emplace_back(local_source_id, local_target_id, weight);
        total_local_weight += weight;
    }

    auto total_in_weight = 0;
    for (const auto& [source_id, target_id, weight] : in_synapses) {
        const auto local_target_id = nit->get_local_id(target_id);

        return_in_synapses.emplace_back(translated_ids.at(source_id), local_target_id, weight);
        total_in_weight += weight;
    }

    auto total_out_weight = 0;
    for (const auto& [source_id, target_id, weight] : out_synapses) {
        const auto local_source_id = nit->get_local_id(source_id);

        return_out_synapses.emplace_back(local_source_id, translated_ids.at(target_id), weight);
        total_out_weight += weight;
    }

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Loaded {} local synapses, {} in synapses, and {} out synapses", local_synapses.size(), in_synapses.size(), out_synapses.size());
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "They had a total weight of: {}, {}, {}", total_local_weight, total_in_weight, total_out_weight);

    return { return_local_synapses, return_in_synapses, return_out_synapses };
}
