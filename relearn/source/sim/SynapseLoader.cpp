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

#include "Types.h"
#include "io/LogFiles.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <fstream>
#include <set>
#include <sstream>
#include <string>

std::pair<std::tuple<LocalSynapses, DistantInSynapses, DistantOutSynapses>,std::tuple<LocalSynapses, DistantInSynapses, DistantOutSynapses>> SynapseLoader::load_synapses() {
    Timers::start(TimerRegion::LOAD_SYNAPSES);
    const auto& synapses_pair = internal_load_synapses();
    const auto& [synapses_static, synapses_plastic] = synapses_pair;
    const auto& [local_synapses, in_synapses, out_synapses] = synapses_plastic;
    Timers::stop_and_add(TimerRegion::LOAD_SYNAPSES);

    RelearnTypes::synapse_weight total_local_weight = 0;
    for (const auto& [_1, _2, weight] : local_synapses) {
        total_local_weight += weight;
    }

    RelearnTypes::synapse_weight total_in_weight = 0;
    for (const auto& [_1, _2, weight] : in_synapses) {
        total_in_weight += weight;
    }

    RelearnTypes::synapse_weight total_out_weight = 0;
    for (const auto& [_1, _2, weight] : out_synapses) {
        total_out_weight += weight;
    }

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Loaded local synapses: {}", local_synapses.size());
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "The local synapses had a weight of: {}", total_local_weight);
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Loaded in synapses: {}", in_synapses.size());
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "The in synapses had a weight of: {}", total_in_weight);
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "Loaded out synapses: {}", out_synapses.size());
    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, "The out synapses had a weight of: {}", total_out_weight);

    return synapses_pair;
}
