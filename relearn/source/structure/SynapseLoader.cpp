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

#include "../util/RelearnException.h"
#include "Partition.h"

#include <fstream>
#include <set>
#include <sstream>
#include <string>

std::pair<FileSynapseLoader::synapses_tuple_type, std::vector<FileSynapseLoader::neuron_id>>
FileSynapseLoader::load_synapses(const std::vector<neuron_id>& affected_neuron_ids) {

    local_synapses_type local_synapses{};
    in_synapses_type in_synapses{};
    out_synapses_type out_synapses{};

    enum class f_status : char {
        not_known = 0,
        local = 1,
        not_local = 2,
    };

    std::string line{};

    std::vector<f_status> id_is_local(partition->get_total_num_neurons(), f_status::not_known);

    std::ifstream file_synapses(path_to_file, std::ios::binary | std::ios::in);

    std::set<neuron_id> foreign_ids{};

    while (std::getline(file_synapses, line)) {
        // Skip line with comments
        if (!line.empty() && '#' == line[0]) {
            continue;
        }

        neuron_id source_id = 0;
        neuron_id target_id = 0;
        synapse_weight weight = 0;

        std::stringstream sstream(line);
        const bool success = (sstream >> source_id) && (sstream >> target_id) && (sstream >> weight);

        RelearnException::check(success, "NetworkGraph::load_synapses: Loading synapses was unsuccessfull!");

        // The neurons start with 1
        source_id--;
        target_id--;

        const f_status source_f = id_is_local[source_id];
        const f_status target_f = id_is_local[target_id];

        bool source_is_local = false;
        bool target_is_local = false;

        if (source_f == f_status::local) {
            source_is_local = true;
        } else if (source_f == f_status::not_local) {
            source_is_local = false;
        } else {
            source_is_local = partition->is_neuron_local(source_id);
            if (source_is_local) {
                id_is_local[source_id] = f_status::local;
            } else {
                id_is_local[source_id] = f_status::not_local;
            }
        }

        if (target_f == f_status::local) {
            target_is_local = true;
        } else if (target_f == f_status::not_local) {
            target_is_local = false;
        } else {
            target_is_local = partition->is_neuron_local(target_id);
            if (target_is_local) {
                id_is_local[target_id] = f_status::local;
            } else {
                id_is_local[target_id] = f_status::not_local;
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

        RelearnException::fail("NetworkGraph::load_synapses: In loading synapses, target and source are not conform.");
    }

    std::vector<neuron_id> global_ids{};
    global_ids.reserve(foreign_ids.size());

    for (const auto& foreign_id : foreign_ids) {
        global_ids.emplace_back(foreign_id);
    }

    auto return_synapses = std::make_tuple(std::move(local_synapses), std::move(in_synapses), std::move(out_synapses));
    auto return_value = std::make_pair(std::move(return_synapses), std::move(global_ids));

    return return_value;
}
