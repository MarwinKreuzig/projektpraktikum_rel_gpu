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

#include "RandomAdapter.h"
#include "tagged_id/tagged_id_adapter.h"

#include "Types.h"
#include "neurons/SignalType.h"
#include "sim/random/SubdomainFromNeuronDensity.h"
#include "structure/Partition.h"

#include <random>
#include <vector>

class NeuronAssignmentAdapter {
public:
    static void generate_random_neurons(std::vector<RelearnTypes::position_type>& positions, std::vector<RelearnTypes::area_id>& neuron_id_to_area_ids, 
        std::vector<RelearnTypes::area_name>& area_id_to_area_name, std::vector<SignalType>& types, std::mt19937& mt) {

        const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
        const auto fraction_excitatory_neurons = RandomAdapter::get_random_percentage<double>(mt);
        const auto um_per_neuron = RandomAdapter::get_random_percentage<double>(mt) * 100.0;

        const auto part = std::make_shared<Partition>(1, 0);
        part->set_total_number_neurons(number_neurons);
        SubdomainFromNeuronDensity sfnd{ number_neurons, fraction_excitatory_neurons, um_per_neuron, part };

        sfnd.initialize();

        positions = sfnd.get_neuron_positions_in_subdomains();
        neuron_id_to_area_ids = sfnd.get_local_area_translator()->get_neuron_ids_to_area_ids();
        area_id_to_area_name = sfnd.get_local_area_translator()->get_all_area_names();
        types = sfnd.get_neuron_types_in_subdomains();

        sfnd.write_neurons_to_file("neurons.tmp");
    }
};
