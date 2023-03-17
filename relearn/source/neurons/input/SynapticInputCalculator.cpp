/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticInputCalculator.h"

#include "mpi/MPIWrapper.h"
#include "neurons/NetworkGraph.h"
#include "neurons/input/FiredStatusCommunicationMap.h"

#include <algorithm>

void SynapticInputCalculator::init(const number_neurons_type number_neurons) {
    RelearnException::check(number_local_neurons == 0, "SynapticInputCalculator::init: Was already initialized");
    RelearnException::check(number_neurons > 0, "SynapticInputCalculator::init: number_neurons was 0");

    number_local_neurons = number_neurons;
    synaptic_input.resize(number_neurons, 0.0);
    raw_ex_input.resize(number_neurons, 0.0);
    raw_inh_input.resize(number_neurons,0.0);

    fired_status_comm->init(number_neurons);
}

void SynapticInputCalculator::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(number_local_neurons > 0, "SynapticInputCalculator::create_neurons: number_local_neurons was 0");
    RelearnException::check(creation_count > 0, "SynapticInputCalculator::create_neurons: creation_count was 0");

    const auto current_size = number_local_neurons;
    const auto new_size = current_size + creation_count;

    number_local_neurons = new_size;
    synaptic_input.resize(new_size, 0.0);
    raw_ex_input.resize(new_size, 0.0);
    raw_inh_input.resize(new_size, 0.0);

    fired_status_comm->create_neurons(creation_count);
}

void SynapticInputCalculator::set_synaptic_input(const double value) noexcept {
    std::ranges::fill(synaptic_input, value);
}

double SynapticInputCalculator::get_local_and_distant_synaptic_input(const NeuronID& neuron_id) {
    auto local_input = 0.0;

    if(transmission_delayer->has_delayed_inputs()) {
        for (const auto &[src_neuron_id, edge_val]: transmission_delayer->get_delayed_inputs(neuron_id)) {
            local_input += synapse_conductance * edge_val;
            if(edge_val > 0.0) {
                raw_ex_input[neuron_id.get_neuron_id()] += edge_val;
            }
            else {
                raw_inh_input[neuron_id.get_neuron_id()] += std::abs(edge_val);
            }
        }
    }

    return local_input;
}
