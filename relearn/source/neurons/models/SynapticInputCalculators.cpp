/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticInputCalculators.h"

#include "neurons/NetworkGraph.h"
#include "util/Timers.h"

void LinearSynapticInputCalculator::update_synaptic_input(const NetworkGraph& network_graph_static, const NetworkGraph& network_graph_plastic, const std::vector<FiredStatus>& fired, const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_INPUT);

    const auto number_local_neurons = get_number_neurons();

#pragma omp parallel for shared(network_graph_static, network_graph_plastic, disable_flags, number_local_neurons, fired, std::ranges::binary_search) default(none)
    for (auto neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        NeuronID id{ neuron_id };
        
        const auto local_input = get_local_synaptic_input(network_graph_static, fired, id) + get_local_synaptic_input(network_graph_plastic, fired, id);
        const auto distant_input = get_distant_synaptic_input(network_graph_static, fired, id) + get_distant_synaptic_input(network_graph_plastic, fired, id);
        const auto total_input = local_input + distant_input;

        set_synaptic_input(neuron_id, total_input);
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);
}

void LogarithmicSynapticInputCalculator::update_synaptic_input(const NetworkGraph& network_graph_static, const NetworkGraph& network_graph_plastic, const std::vector<FiredStatus>& fired, const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_INPUT);

    const auto number_local_neurons = get_number_neurons();

#pragma omp parallel for shared(network_graph_static, network_graph_plastic, disable_flags, number_local_neurons, fired, std::ranges::binary_search) default(none)
    for (auto neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        NeuronID id{ neuron_id };

        const auto local_input = get_local_synaptic_input(network_graph_static, fired, id) + get_local_synaptic_input(network_graph_plastic, fired, id);
        const auto distant_input = get_distant_synaptic_input(network_graph_static, fired, id) + get_distant_synaptic_input(network_graph_plastic, fired, id);
        const auto total_input = local_input + distant_input;

        // Avoid negative numbers
        const auto shifted_input = total_input + 1;
        const auto logarithmic_input = std::log(shifted_input);

        set_synaptic_input(neuron_id, logarithmic_input);
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);
}
