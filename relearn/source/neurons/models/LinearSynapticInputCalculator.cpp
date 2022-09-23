/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "LinearSynapticInputCalculator.h"

#include "neurons/NetworkGraph.h"
#include "util/Random.h"
#include "util/Timers.h"

void LinearSynapticInputCalculator::update_synaptic_input(const NetworkGraph& network_graph, const std::vector<FiredStatus> fired, const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_INPUT);

    const auto number_local_neurons = get_number_neurons();

#pragma omp parallel for shared(network_graph, disable_flags, std::ranges::binary_search) default(none)
    for (auto neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        NeuronID id{ neuron_id };
        
        const auto local_input = get_local_synaptic_input(network_graph, fired, id);
        const auto distant_input = get_distant_synaptic_input(network_graph, fired, id);
        const auto total_input = local_input + distant_input;

        set_synaptic_input(neuron_id, total_input);
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);
}

void LinearSynapticInputCalculator::update_background_activity(const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);

    const auto background_activity_stddev = get_background_activity_stddev();
    const auto background_activity_mean = get_background_activity_mean();
    const auto base_background_activity = get_base_background_activity();

    const auto number_local_neurons = get_number_neurons();

    // There might be background activity
    if (background_activity_stddev > 0.0) {
        for (size_t neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
            if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
                continue;
            }

            const double rnd = RandomHolder::get_random_normal_double(RandomHolderKey::NeuronModel, background_activity_mean, background_activity_stddev);
            const double input = base_background_activity + rnd;

            set_background_activity(neuron_id, input);
        }
    } else {
        set_background_activity(base_background_activity);
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
}
