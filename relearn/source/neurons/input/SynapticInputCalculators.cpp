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
#include "util/MemoryFootprint.h"
#include "util/NeuronID.h"
#include "util/Timers.h"

#include <cmath>

void LinearSynapticInputCalculator::record_memory_footprint(const std::unique_ptr<MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) - sizeof(SynapticInputCalculator);
    footprint->emplace("LinearSynapticInputCalculator", my_footprint);

    SynapticInputCalculator::record_memory_footprint(footprint);
}

void LinearSynapticInputCalculator::update_synaptic_input(const std::span<const FiredStatus> fired) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_INPUT);

    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_local_neurons = get_number_neurons();

#pragma omp parallel for shared(disable_flags, number_local_neurons, fired) default(none)
    for (auto neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const NeuronID id{ neuron_id };

        const auto total_input = get_local_and_distant_synaptic_input(fired, id);

        set_synaptic_input(neuron_id, total_input);
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);
}

void LogarithmicSynapticInputCalculator::record_memory_footprint(const std::unique_ptr<MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) - sizeof(SynapticInputCalculator);
    footprint->emplace("LogarithmicSynapticInputCalculator", my_footprint);

    SynapticInputCalculator::record_memory_footprint(footprint);
}

void LogarithmicSynapticInputCalculator::update_synaptic_input(const std::span<const FiredStatus> fired) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_INPUT);

    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_local_neurons = get_number_neurons();

#pragma omp parallel for shared(disable_flags, number_local_neurons, fired) default(none)
    for (auto neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const NeuronID id{ neuron_id };

        const auto total_input = get_local_and_distant_synaptic_input(fired, id);

        // Avoid negative numbers
        const auto shifted_input = total_input + 1;
        const auto logarithmic_input = scale_factor * std::log10(shifted_input);

        set_synaptic_input(neuron_id, logarithmic_input);
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);
}

void HyperbolicTangentSynapticInputCalculator::record_memory_footprint(const std::unique_ptr<MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this) - sizeof(SynapticInputCalculator);
    footprint->emplace("HyperbolicTangentSynapticInputCalculator", my_footprint);

    SynapticInputCalculator::record_memory_footprint(footprint);
}

void HyperbolicTangentSynapticInputCalculator::update_synaptic_input(const std::span<const FiredStatus> fired) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_INPUT);

    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_local_neurons = get_number_neurons();

#pragma omp parallel for shared(disable_flags, number_local_neurons, fired) default(none)
    for (auto neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        NeuronID id{ neuron_id };

        const auto total_input = get_local_and_distant_synaptic_input(fired, id);

        const auto hyp_tan_input = std::tanh(total_input);
        const auto scaled_input = hyp_tan_input * scale_factor;

        set_synaptic_input(neuron_id, scaled_input);
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);
}
