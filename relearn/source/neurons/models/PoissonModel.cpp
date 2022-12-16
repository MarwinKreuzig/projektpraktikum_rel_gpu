/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronModels.h"

#include "util/Random.h"
#include "util/Timers.h"

using models::PoissonModel;

PoissonModel::PoissonModel(
    const unsigned int h,
    std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator,
    std::unique_ptr<Stimulus>&& stimulus_calculator,
    const double x_0,
    const double tau_x,
    const unsigned int refrac_time)
    : NeuronModel{ h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator) }
    , x_0{ x_0 }
    , tau_x{ tau_x }
    , refrac_time{ refrac_time } {
}

[[nodiscard]] std::unique_ptr<NeuronModel> PoissonModel::clone() const {
    return std::make_unique<PoissonModel>(get_h(), get_synaptic_input_calculator()->clone(), get_background_activity_calculator()->clone(),
        get_stimulus_calculator()->clone(), x_0, tau_x, refrac_time);
}

[[nodiscard]] std::vector<ModelParameter> PoissonModel::get_parameter() {
    auto res{ NeuronModel::get_parameter() };
    res.emplace_back(Parameter<double>{ "x_0", x_0, PoissonModel::min_x_0, PoissonModel::max_x_0 });
    res.emplace_back(Parameter<double>{ "tau_x", tau_x, PoissonModel::min_tau_x, PoissonModel::max_tau_x });
    res.emplace_back(Parameter<unsigned int>{ "refrac_time", refrac_time, PoissonModel::min_refrac_time, PoissonModel::max_refrac_time });
    return res;
}

[[nodiscard]] std::string PoissonModel::name() {
    return "PoissonModel";
}

void PoissonModel::init(const number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);
    refrac.resize(number_neurons, 0);
    init_neurons(0, number_neurons);
}

void PoissonModel::create_neurons(const number_neurons_type creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    refrac.resize(old_size + creation_count, 0);
    init_neurons(old_size, creation_count);
}

void PoissonModel::update_activity(const NeuronID& neuron_id) {
    const auto h = get_h();

    const auto synaptic_input = get_synaptic_input(neuron_id);
    const auto background = get_background_activity(neuron_id);
    const auto stimulus = get_stimulus(neuron_id);
    const auto input = synaptic_input + background + stimulus;

    auto x = get_x(neuron_id);

    for (unsigned int integration_steps = 0; integration_steps < h; integration_steps++) {
        // Update the membrane potential
        x += iter_x(x, input) / h;
    }

    const auto local_neuron_id = neuron_id.get_neuron_id();

    // Neuron ready to fire again
    if (refrac[local_neuron_id] == 0) {
        const auto threshold = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, 0.0, 1.0);
        const bool f = x >= threshold;
        if (f) {
            set_fired(neuron_id, FiredStatus::Fired);
            refrac[local_neuron_id] = refrac_time;
        } else {
            set_fired(neuron_id, FiredStatus::Inactive);
        }
    }
    // Neuron now/still in refractory state
    else {
        set_fired(neuron_id, FiredStatus::Inactive); // Set neuron inactive
        --refrac[local_neuron_id]; // Decrease refractory time
    }

    set_x(neuron_id, x);
}

void PoissonModel::update_activity_benchmark(const NeuronID& neuron_id) {
    const auto h = get_h();

    const auto synaptic_input = get_synaptic_input(neuron_id);
    const auto background = get_background_activity(neuron_id);
    const auto input = synaptic_input + background;

    auto x = get_x(neuron_id);
    const auto scale = 1.0 / h;

    for (unsigned int integration_steps = 0; integration_steps < h; integration_steps++) {
        // Update the membrane potential
        x += iter_x(x, input) * scale;
    }

    const auto local_neuron_id = neuron_id.get_neuron_id();

    // Neuron ready to fire again
    if (refrac[local_neuron_id] == 0) {
        const auto threshold = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, 0.0, 1.0);
        const bool f = x >= threshold;
        if (f) {
            set_fired(neuron_id, FiredStatus::Fired);
            refrac[local_neuron_id] = refrac_time;
        } else {
            set_fired(neuron_id, FiredStatus::Inactive);
        }
    }
    // Neuron now/still in refractory state
    else {
        set_fired(neuron_id, FiredStatus::Inactive); // Set neuron inactive
        --refrac[local_neuron_id]; // Decrease refractory time
    }

    set_x(neuron_id, x);
}

void PoissonModel::init_neurons(const number_neurons_type start_id, const number_neurons_type end_id) {
}
