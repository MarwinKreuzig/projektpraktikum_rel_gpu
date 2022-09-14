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
    const double k,
    const unsigned int h,
    const double base_background_activity,
    const double background_activity_mean,
    const double background_activity_stddev,
    const double x_0,
    const double tau_x,
    const unsigned int refrac_time)
    : NeuronModel{ k, h, base_background_activity, background_activity_mean, background_activity_stddev }
    , x_0{ x_0 }
    , tau_x{ tau_x }
    , refrac_time{ refrac_time } {
}

[[nodiscard]] std::unique_ptr<NeuronModel> PoissonModel::clone() const {
    return std::make_unique<PoissonModel>(get_k(), get_h(), get_base_background_activity(), get_background_activity_mean(), get_background_activity_stddev(), x_0, tau_x, refrac_time);
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

void PoissonModel::init(const size_t number_neurons) {
    NeuronModel::init(number_neurons);
    refrac.resize(number_neurons, 0);
    theta_values.resize(number_neurons, 0.0);
    init_neurons(0, number_neurons);
}

void PoissonModel::create_neurons(const size_t creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    refrac.resize(old_size + creation_count, 0);
    theta_values.resize(old_size + creation_count, 0.0);
    init_neurons(old_size, creation_count);
}

void PoissonModel::update_activity(const NeuronID& neuron_id) {
    const auto h = get_h();
    
    const auto synaptic_input = get_synaptic_input(neuron_id);
    const auto background = get_background_activity(neuron_id);
    const auto input = synaptic_input + background;

    auto x = get_x(neuron_id);

    for (unsigned int integration_steps = 0; integration_steps < h; integration_steps++) {
        // Update the membrane potential
        x += iter_x(x, input) / h;
    }

    const auto local_neuron_id = neuron_id.get_neuron_id();

    // Neuron ready to fire again
    if (refrac[local_neuron_id] == 0) {
        const bool f = x >= theta_values[local_neuron_id];
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

void PoissonModel::init_neurons(const size_t start_id, const size_t end_id) {

}

void PoissonModel::update_electrical_activity_serial_initialize(const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SERIAL_ACTIVITY);

#pragma omp parallel for shared(disable_flags) default(none) // NOLINTNEXTLINE
    for (int neuron_id = 0; neuron_id < theta_values.size(); neuron_id++) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        const double threshold = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, 0.0, 1.0);
        theta_values[neuron_id] = threshold;
    }

    Timers::stop_and_add(TimerRegion::CALC_SERIAL_ACTIVITY);
}
