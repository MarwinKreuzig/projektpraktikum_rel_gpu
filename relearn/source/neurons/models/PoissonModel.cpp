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

#include "../../util/Random.h"
#include "../../util/Timers.h"

using models::PoissonModel;

PoissonModel::PoissonModel(double k, double tau_C, double beta, unsigned int h, double background_activity, double background_activity_mean, double background_activity_stddev, const double x_0, const double tau_x, unsigned int refrac_time)
    : NeuronModels{ k, tau_C, beta, h, background_activity, background_activity_mean, background_activity_stddev }
    , x_0{ x_0 }
    , tau_x{ tau_x }
    , refrac_time{ refrac_time } {
}

[[nodiscard]] std::unique_ptr<NeuronModels> PoissonModel::clone() const {
    return std::make_unique<PoissonModel>(get_k(), get_tau_C(), get_beta(), get_h(), get_base_background_activity(), get_background_activity_mean(), get_background_activity_stddev(), x_0, tau_x, refrac_time);
}

[[nodiscard]] double PoissonModel::get_secondary_variable(const size_t i) const noexcept {
    return refrac[i];
}

[[nodiscard]] std::vector<ModelParameter> PoissonModel::get_parameter() {
    auto res{ NeuronModels::get_parameter() };
    res.emplace_back(Parameter<double>{ "x_0", x_0, PoissonModel::min_x_0, PoissonModel::max_x_0 });
    res.emplace_back(Parameter<double>{ "tau_x", tau_x, PoissonModel::min_tau_x, PoissonModel::max_tau_x });
    res.emplace_back(Parameter<unsigned int>{ "refrac_time", refrac_time, PoissonModel::min_refrac_time, PoissonModel::max_refrac_time });
    return res;
}

[[nodiscard]] std::string PoissonModel::name() {
    return "PoissonModel";
}

void PoissonModel::init(size_t num_neurons) {
    NeuronModels::init(num_neurons);
    refrac.resize(num_neurons, 0);
    theta_values.resize(num_neurons, 0.0);
    init_neurons();
}

void PoissonModel::update_activity(const size_t i) {
    const auto h = get_h();
    const auto I_syn = get_I_syn(i);
    auto x = get_x(i);

    for (unsigned int integration_steps = 0; integration_steps < h; integration_steps++) {
        // Update the membrane potential
        x += iter_x(x, I_syn) / h;
    }

    // Neuron ready to fire again
    if (refrac[i] == 0) {
        const bool f = x >= theta_values[i];
        set_fired(i, f); // Decide whether a neuron fires depending on its firing rate
        refrac[i] = f ? refrac_time : 0; // After having fired, a neuron is in a refractory state
    }
    // Neuron now/still in refractory state
    else {
        set_fired(i, false); // Set neuron inactive
        --refrac[i]; // Decrease refractory time
    }

    set_x(i, x);
}

void PoissonModel::init_neurons() {
    const auto num_neurons = get_num_neurons();
    for (size_t i = 0; i < num_neurons; ++i) {
        const auto x = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, 0.0, 1.0);
        const double threshold = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, 0.0, 1.0);
        const bool f = x >= threshold;
        set_fired(i, true);
        set_fired(i, f); // Decide whether a neuron fires depending on its firing rate
        refrac[i] = f ? refrac_time : 0; // After having fired, a neuron is in a refractory state

        set_x(i, x);
    }
}

void models::PoissonModel::update_electrical_activity_serial_initialize(const std::vector<char>& disable_flags) {
    GlobalTimers::timers.start(TimerRegion::CALC_SERIAL_ACTIVITY);

#pragma omp parallel for shared(disable_flags) default(none) // NOLINTNEXTLINE
    for (int neuron_id = 0; neuron_id < theta_values.size(); neuron_id++) {
        if (disable_flags[neuron_id] == 0) {
            continue;
        }

        const double threshold = RandomHolder::get_random_uniform_double(RandomHolderKey::PoissonModel, 0.0, 1.0);
        theta_values[neuron_id] = threshold;
    }

    GlobalTimers::timers.stop_and_add(TimerRegion::CALC_SERIAL_ACTIVITY);
}

[[nodiscard]] double PoissonModel::iter_x(const double x, const double I_syn) const noexcept {
    return ((x_0 - x) / tau_x + I_syn);
}
