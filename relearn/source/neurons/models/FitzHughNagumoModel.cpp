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

using models::FitzHughNagumoModel;

FitzHughNagumoModel::FitzHughNagumoModel(
    const double k,
    const double tau_C,
    const double beta,
    const unsigned int h,
    const double base_background_activity,
    const double background_activity_mean,
    const double background_activity_stddev,
    const double a,
    const double b,
    const double phi)
    : NeuronModel{ k, tau_C, beta, h, base_background_activity, background_activity_mean, background_activity_stddev }
    , a{ a }
    , b{ b }
    , phi{ phi } {
}

std::unique_ptr<NeuronModel> FitzHughNagumoModel::clone() const {
    return std::make_unique<FitzHughNagumoModel>(get_k(), get_tau_C(), get_beta(), get_h(), get_base_background_activity(), get_background_activity_mean(), get_background_activity_stddev(), a, b, phi);
}

std::vector<ModelParameter> FitzHughNagumoModel::get_parameter() {
    auto res{ NeuronModel::get_parameter() };
    res.emplace_back(Parameter<double>{ "a", a, FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a });
    res.emplace_back(Parameter<double>{ "b", b, FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b });
    res.emplace_back(Parameter<double>{ "phi", phi, FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi });
    return res;
}

std::string FitzHughNagumoModel::name() {
    return "FitzHughNagumoModel";
}

void FitzHughNagumoModel::init(size_t number_neurons) {
    NeuronModel::init(number_neurons);
    w.resize(number_neurons);
    init_neurons(0, number_neurons);
}

void FitzHughNagumoModel::create_neurons(size_t creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    w.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void FitzHughNagumoModel::update_activity(const NeuronID& neuron_id) {
    const auto h = get_h();

    const auto synaptic_input = get_synaptic_input(neuron_id);
    const auto background = get_background_activity(neuron_id);
    const auto input = synaptic_input + background;

    auto x = get_x(neuron_id);

    const auto local_neuron_id = neuron_id.get_local_id();

    for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
        x += iter_x(x, w[local_neuron_id], input) / h;
        w[local_neuron_id] += iter_refrac(w[local_neuron_id], x) / h;
    }

    if (FitzHughNagumoModel::spiked(x, w[local_neuron_id])) {
        set_fired(neuron_id, FiredStatus::Fired);
    } else {
        set_fired(neuron_id, FiredStatus::Inactive);
    }

    set_x(neuron_id, x);
}

void FitzHughNagumoModel::init_neurons(const size_t start_id, const size_t end_id) {
    for (size_t neuron_id = start_id; neuron_id < end_id; ++neuron_id) {
        const auto id = NeuronID{ neuron_id };
        w[neuron_id] = FitzHughNagumoModel::init_w;
        set_x(id, FitzHughNagumoModel::init_x);
    }
}

double FitzHughNagumoModel::iter_x(const double x, const double w, const double input) noexcept {
    return x - x * x * x / 3 - w + input;
}

double FitzHughNagumoModel::iter_refrac(const double w, const double x) const noexcept {
    return phi * (x + a - b * w);
}

bool FitzHughNagumoModel::spiked(const double x, const double w) noexcept {
    return w > iter_x(x, 0, 0) && x > 1.;
}
