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

FitzHughNagumoModel::FitzHughNagumoModel(double k, double tau_C, double beta, unsigned int h, double base_background_activity, double background_activity_mean, double background_activity_stddev, const double a, const double b, const double phi)
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

void FitzHughNagumoModel::init(size_t num_neurons) {
    NeuronModel::init(num_neurons);
    w.resize(num_neurons);
    init_neurons(0, num_neurons);
}

void models::FitzHughNagumoModel::create_neurons(size_t creation_count) {
    const auto old_size = NeuronModel::get_num_neurons();
    NeuronModel::create_neurons(creation_count);
    w.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void FitzHughNagumoModel::update_activity(const size_t neuron_id) {
    const auto h = get_h();
    const auto I_syn = get_I_syn(neuron_id);
    auto x = get_x(neuron_id);

    for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
        x += iter_x(x, w[neuron_id], I_syn) / h;
        w[neuron_id] += iter_refrac(w[neuron_id], x) / h;
    }

    if (FitzHughNagumoModel::spiked(x, w[neuron_id])) {
        set_fired(neuron_id, true);
    } else {
        set_fired(neuron_id, false);
    }

    set_x(neuron_id, x);
}

void FitzHughNagumoModel::init_neurons(size_t start_id, size_t end_id) {
    for (size_t neuron_id = start_id; neuron_id < end_id; ++neuron_id) {
        const auto x = FitzHughNagumoModel::init_x;
        w[neuron_id] = iter_refrac(FitzHughNagumoModel::init_w, x);
        const auto f = spiked(x, w[neuron_id]);

        set_fired(neuron_id, f);
        set_x(neuron_id, x);
    }
}

double FitzHughNagumoModel::iter_x(const double x, const double w, const double I_syn) noexcept {
    return x - x * x * x / 3 - w + I_syn;
}

double FitzHughNagumoModel::iter_refrac(const double w, const double x) const noexcept {
    return phi * (x + a - b * w);
}

bool FitzHughNagumoModel::spiked(const double x, const double w) noexcept {
    return w > iter_x(x, 0, 0) && x > 1.;
}
