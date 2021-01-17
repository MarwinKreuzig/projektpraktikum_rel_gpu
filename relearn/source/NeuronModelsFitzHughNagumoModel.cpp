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

FitzHughNagumoModel::FitzHughNagumoModel(double k, double tau_C, double beta, unsigned int h, const double a, const double b, const double phi)
    : NeuronModels{ k, tau_C, beta, h }
    , a{ a }
    , b{ b }
    , phi{ phi } {
}

std::unique_ptr<NeuronModels> FitzHughNagumoModel::clone() const {
    return std::make_unique<FitzHughNagumoModel>(k, tau_C, beta, h, a, b, phi);
}

double FitzHughNagumoModel::get_secondary_variable(const size_t i) const noexcept {
    return w[i];
}

std::vector<ModelParameter> FitzHughNagumoModel::get_parameter() {
    auto res{ NeuronModels::get_parameter() };
    res.emplace_back(Parameter<double>{ "a", a, FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a });
    res.emplace_back(Parameter<double>{ "b", b, FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b });
    res.emplace_back(Parameter<double>{ "phi", phi, FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi });
    return res;
}

std::string FitzHughNagumoModel::name() {
    return "FitzHughNagumoModel";
}

void FitzHughNagumoModel::init(size_t num_neurons) {
    NeuronModels::init(num_neurons);
    w.resize(num_neurons);
    init_neurons();
}

void FitzHughNagumoModel::update_activity(const size_t i) {
    // Update the membrane potential
    for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
        x[i] += iter_x(x[i], w[i], I_syn[i]) / h;
        w[i] += iter_refrac(w[i], x[i]) / h;

        if (FitzHughNagumoModel::spiked(x[i], w[i])) {
            fired[i] = true;
        }
    }
}

void FitzHughNagumoModel::init_neurons() {
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = FitzHughNagumoModel::init_x;
        w[i] = iter_refrac(FitzHughNagumoModel::init_w, x[i]);
        fired[i] = spiked(x[i], w[i]);
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
