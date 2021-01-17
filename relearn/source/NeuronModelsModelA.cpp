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

using namespace models;

ModelA::ModelA(double k, double tau_C, double beta, unsigned int h, const double x_0, const double tau_x, unsigned int refrac_time)
    : NeuronModels{ k, tau_C, beta, h }
    , x_0{ x_0 }
    , tau_x{ tau_x }
    , refrac_time{ refrac_time } {
}

[[nodiscard]] std::unique_ptr<NeuronModels> ModelA::clone() const {
    return std::make_unique<ModelA>(k, tau_C, beta, h, x_0, tau_x, refrac_time);
}

[[nodiscard]] double ModelA::get_secondary_variable(const size_t i) const noexcept {
    return refrac[i];
}

[[nodiscard]] std::vector<ModelParameter> ModelA::get_parameter() {
    auto res{ NeuronModels::get_parameter() };
    res.reserve(res.size() + 3);
    res.emplace_back(Parameter<double>{ "x_0", x_0, ModelA::min_x_0, ModelA::max_x_0 });
    res.emplace_back(Parameter<double>{ "tau_x", tau_x, ModelA::min_tau_x, ModelA::max_tau_x });
    res.emplace_back(Parameter<unsigned int>{ "refrac_time", refrac_time, ModelA::min_refrac_time, ModelA::max_refrac_time });
    return res;
}

[[nodiscard]] std::string ModelA::name() {
    return "ModelA";
}

void ModelA::init(size_t num_neurons) {
    NeuronModels::init(num_neurons);
    refrac.resize(my_num_neurons, 0);
    init_neurons();
}

void ModelA::update_activity(const size_t i) {
    for (unsigned int integration_steps = 0; integration_steps < h; integration_steps++) {
        // Update the membrane potential
        x[i] += iter_x(x[i], I_syn[i]) / h;
    }

    // Neuron ready to fire again
    if (refrac[i] == 0) {
        const bool f = theta(x[i]);
        fired[i] = f; // Decide whether a neuron fires depending on its firing rate
        refrac[i] = f ? refrac_time : 0; // After having fired, a neuron is in a refractory state
    }
    // Neuron now/still in refractory state
    else {
        fired[i] = false; // Set neuron inactive
        --refrac[i]; // Decrease refractory time
    }
}

void ModelA::init_neurons() {
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = random_number_distribution(random_number_generator);
        const bool f = theta(x[i]);
        fired[i] = f; // Decide whether a neuron fires depending on its firing rate
        refrac[i] = f ? refrac_time : 0; // After having fired, a neuron is in a refractory state
    }
}

[[nodiscard]] double ModelA::iter_x(const double x, const double I_syn) const noexcept {
    return ((x_0 - x) / tau_x + I_syn);
}

[[nodiscard]] bool ModelA::theta(const double x) {
    // 1: fire, 0: inactive
    const double threshold = random_number_distribution(random_number_generator);
    return x >= threshold;
}
