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

#include <cmath>

using models::AEIFModel;

AEIFModel::AEIFModel(
    const double k, 
    const double tau_C, 
    const double beta, 
    const unsigned int h, 
    const double base_background_activity, 
    const double background_activity_mean, 
    const double background_activity_stddev, 
    const double C, 
    const double g_L,
    const double E_L,
    const double V_T, 
    const double d_T, 
    const double tau_w, 
    const double a,
    const double b, 
    const double V_spike)
    : NeuronModel{ k, tau_C, beta, h, base_background_activity, background_activity_mean, background_activity_stddev }
    , C{ C }
    , g_L{ g_L }
    , E_L{ E_L }
    , V_T{ V_T }
    , d_T{ d_T }
    , tau_w{ tau_w }
    , a{ a }
    , b{ b }
    , V_spike{ V_spike } {
}

[[nodiscard]] std::unique_ptr<NeuronModel> AEIFModel::clone() const {
    return std::make_unique<AEIFModel>(get_k(), get_tau_C(), get_beta(), get_h(), get_base_background_activity(), get_background_activity_mean(), get_background_activity_stddev(), C, g_L, E_L, V_T, d_T, tau_w, a, b, V_spike);
}

[[nodiscard]] std::vector<ModelParameter> AEIFModel::get_parameter() {
    auto res{ NeuronModel::get_parameter() };
    res.emplace_back(Parameter<double>{ "C", C, AEIFModel::min_C, AEIFModel::max_C });
    res.emplace_back(Parameter<double>{ "g_L", g_L, AEIFModel::min_g_L, AEIFModel::max_g_L });
    res.emplace_back(Parameter<double>{ "E_L", E_L, AEIFModel::min_E_L, AEIFModel::max_E_L });
    res.emplace_back(Parameter<double>{ "V_T", V_T, AEIFModel::min_V_T, AEIFModel::max_V_T });
    res.emplace_back(Parameter<double>{ "d_T", d_T, AEIFModel::min_d_T, AEIFModel::max_d_T });
    res.emplace_back(Parameter<double>{ "tau_w", tau_w, AEIFModel::min_tau_w, AEIFModel::max_tau_w });
    res.emplace_back(Parameter<double>{ "a", a, AEIFModel::min_a, AEIFModel::max_a });
    res.emplace_back(Parameter<double>{ "b", b, AEIFModel::min_b, AEIFModel::max_b });
    res.emplace_back(Parameter<double>{ "V_spike", V_spike, AEIFModel::min_V_spike, AEIFModel::max_V_spike });
    return res;
}

[[nodiscard]] std::string AEIFModel::name() {
    return "AEIFModel";
}

void AEIFModel::init(const size_t num_neurons) {
    NeuronModel::init(num_neurons);
    w.resize(num_neurons);
    init_neurons(0, num_neurons);
}

void models::AEIFModel::create_neurons(const size_t creation_count) {
    const auto old_size = NeuronModel::get_num_neurons();
    NeuronModel::create_neurons(creation_count);
    w.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void AEIFModel::update_activity(const size_t neuron_id) {
    const auto h = get_h();
    const auto I_syn = get_I_syn(neuron_id);
    auto x = get_x(neuron_id);

    auto has_spiked = false;

    for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
        x += iter_x(x, w[neuron_id], I_syn) / h;
        w[neuron_id] += iter_refrac(w[neuron_id], x) / h;

        if (x >= V_spike) {
            x = E_L;
            w[neuron_id] += b;
            has_spiked = true;
            break;
        }
    }

    set_fired(neuron_id, has_spiked);
    set_x(neuron_id, x);
}

void AEIFModel::init_neurons(const size_t start_id, const size_t end_id) {
    for (size_t neuron_id = start_id; neuron_id < end_id; ++neuron_id) {
        const auto x = E_L;
        w[neuron_id] = iter_refrac(0, x);

        set_fired(neuron_id, x >= V_spike);
        set_x(neuron_id, x);
    }
}

[[nodiscard]] double AEIFModel::f(const double x) const noexcept {
    const auto linear_part = -g_L * (x - E_L);
    const auto exp_part = g_L * d_T * exp((x - V_T) / d_T);
    return linear_part + exp_part;
}

[[nodiscard]] double AEIFModel::iter_x(const double x, const double w, const double I_syn) const noexcept {
    const auto f_x = f(x);
    return (f(x) - w + I_syn) / C;
}

[[nodiscard]] double AEIFModel::iter_refrac(const double w, const double x) const noexcept {
    return (a * (x - E_L) - w) / tau_w;
}
