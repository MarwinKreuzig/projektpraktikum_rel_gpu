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

using models::IzhikevichModel;

IzhikevichModel::IzhikevichModel(double k, double tau_C, double beta, unsigned int h, double background_activity, double background_activity_mean, double background_activity_stddev, const double a, const double b, const double c, const double d, const double V_spike, const double k1, const double k2, const double k3)
    : NeuronModels{ k, tau_C, beta, h, background_activity, background_activity_mean, background_activity_stddev }
    , a{ a }
    , b{ b }
    , c{ c }
    , d{ d }
    , V_spike{ V_spike }
    , k1{ k1 }
    , k2{ k2 }
    , k3{ k3 } {
}

[[nodiscard]] std::unique_ptr<NeuronModels> IzhikevichModel::clone() const {
    return std::make_unique<IzhikevichModel>(get_k(), get_tau_C(), get_beta(), get_h(), get_base_background_activity(), get_background_activity_mean(), get_background_activity_stddev(), a, b, c, d, V_spike, k1, k2, k3);
}

[[nodiscard]] double IzhikevichModel::get_secondary_variable(const size_t i) const noexcept {
    return u[i];
}

[[nodiscard]] std::vector<ModelParameter> IzhikevichModel::get_parameter() {
    auto res{ NeuronModels::get_parameter() };
    res.emplace_back(Parameter<double>{ "a", a, IzhikevichModel::min_a, IzhikevichModel::max_a });
    res.emplace_back(Parameter<double>{ "b", b, IzhikevichModel::min_b, IzhikevichModel::max_b });
    res.emplace_back(Parameter<double>{ "c", c, IzhikevichModel::min_c, IzhikevichModel::max_c });
    res.emplace_back(Parameter<double>{ "d", d, IzhikevichModel::min_d, IzhikevichModel::max_d });
    res.emplace_back(Parameter<double>{ "V_spike", V_spike, IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike });
    res.emplace_back(Parameter<double>{ "k1", k1, IzhikevichModel::min_k1, IzhikevichModel::max_k1 });
    res.emplace_back(Parameter<double>{ "k2", k2, IzhikevichModel::min_k2, IzhikevichModel::max_k2 });
    res.emplace_back(Parameter<double>{ "k3", k3, IzhikevichModel::min_k3, IzhikevichModel::max_k3 });
    return res;
}

[[nodiscard]] std::string IzhikevichModel::name() {
    return "IzhikevichModel";
}

void IzhikevichModel::init(size_t num_neurons) {
    NeuronModels::init(num_neurons);
    u.resize(num_neurons);
    init_neurons(0, num_neurons);
}

void models::IzhikevichModel::create_neurons(size_t creation_count) {
    const auto old_size = NeuronModels::get_num_neurons();
    NeuronModels::create_neurons(creation_count);
    u.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void IzhikevichModel::update_activity(const size_t i) {
    const auto h = get_h();
    const auto I_syn = get_I_syn(i);
    auto x = get_x(i);

    for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
        x += iter_x(x, u[i], I_syn) / h;
        u[i] += iter_refrac(u[i], x) / h;

        if (spiked(x)) {
            set_fired(i, true);
            x = c;
            u[i] += d;
        }
    }

    set_x(i, x);
}

void IzhikevichModel::init_neurons(size_t start_id, size_t end_id) {
    for (size_t i = start_id; i < end_id; ++i) {
        const auto x = c;
        u[i] = iter_refrac(b * c, x);

        set_fired(i, x >= V_spike);
        set_x(i, x);
    }
}

[[nodiscard]] double IzhikevichModel::iter_x(const double x, const double u, const double I_syn) const noexcept {
    return k1 * x * x + k2 * x + k3 - u + I_syn;
}

[[nodiscard]] double IzhikevichModel::iter_refrac(const double u, const double x) const noexcept {
    return a * (b * x - u);
}

[[nodiscard]] bool IzhikevichModel::spiked(const double x) const noexcept {
    return x >= V_spike;
}
