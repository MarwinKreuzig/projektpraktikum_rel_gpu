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

IzhikevichModel::IzhikevichModel(
    const double k,
    const double tau_C,
    const double beta,
    const unsigned int h,
    const double base_background_activity,
    const double background_activity_mean,
    const double background_activity_stddev,
    const double a,
    const double b,
    const double c,
    const double d,
    const double V_spike,
    const double k1,
    const double k2,
    const double k3)
    : NeuronModel{ k, tau_C, beta, h, base_background_activity, background_activity_mean, background_activity_stddev }
    , a{ a }
    , b{ b }
    , c{ c }
    , d{ d }
    , V_spike{ V_spike }
    , k1{ k1 }
    , k2{ k2 }
    , k3{ k3 } {
}

[[nodiscard]] std::unique_ptr<NeuronModel> IzhikevichModel::clone() const {
    return std::make_unique<IzhikevichModel>(get_k(), get_tau_C(), get_beta(), get_h(), get_base_background_activity(), get_background_activity_mean(), get_background_activity_stddev(), a, b, c, d, V_spike, k1, k2, k3);
}

[[nodiscard]] std::vector<ModelParameter> IzhikevichModel::get_parameter() {
    auto res{ NeuronModel::get_parameter() };
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

void IzhikevichModel::init(const size_t number_neurons) {
    NeuronModel::init(number_neurons);
    u.resize(number_neurons);
    init_neurons(0, number_neurons);
}

void models::IzhikevichModel::create_neurons(const size_t creation_count) {
    const auto old_size = NeuronModel::get_num_neurons();
    NeuronModel::create_neurons(creation_count);
    u.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void IzhikevichModel::update_activity(const NeuronID& neuron_id) {
    const auto h = get_h();
    const auto I_syn = get_I_syn(neuron_id);
    auto x = get_x(neuron_id);

    const auto local_neuron_id = neuron_id.get_local_id();

    auto has_spiked = false;

    for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
        x += iter_x(x, u[local_neuron_id], I_syn) / h;
        u[local_neuron_id] += iter_refrac(u[local_neuron_id], x) / h;

        if (spiked(x)) {
            x = c;
            u[local_neuron_id] += d;
            has_spiked = true;
            break;
        }
    }

    set_fired(neuron_id, static_cast<char>(has_spiked));
    set_x(neuron_id, x);
}

void IzhikevichModel::init_neurons(const size_t start_id, const size_t end_id) {
    for (size_t neuron_id = start_id; neuron_id < end_id; ++neuron_id) {
        const auto x = c;
        u[neuron_id] = iter_refrac(b * c, x);
        const auto id = NeuronID{ neuron_id };
        set_fired(id, static_cast<char>(x >= V_spike));
        set_x(id, x);
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
