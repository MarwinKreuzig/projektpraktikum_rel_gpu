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
    const unsigned int h,
    std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator,
    const double a,
    const double b,
    const double c,
    const double d,
    const double V_spike,
    const double k1,
    const double k2,
    const double k3)
    : NeuronModel{ h, std::move(synaptic_input_calculator), std::move(background_activity_calculator) }
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
    return std::make_unique<IzhikevichModel>(get_h(), get_synaptic_input_calculator()->clone(), get_background_activity_calculator()->clone(),
        a, b, c, d, V_spike, k1, k2, k3);
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

void IzhikevichModel::create_neurons(const size_t creation_count) {
    const auto old_size = NeuronModel::get_number_neurons();
    NeuronModel::create_neurons(creation_count);
    u.resize(old_size + creation_count);
    init_neurons(old_size, creation_count);
}

void IzhikevichModel::update_activity(const NeuronID& neuron_id) {
    const auto h = get_h();

    const auto synaptic_input = get_synaptic_input(neuron_id);
    const auto background = get_background_activity(neuron_id);
    const auto input = synaptic_input + background;

    auto x = get_x(neuron_id);

    const auto local_neuron_id = neuron_id.get_neuron_id();

    auto has_spiked = FiredStatus::Inactive;

    for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
        x += iter_x(x, u[local_neuron_id], input) / h;
        u[local_neuron_id] += iter_refrac(u[local_neuron_id], x) / h;

        if (spiked(x)) {
            x = c;
            u[local_neuron_id] += d;
            has_spiked = FiredStatus::Fired;
            break;
        }
    }

    set_fired(neuron_id, has_spiked);
    set_x(neuron_id, x);
}

void IzhikevichModel::init_neurons(const size_t start_id, const size_t end_id) {
    for (size_t neuron_id = start_id; neuron_id < end_id; ++neuron_id) {
        const auto id = NeuronID{ neuron_id };
        u[neuron_id] = iter_refrac(b * c, c);
        set_x(id, c);
    }
}

[[nodiscard]] double IzhikevichModel::iter_x(const double x, const double u, const double input) const noexcept {
    return k1 * x * x + k2 * x + k3 - u + input;
}

[[nodiscard]] double IzhikevichModel::iter_refrac(const double u, const double x) const noexcept {
    return a * (b * x - u);
}

[[nodiscard]] bool IzhikevichModel::spiked(const double x) const noexcept {
    return x >= V_spike;
}
