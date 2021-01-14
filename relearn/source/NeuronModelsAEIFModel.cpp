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

AEIFModel::AEIFModel(double k, double tau_C, double beta, unsigned int h, const double C, const double g_L, const double E_L, const double V_T, const double, const double tau_w, const double a, const double b, const double V_peak)
  : NeuronModels{ k, tau_C, beta, h },
	C{ C },
	g_L{ g_L },
	E_L{ E_L },
	V_T{ V_T },
	d_T{ d_T },
	tau_w{ tau_w },
	a{ a },
	b{ b },
	V_peak{ V_peak } {
}

[[nodiscard]] std::unique_ptr<NeuronModels> AEIFModel::clone() const {
	return std::make_unique<AEIFModel>(k, tau_C, beta, h, C, g_L, E_L, V_T, d_T, tau_w, a, b, V_peak);
}

[[nodiscard]] double AEIFModel::get_secondary_variable(const size_t i) const noexcept {
	return w[i];
}

[[nodiscard]] std::vector<ModelParameter> AEIFModel::get_parameter() {
	auto res{ NeuronModels::get_parameter() };
	res.reserve(res.size() + 9);
	res.emplace_back(Parameter<double>{ "C", C, 100., 500. });
	res.emplace_back(Parameter<double>{ "g_L", g_L, 0., 100. });
	res.emplace_back(Parameter<double>{ "E_L", E_L, -150., -20. });
	res.emplace_back(Parameter<double>{ "V_T", V_T, -150., 0. });
	res.emplace_back(Parameter<double>{ "d_T", d_T, 0., 10. });
	res.emplace_back(Parameter<double>{ "tau_w", tau_w, 100., 200. });
	res.emplace_back(Parameter<double>{ "a", a, 0., 10. });
	res.emplace_back(Parameter<double>{ "b", b, 0., .3 });
	res.emplace_back(Parameter<double>{ "V_peak", V_peak, 0., 1. });
	return res;
}

[[nodiscard]] std::string AEIFModel::name() {
	return "AEIFModel";
}

void AEIFModel::init(size_t num_neurons) {
	NeuronModels::init(num_neurons);
	w.resize(num_neurons);
	init_neurons();
}

void AEIFModel::update_activity(const size_t i) {
	for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
		x[i] += iter_x(x[i], w[i], I_syn[i]) / h;
		w[i] += iter_refrac(w[i], x[i]) / h;

		if (x[i] >= V_peak) {
			fired[i] = true;
			x[i] = E_L;
			w[i] += b;
		}
	}
}

void AEIFModel::init_neurons() {
	for (size_t i = 0; i < x.size(); ++i) {
		x[i] = E_L;
		w[i] = iter_refrac(0, x[i]);
		fired[i] = x[i] >= V_peak;
	}
}

[[nodiscard]] double AEIFModel::f(const double x) const noexcept {
	return -g_L * (x - E_L) + g_L * d_T * exp((x - V_T) / d_T);
}

[[nodiscard]] double AEIFModel::iter_x(const double x, const double w, const double I_syn) const noexcept {
	return (f(x) - w + I_syn) / C;
}

[[nodiscard]] double AEIFModel::iter_refrac(const double w, const double x) const noexcept {
	return (a * (x - E_L) - w) / tau_w;
}
