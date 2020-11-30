/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include "LogMessages.h"
#include "MPIWrapper.h"
#include "NetworkGraph.h"
#include "Random.h"
#include "Timers.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <random>
#include <variant>
#include <vector>

class NeuronMonitor;

template <typename... Ts>
struct overloaded : Ts...
{
	using Ts::operator()...;
};

template <typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

class NeuronModels {
	friend class NeuronMonitor;

public:
	/**
	 * Type for firing neuron ids which are used with MPI
	 */
	class FiringNeuronIds {
	public:
		// Return size
		[[nodiscard]] size_t size() const noexcept {
			return neuron_ids.size();
		}

		// Resize the number of neuron ids
		void resize(size_t size) {
			neuron_ids.resize(size);
		}

		// Append neuron id
		//
		// NOTE: This function asks the user to guarantee
		// that elements are appended in increasing/decreasing order.
		// That is they must be sorted. Otherwise, behavior is undefined.
		void append_if_not_found_sorted(size_t neuron_id) {
			// Neuron id not included yet
			const bool found = find(neuron_id);
			if (!found) {
				neuron_ids.push_back(neuron_id);
			}
		}

		// Test if "neuron_id" exists
		[[nodiscard]] bool find(size_t neuron_id) const {
			return std::binary_search(neuron_ids.begin(), neuron_ids.end(), neuron_id);
		}

		// Get neuron id at index "neuron_id_index"
		[[nodiscard]] size_t get_neuron_id(size_t neuron_id_index) const noexcept {
			return neuron_ids[neuron_id_index];
		}

		// Get pointer to data
		[[nodiscard]] size_t* get_neuron_ids() noexcept {
			return neuron_ids.data();
		}

		[[nodiscard]] const size_t* get_neuron_ids() const noexcept {
			return neuron_ids.data();
		}

		[[nodiscard]] size_t get_neuron_ids_size_in_bytes() const noexcept {
			return neuron_ids.size() * sizeof(size_t);
		}

	private:
		std::vector<size_t> neuron_ids; // Firing neuron ids
										 // This vector is used as MPI communication buffer
	};

	/**
	 * Parameter of a model of type T
	 */
	template <typename T>
	class Parameter {
	public:
		using value_type = T;

		Parameter(std::string name, const T& value, const T& min, const T& max) : name_{ std::move(name) }, value_{ value }, min_{ min }, max_{ max } {}

		[[nodiscard]] const std::string& name() const noexcept {
			return name_;
		}

		[[nodiscard]] value_type& value() noexcept {
			return value_;
		}

		[[nodiscard]] const value_type& value() const noexcept {
			return value_;
		}

		[[nodiscard]] const value_type& min() const noexcept {
			return min_;
		}

		[[nodiscard]] const value_type& max() const noexcept {
			return max_;
		}

		/**
		 * Compares min, max and the name for equality, checking if lhs and rhs are for the same parameter. value may be different.
		 */
		[[nodiscard]] friend bool operator==(const Parameter<T>& lhs, const Parameter<T>& rhs) noexcept {
			return lhs.min_ == rhs.min_ && lhs.max_ == rhs.max_ && lhs.name_ == rhs.name_;
		}

		/**
		 * Checks if the saved value is in the range of the Parameter
		 */
		[[nodiscard]] bool value_in_range() const noexcept {
			return min_ <= value_ && value_ <= max_;
		}

		/**
		 * Assigns the value of source to the value of the first Parameter that compares equal to the source
		 */
		template <typename W, typename U, typename... Us, std::enable_if_t<std::is_same_v<Parameter<T>, W> && std::is_same_v<Parameter<T>, U> && (... && std::is_same_v<Parameter<T>, Us>), int> = 0>
		static void do_parameter_assignment(const W& source, U& p, Us&... ps) noexcept {
			if (p == source) {
				p.value() = source.value();
				return;
			}
			do_parameter_assignment(source, ps...);
		}

		/**
		 * Assigns the value of source to the value of p if it compares equal to the source
		 */
		template <typename W, typename U, std::enable_if_t<std::is_same_v<Parameter<T>, W> && std::is_same_v<Parameter<T>, U>, int> = 0>
		static void do_parameter_assignment(const W& source, U& p) noexcept {
			if (p == source) {
				p.value() = source.value();
				return;
			}
		}

	private:
		const std::string name_{}; // name of the parameter
		T value_{};				   // value of the parameter
		const T min_{};			   // minimum value of the parameter
		const T max_{};			   // maximum value of the parameter
	};

	/**
	 * Variant of every Parameter of type T
	 */
	using ModelParameter = std::variant<Parameter<unsigned int>, Parameter<double>>;

	/**
	 * Map of (MPI rank; FiringNeuronIds)
	 * The MPI rank specifies the corresponding process
	 */
	using MapFiringNeuronIds = std::map<int, FiringNeuronIds>;

	NeuronModels(size_t num_neurons, double k, double tau_C, double beta, int h);

	virtual ~NeuronModels() = default;

	NeuronModels(const NeuronModels& other) = delete;
	NeuronModels& operator=(const NeuronModels& other) = delete;

	NeuronModels(NeuronModels&& other) = default;
	NeuronModels& operator=(NeuronModels&& other) = default;

	template <typename T, typename... Ts, std::enable_if_t<std::is_base_of<NeuronModels, T>::value, int> = 0>
	[[nodiscard]] static std::unique_ptr<T> create(size_t num_neurons, double k, double tau_C, double beta, int h, Ts... model_specific_args) {
		return std::make_unique<T>(num_neurons, k, tau_C, beta, h, model_specific_args...);
	}

	[[nodiscard]] virtual std::unique_ptr<NeuronModels> clone() const = 0;

	[[nodiscard]] double get_beta() const noexcept {
		return beta;
	}

	[[nodiscard]] bool get_fired(const size_t i) const noexcept {
		return fired[i];
	}

	[[nodiscard]] double get_x(const size_t i) const noexcept {
		return x[i];
	}

	[[nodiscard]] const std::vector<double>& get_x() const noexcept {
		return x;
	}

	[[nodiscard]] virtual double get_secondary_variable(size_t i) const noexcept = 0;

	/* Performs one iteration step of update in electrical activity */
	void update_electrical_activity(const NetworkGraph& network_graph, std::vector<double>& C);

	/**
	 * Returns a vector of ModelParameter of the model
	 */
	[[nodiscard]] virtual std::vector<ModelParameter> get_parameter() const = 0;

	/**
	 * Sets the parameter of the model to the ones in param, if param are parameters of the model
	 */
	virtual void set_parameter(const std::vector<ModelParameter>& param) = 0;

protected:
	virtual void update_activity(size_t i) = 0;

	virtual void init_neurons() = 0;

	/**
	 * Checks if every parameter in param is a parameter of the model
	 * returns true if they are, 
	 * false if at least one of param is not a parameter of the model
	 */
	[[nodiscard]] bool are_params_of_model(const std::vector<ModelParameter>& param) const {
		const auto model_param = get_parameter();
		auto is_param_of_model = [&](const auto& p) {
			return std::any_of(std::begin(model_param), std::end(model_param), [&](auto& v) { return v == p; });
		};
		return std::all_of(std::begin(param), std::end(param), is_param_of_model);
	}

	/**
	 * Checks if every parameter in param is a parameter of the model via are_params_of_model
	 * Throws an exception if are_params_of_model returns false
	 */
	void check_if_params_are_from_model(const std::vector<ModelParameter>& param) const {
		if (!are_params_of_model(param)) {
			RelearnException::fail("Received ModelParameter contain parameter that do not belong to the current Model");
		}
	}

	// My local number of neurons
	size_t my_num_neurons;

	// // Model parameters for all neurons
	double k;	  // Proportionality factor for synapses in Hz
	double tau_C; // Decay time of calcium
	double beta;  // Increase in calcium each time a neuron fires
	int h;		  // Precision for Euler integration

	// // Variables for each neuron where the array index denotes the neuron ID
	std::vector<double> x;				// membrane potential v
	std::vector<bool> fired;			// 1: neuron has fired, 0: neuron is inactive
	std::vector<double> I_syn;			// Synaptic input
};

namespace models {
	class ModelA : public NeuronModels {
	public:
		ModelA(size_t num_neurons, double k, double tau_C, double beta, int h, const double x_0, const double tau_x, unsigned int refrac_time)
		  : NeuronModels{ num_neurons, k, tau_C, beta, h },
			refrac(num_neurons),
			x_0{ "x_0", x_0, 0., 1. },
			tau_x{ "tau_x", tau_x, 0., 1000. },
			refrac_time{ "refrac_time", refrac_time, 0, 1000 } {
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<ModelA>(my_num_neurons, k, tau_C, beta, h, x_0.value(), tau_x.value(), refrac_time.value());
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return refrac[i];
		}

		[[nodiscard]] std::vector<ModelParameter> get_parameter() const final {
			return { ModelParameter{ x_0 }, ModelParameter{ tau_x }, ModelParameter{ refrac_time } };
		}

		void set_parameter(const std::vector<ModelParameter>& param) final {
			check_if_params_are_from_model(param);

			std::for_each(std::begin(param), std::end(param), [&](const auto& par) {
				std::visit(overloaded{
						   [&](const Parameter<double>& v) {
							   if (v.value_in_range()) {
								   Parameter<double>::do_parameter_assignment(v, x_0, tau_x);
							   }
						   },
						   [&](const Parameter<unsigned int>& v) {
							   if (v.value_in_range()) {
								   Parameter<unsigned int>::do_parameter_assignment(v, refrac_time);
							   }
						   },
						   [](auto&) { RelearnException::fail(); } },
						   par);
			});
		}

	protected:
		void update_activity(const size_t i) final {
			for (int integration_steps = 0; integration_steps < h; integration_steps++) {
				// Update the membrane potential
				x[i] += iter_x(x[i], I_syn[i]) / h;
			}

			// Neuron ready to fire again
			if (refrac[i] == 0) {
				const bool f = theta(x[i]);
				fired[i] = f;							 // Decide whether a neuron fires depending on its firing rate
				refrac[i] = f ? refrac_time.value() : 0; // After having fired, a neuron is in a refractory state
			}
			// Neuron now/still in refractory state
			else {
				fired[i] = false; // Set neuron inactive
				--refrac[i];	  // Decrease refractory time
			}
		}

		void init_neurons() final {
			for (size_t i = 0; i < x.size(); ++i) {
				x[i] = random_number_distribution(random_number_generator);
				const bool f = theta(x[i]);
				fired[i] = f;							 // Decide whether a neuron fires depending on its firing rate
				refrac[i] = f ? refrac_time.value() : 0; // After having fired, a neuron is in a refractory state
			}
		}

	private:
		[[nodiscard]] double iter_x(const double x, const double I_syn) const noexcept {
			return ((x_0.value() - x) / tau_x.value() + I_syn);
		}

		[[nodiscard]] bool theta(const double x) {
			// 1: fire, 0: inactive
			const double threshold = random_number_distribution(random_number_generator);
			return x >= threshold;
		}

		std::vector<unsigned int> refrac; // refractory time

		Parameter<double> x_0;				 // Background or resting activity
		Parameter<double> tau_x;			 // Decay time of firing rate in msec
		Parameter<unsigned int> refrac_time; // Length of refractory period in msec. After an action potential a neuron cannot fire for this time

		// Random number generator for this class (C++11)
		std::mt19937& random_number_generator{ RandomHolder<ModelA>::get_random_generator() };
		// Random number distribution used together with "random_number_generator" (C++11)
		// Uniform distribution for interval [0, 1]
		std::uniform_real_distribution<double> random_number_distribution{ 0.0, nextafter(1.0, 2.0) };
	};

	class IzhikevichModel : public NeuronModels {
	public:
		IzhikevichModel(size_t num_neurons, double k, double tau_C, double beta, int h, const double a = 0.1, const double b = 0.2, const double c = -65., const double d = 2., const double V_spike = 30., const double k1 = 0.04, const double k2 = 5., const double k3 = 140.)
		  : NeuronModels{ num_neurons, k, tau_C, beta, h },
			u(num_neurons),
			a{ "a", a, 0., 1. },
			b{ "b", b, 0., 1. },
			c{ "c", c, -150., -50. },
			d{ "d", d, 0., 10. },
			V_spike{ "V_spike", V_spike, 0., 100. },
			k1{ "k1", k1, 0., 1. },
			k2{ "k2", k2, 0., 10. },
			k3{ "k3", k3, 50., 200. } {
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<IzhikevichModel>(my_num_neurons, k, tau_C, beta, h, a.value(), b.value(), c.value(), d.value(), V_spike.value(), k1.value(), k2.value(), k3.value());
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return u[i];
		}

		[[nodiscard]] std::vector<ModelParameter> get_parameter() const final {
			return { ModelParameter{ a }, ModelParameter{ b }, ModelParameter{ c }, ModelParameter{ d }, ModelParameter{ V_spike }, ModelParameter{ k1 }, ModelParameter{ k2 }, ModelParameter{ k3 } };
		}

		void set_parameter(const std::vector<ModelParameter>& param) final {
			check_if_params_are_from_model(param);

			std::for_each(std::begin(param), std::end(param), [&](const auto& par) {
				std::visit(overloaded{
						   [&](const Parameter<double>& v) {
							   if (v.value_in_range()) {
								   Parameter<double>::do_parameter_assignment(v, a, b, c, d, V_spike, k1, k2, k3);
							   }
						   },
						   [](auto&) { RelearnException::fail(); } },
						   par);
			});
		}

	protected:
		void update_activity(const size_t i) final {
			for (int integration_steps = 0; integration_steps < h; ++integration_steps) {
				x[i] += iter_x(x[i], u[i], I_syn[i]) / h;
				u[i] += iter_refrac(u[i], x[i]) / h;

				if (spiked(x[i])) {
					fired[i] = true;
					x[i] = c.value();
					u[i] += d.value();
				}
			}
		}

		void init_neurons() final {
			for (auto i = 0; i < x.size(); ++i) {
				x[i] = c.value();
				u[i] = iter_refrac(b.value() * c.value(), x[i]);
				fired[i] = x[i] >= V_spike.value();
			}
		}

	private:
		[[nodiscard]] double iter_x(const double x, const double u, const double I_syn) const noexcept {
			return k1.value() * x * x + k2.value() * x + k3.value() - u + I_syn;
		}

		[[nodiscard]] double iter_refrac(const double u, const double x) const noexcept {
			return a.value() * (b.value() * x - u);
		}

		[[nodiscard]] bool spiked(const double x) const noexcept {
			return x >= V_spike.value();
		}

		std::vector<double> u; // membrane recovery

		Parameter<double> a; // time-scale of membrane recovery u
		Parameter<double> b; // sensitivity of membrane recovery to membrane potential v (x)
		Parameter<double> c; // after-spike reset value for membrane potential v (x)
		Parameter<double> d; // after-spike reset of membrane recovery u

		Parameter<double> V_spike;

		Parameter<double> k1;
		Parameter<double> k2;
		Parameter<double> k3;
	};

	class FitzHughNagumoModel : public NeuronModels {
	public:
		FitzHughNagumoModel(size_t num_neurons, double k, double tau_C, double beta, int h, const double a = 0.7, const double b = 0.8, const double phi = 0.08)
		  : NeuronModels{ num_neurons, k, tau_C, beta, h },
			w(num_neurons),
			a{ "a", a, 0., 5. },
			b{ "b", b, 0., 5. },
			phi{ "phi", phi, 0., .3 } {
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<FitzHughNagumoModel>(my_num_neurons, k, tau_C, beta, h, a.value(), b.value(), phi.value());
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return w[i];
		}

		[[nodiscard]] std::vector<ModelParameter> get_parameter() const final {
			return { ModelParameter{ a }, ModelParameter{ b }, ModelParameter{ phi } };
		}

		void set_parameter(const std::vector<ModelParameter>& param) final {
			check_if_params_are_from_model(param);

			std::for_each(std::begin(param), std::end(param), [&](const auto& par) {
				std::visit(overloaded{
						   [&](const Parameter<double>& v) {
							   if (v.value_in_range()) {
								   Parameter<double>::do_parameter_assignment(v, a, b, phi);
							   }
						   },
						   [](auto&) { RelearnException::fail(); } },
						   par);
			});
		}

	protected:
		void update_activity(const size_t i) final {
			fired[i] = false;

			// Update the membrane potential
			for (int integration_steps = 0; integration_steps < h; ++integration_steps) {
				x[i] += iter_x(x[i], w[i], I_syn[i]) / h;
				w[i] += iter_refrac(w[i], x[i]) / h;

				if (spiked(x[i], w[i])) {
					fired[i] = true;
				}
			}
		}

		void init_neurons() final {
			for (auto i = 0; i < x.size(); ++i) {
				x[i] = -1.2;
				w[i] = iter_refrac(-.6, x[i]);
				fired[i] = spiked(x[i], w[i]);
			}
		}

	private:
		[[nodiscard]] static double iter_x(const double x, const double w, const double I_syn) noexcept {
			return x - x * x * x / 3 - w + I_syn;
		}

		[[nodiscard]] double iter_refrac(const double w, const double x) const noexcept {
			return phi.value() * (x + a.value() - b.value() * w);
		}

		[[nodiscard]] static bool spiked(const double x, const double w) noexcept {
			return w > iter_x(x, 0, 0) && x > 1.;
		}

		std::vector<double> w; // recovery variable

		Parameter<double> a;
		Parameter<double> b;
		Parameter<double> phi;
	};

	class AEIFModel : public NeuronModels {
	public:
		AEIFModel(size_t num_neurons, double k, double tau_C, double beta, int h, const double C = 281., const double g_L = 30., const double E_L = -70.6, const double V_T = -50.4, const double d_T = 2., const double tau_w = 144., const double a = 4., const double b = 0.0805, const double V_peak = 20.)
		  : NeuronModels{ num_neurons, k, tau_C, beta, h },
			w(num_neurons),
			C{ "C", C, 100., 500. },
			g_L{ "g_L", g_L, 0., 100. },
			E_L{ "E_L", E_L, -150., -20. },
			V_T{ "V_T", V_T, -150., 0. },
			d_T{ "d_T", d_T, 0., 10. },
			tau_w{ "tau_w", tau_w, 100., 200. },
			a{ "a", a, 0., 10. },
			b{ "b", b, 0., .3 },
			V_peak{ "V_peak", V_peak, 0., 1. } {
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<AEIFModel>(my_num_neurons, k, tau_C, beta, h, C.value(), g_L.value(), E_L.value(), V_T.value(), d_T.value(), tau_w.value(), a.value(), b.value(), V_peak.value());
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return w[i];
		}

		[[nodiscard]] std::vector<ModelParameter> get_parameter() const final {
			return { ModelParameter{ C }, ModelParameter{ g_L }, ModelParameter{ E_L }, ModelParameter{ V_T }, ModelParameter{ d_T }, ModelParameter{ tau_w }, ModelParameter{ a }, ModelParameter{ b }, ModelParameter{ V_peak } };
		}

		void set_parameter(const std::vector<ModelParameter>& param) final {
			check_if_params_are_from_model(param);

			std::for_each(std::begin(param), std::end(param), [&](const auto& par) {
				std::visit(overloaded{
						   [&](const Parameter<double>& v) {
							   if (v.value_in_range()) {
								   Parameter<double>::do_parameter_assignment(v, C, g_L, E_L, V_T, d_T, tau_w, a, b, V_peak);
							   }
						   },
						   [](auto&) { RelearnException::fail(); } },
						   par);
			});
		}

	protected:
		void update_activity(const size_t i) final {
			for (int integration_steps = 0; integration_steps < h; ++integration_steps) {
				x[i] += iter_x(x[i], w[i], I_syn[i]) / h;
				w[i] += iter_refrac(w[i], x[i]) / h;

				if (x[i] >= V_peak.value()) {
					fired[i] = true;
					x[i] = E_L.value();
					w[i] += b.value();
				}
			}
		}

		void init_neurons() final {
			for (int i = 0; i < x.size(); ++i) {
				x[i] = E_L.value();
				w[i] = iter_refrac(0, x[i]);
				fired[i] = x[i] >= V_peak.value();
			}
		}

	private:
		[[nodiscard]] double f(const double x) const noexcept {
			return -g_L.value() * (x - E_L.value()) + g_L.value() * d_T.value() * exp((x - V_T.value()) / d_T.value());
		}

		[[nodiscard]] double iter_x(const double x, const double w, const double I_syn) const noexcept {
			return (f(x) - w + I_syn) / C.value();
		}

		[[nodiscard]] double iter_refrac(const double w, const double x) const noexcept {
			return (a.value() * (x - E_L.value()) - w) / tau_w.value();
		}

		std::vector<double> w; // adaption variable

		Parameter<double> C;	 // membrance capacitance
		Parameter<double> g_L;	 // leak conductance
		Parameter<double> E_L;	 // leak reversal potential
		Parameter<double> V_T;	 // spike threshold
		Parameter<double> d_T;	 // slope factor
		Parameter<double> tau_w; // adaptation time constant
		Parameter<double> a;	 // subthreshold
		Parameter<double> b;	 // spike-triggered adaptation

		Parameter<double> V_peak; // spike trigger
	};

} // namespace models
