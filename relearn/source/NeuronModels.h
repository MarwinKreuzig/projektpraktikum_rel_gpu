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

		Parameter(std::string name, T& value, const T& min, const T& max) : name_{ std::move(name) }, value_{ value }, min_{ min }, max_{ max } {}

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

	private:
		const std::string name_{}; // name of the parameter
		T& value_{};			   // value of the parameter
		const T min_{};			   // minimum value of the parameter
		const T max_{};			   // maximum value of the parameter
	};

	/**
	 * Variant of every Parameter of type T
	 */
	using ModelParameter = std::variant<Parameter<unsigned int>, Parameter<double>, Parameter<size_t>>;

	/**
	 * Map of (MPI rank; FiringNeuronIds)
	 * The MPI rank specifies the corresponding process
	 */
	using MapFiringNeuronIds = std::map<int, FiringNeuronIds>;

	NeuronModels(size_t num_neurons, double k, double tau_C, double beta, unsigned int h);

	virtual ~NeuronModels() = default;

	NeuronModels(const NeuronModels& other) = delete;
	NeuronModels& operator=(const NeuronModels& other) = delete;

	NeuronModels(NeuronModels&& other) = default;
	NeuronModels& operator=(NeuronModels&& other) = default;

	template <typename T, typename... Ts, std::enable_if_t<std::is_base_of<NeuronModels, T>::value, int> = 0>
	[[nodiscard]] static std::unique_ptr<T> create( Ts... args) {
		return std::make_unique<T>(args...);
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
	 * Returns a vector of all possible models
	 */
	[[nodiscard]] static std::vector<std::unique_ptr<NeuronModels>> get_models();

	/**
	 * Returns a vector of ModelParameter of the model
	 */
	[[nodiscard]] virtual std::vector<ModelParameter> get_parameter() {
		return {
			Parameter<size_t>{ "my_num_neurons", my_num_neurons, 0, 10000000000 },
			Parameter<double>{ "k", k, 0., 1. },
			Parameter<double>{ "tau_C", tau_C, 0., 10.e+6 },
			Parameter<double>{ "beta", beta, 0., 1. },
			Parameter<unsigned int>{ "h", h, 0, 1000 },
		};
	}

	/**
	 * Resizes the vectors and initializes their values
	 */
	virtual void init() {
		x.resize(my_num_neurons);
		fired.resize(my_num_neurons);
		I_syn.resize(my_num_neurons);
	}

	/**
	 * Returns the name of the model
	 */
	[[nodiscard]] virtual std::string name() = 0;

protected:
	virtual void update_activity(const size_t i) = 0;

	virtual void init_neurons() = 0;

	static constexpr size_t default_my_num_neurons{ 100 };
	static constexpr double default_k{ 0.03 };
	static constexpr double default_tau_C{ 10000 };
	static constexpr double default_beta{ 0.001 };
	static constexpr unsigned int default_h{ 10 };

	// My local number of neurons
	size_t my_num_neurons;

	// // Model parameters for all neurons
	double k;		// Proportionality factor for synapses in Hz
	double tau_C;	// Decay time of calcium
	double beta;	// Increase in calcium each time a neuron fires
	unsigned int h; // Precision for Euler integration

	// // Variables for each neuron where the array index denotes the neuron ID
	std::vector<double> x;	   // membrane potential v
	std::vector<bool> fired;   // 1: neuron has fired, 0: neuron is inactive
	std::vector<double> I_syn; // Synaptic input
};

namespace models {
	class ModelA : public NeuronModels {
	public:
		explicit ModelA(size_t num_neurons = NeuronModels::default_my_num_neurons, double k = NeuronModels::default_k, double tau_C = NeuronModels::default_tau_C, double beta = NeuronModels::default_beta, unsigned int h = NeuronModels::default_h, const double x_0 = 0.05, const double tau_x = 5., unsigned int refrac_time = 4)
		  : NeuronModels{ num_neurons, k, tau_C, beta, h },
			refrac(num_neurons),
			x_0{ x_0 },
			tau_x{ tau_x },
			refrac_time{ refrac_time } {
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<ModelA>(my_num_neurons, k, tau_C, beta, h, x_0, tau_x, refrac_time);
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return refrac[i];
		}

		[[nodiscard]] std::vector<ModelParameter> get_parameter() final {
			auto res{ NeuronModels::get_parameter() };
			res.reserve(res.size() + 3);
			res.emplace_back(Parameter<double>{ "x_0", x_0, 0., 1. });
			res.emplace_back(Parameter<double>{ "tau_x", tau_x, 0., 1000. });
			res.emplace_back(Parameter<unsigned int>{ "refrac_time", refrac_time, 0, 1000 });
			return res;
		}

		[[nodiscard]] virtual std::string name() {
			return "ModelA";
		}

		void init() final {
			NeuronModels::init();
			refrac.resize(my_num_neurons);
			init_neurons();
		}

	protected:
		void update_activity(const size_t i) final {
			for (unsigned int integration_steps = 0; integration_steps < h; integration_steps++) {
				// Update the membrane potential
				x[i] += iter_x(x[i], I_syn[i]) / h;
			}

			// Neuron ready to fire again
			if (refrac[i] == 0) {
				const bool f = theta(x[i]);
				fired[i] = f;					 // Decide whether a neuron fires depending on its firing rate
				refrac[i] = f ? refrac_time : 0; // After having fired, a neuron is in a refractory state
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
				fired[i] = f;					 // Decide whether a neuron fires depending on its firing rate
				refrac[i] = f ? refrac_time : 0; // After having fired, a neuron is in a refractory state
			}
		}

	private:
		[[nodiscard]] double iter_x(const double x, const double I_syn) const noexcept {
			return ((x_0 - x) / tau_x + I_syn);
		}

		[[nodiscard]] bool theta(const double x) {
			// 1: fire, 0: inactive
			const double threshold = random_number_distribution(random_number_generator);
			return x >= threshold;
		}

		std::vector<unsigned int> refrac; // refractory time

		double x_0;				  // Background or resting activity
		double tau_x;			  // Decay time of firing rate in msec
		unsigned int refrac_time; // Length of refractory period in msec. After an action potential a neuron cannot fire for this time

		// Random number generator for this class (C++11)
		std::mt19937& random_number_generator{ RandomHolder<ModelA>::get_random_generator() };
		// Random number distribution used together with "random_number_generator" (C++11)
		// Uniform distribution for interval [0, 1]
		std::uniform_real_distribution<double> random_number_distribution{ 0.0, nextafter(1.0, 2.0) };
	};

	class IzhikevichModel : public NeuronModels {
	public:
		explicit IzhikevichModel(size_t num_neurons = NeuronModels::default_my_num_neurons, double k = NeuronModels::default_k, double tau_C = NeuronModels::default_tau_C, double beta = NeuronModels::default_beta, unsigned int h = NeuronModels::default_h, const double a = 0.1, const double b = 0.2, const double c = -65., const double d = 2., const double V_spike = 30., const double k1 = 0.04, const double k2 = 5., const double k3 = 140.)
		  : NeuronModels{ num_neurons, k, tau_C, beta, h },
			u(num_neurons),
			a{ a },
			b{ b },
			c{ c },
			d{ d },
			V_spike{ V_spike },
			k1{ k1 },
			k2{ k2 },
			k3{ k3 } {
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<IzhikevichModel>(my_num_neurons, k, tau_C, beta, h, a, b, c, d, V_spike, k1, k2, k3);
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return u[i];
		}

		[[nodiscard]] std::vector<ModelParameter> get_parameter() final {
			auto res{ NeuronModels::get_parameter() };
			res.reserve(res.size() + 8);
			res.emplace_back(Parameter<double>{ "a", a, 0., 1. });
			res.emplace_back(Parameter<double>{ "b", b, 0., 1. });
			res.emplace_back(Parameter<double>{ "c", c, -150., -50. });
			res.emplace_back(Parameter<double>{ "d", d, 0., 10. });
			res.emplace_back(Parameter<double>{ "V_spike", V_spike, 0., 100. });
			res.emplace_back(Parameter<double>{ "k1", k1, 0., 1. });
			res.emplace_back(Parameter<double>{ "k2", k2, 0., 10. });
			res.emplace_back(Parameter<double>{ "k3", k3, 50., 200. });
			return res;
		}

		[[nodiscard]] virtual std::string name() {
			return "IzhikevichModel";
		}

		void init() final {
			NeuronModels::init();
			init_neurons();
		}

	protected:
		void update_activity(const size_t i) final {
			for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
				x[i] += iter_x(x[i], u[i], I_syn[i]) / h;
				u[i] += iter_refrac(u[i], x[i]) / h;

				if (spiked(x[i])) {
					fired[i] = true;
					x[i] = c;
					u[i] += d;
				}
			}
		}

		void init_neurons() final {
			for (size_t i = 0; i < x.size(); ++i) {
				x[i] = c;
				u[i] = iter_refrac(b * c, x[i]);
				fired[i] = x[i] >= V_spike;
			}
		}

	private:
		[[nodiscard]] double iter_x(const double x, const double u, const double I_syn) const noexcept {
			return k1 * x * x + k2 * x + k3 - u + I_syn;
		}

		[[nodiscard]] double iter_refrac(const double u, const double x) const noexcept {
			return a * (b * x - u);
		}

		[[nodiscard]] bool spiked(const double x) const noexcept {
			return x >= V_spike;
		}

		std::vector<double> u; // membrane recovery

		double a; // time-scale of membrane recovery u
		double b; // sensitivity of membrane recovery to membrane potential v (x)
		double c; // after-spike reset value for membrane potential v (x)
		double d; // after-spike reset of membrane recovery u

		double V_spike;

		double k1;
		double k2;
		double k3;
	};

	class FitzHughNagumoModel : public NeuronModels {
	public:
		explicit FitzHughNagumoModel(size_t num_neurons = NeuronModels::default_my_num_neurons, double k = NeuronModels::default_k, double tau_C = NeuronModels::default_tau_C, double beta = NeuronModels::default_beta, unsigned int h = NeuronModels::default_h, const double a = 0.7, const double b = 0.8, const double phi = 0.08)
		  : NeuronModels{ num_neurons, k, tau_C, beta, h },
			w(num_neurons),
			a{ a },
			b{ b },
			phi{ phi } {
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<FitzHughNagumoModel>(my_num_neurons, k, tau_C, beta, h, a, b, phi);
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return w[i];
		}

		[[nodiscard]] std::vector<ModelParameter> get_parameter() final {
			auto res{ NeuronModels::get_parameter() };
			res.reserve(res.size() + 3);
			res.emplace_back(Parameter<double>{ "a", a, 0., 5. });
			res.emplace_back(Parameter<double>{ "b", b, 0., 5. });
			res.emplace_back(Parameter<double>{ "phi", phi, 0., .3 });
			return res;
		}

		[[nodiscard]] virtual std::string name() {
			return "FitzHughNagumoModel";
		}

		void init() final {
			NeuronModels::init();
			init_neurons();
		}

	protected:
		void update_activity(const size_t i) final {
			fired[i] = false;

			// Update the membrane potential
			for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
				x[i] += iter_x(x[i], w[i], I_syn[i]) / h;
				w[i] += iter_refrac(w[i], x[i]) / h;

				if (spiked(x[i], w[i])) {
					fired[i] = true;
				}
			}
		}

		void init_neurons() final {
			for (size_t i = 0; i < x.size(); ++i) {
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
			return phi * (x + a - b * w);
		}

		[[nodiscard]] static bool spiked(const double x, const double w) noexcept {
			return w > iter_x(x, 0, 0) && x > 1.;
		}

		std::vector<double> w; // recovery variable

		double a;
		double b;
		double phi;
	};

	class AEIFModel : public NeuronModels {
	public:
		explicit AEIFModel(size_t num_neurons = NeuronModels::default_my_num_neurons, double k = NeuronModels::default_k, double tau_C = NeuronModels::default_tau_C, double beta = NeuronModels::default_beta, unsigned int h = NeuronModels::default_h, const double C = 281., const double g_L = 30., const double E_L = -70.6, const double V_T = -50.4, const double d_T = 2., const double tau_w = 144., const double a = 4., const double b = 0.0805, const double V_peak = 20.)
		  : NeuronModels{ num_neurons, k, tau_C, beta, h },
			w(num_neurons),
			C{ C },
			g_L{ g_L },
			E_L{ E_L },
			V_T{ V_T },
			d_T{ d_T },
			tau_w{ tau_w },
			a{ a },
			b{ b },
			V_peak{ V_peak } {
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<AEIFModel>(my_num_neurons, k, tau_C, beta, h, C, g_L, E_L, V_T, d_T, tau_w, a, b, V_peak);
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return w[i];
		}

		[[nodiscard]] std::vector<ModelParameter> get_parameter() final {
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

		[[nodiscard]] virtual std::string name() {
			return "AEIFModel";
		}

		void init() final {
			NeuronModels::init();
			init_neurons();
		}

	protected:
		void update_activity(const size_t i) final {
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

		void init_neurons() final {
			for (size_t i = 0; i < x.size(); ++i) {
				x[i] = E_L;
				w[i] = iter_refrac(0, x[i]);
				fired[i] = x[i] >= V_peak;
			}
		}

	private:
		[[nodiscard]] double f(const double x) const noexcept {
			return -g_L * (x - E_L) + g_L * d_T * exp((x - V_T) / d_T);
		}

		[[nodiscard]] double iter_x(const double x, const double w, const double I_syn) const noexcept {
			return (f(x) - w + I_syn) / C;
		}

		[[nodiscard]] double iter_refrac(const double w, const double x) const noexcept {
			return (a * (x - E_L) - w) / tau_w;
		}

		std::vector<double> w; // adaption variable

		double C;	  // membrance capacitance
		double g_L;	  // leak conductance
		double E_L;	  // leak reversal potential
		double V_T;	  // spike threshold
		double d_T;	  // slope factor
		double tau_w; // adaptation time constant
		double a;	  // subthreshold
		double b;	  // spike-triggered adaptation

		double V_peak; // spike trigger
	};

} // namespace models
