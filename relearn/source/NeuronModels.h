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

protected:
	virtual void update_activity(size_t i) = 0;

	virtual void init_neurons() = 0;

	// My local number of neurons
	size_t my_num_neurons;

	// // Model parameters for all neurons
	double k;	 // Proportionality factor for synapses in Hz
	double tau_C; // Decay time of calcium
	double beta; // Increase in calcium each time a neuron fires
	int h;		 // Precision for Euler integration

	// // Variables for each neuron where the array index denotes the neuron ID
	std::vector<double> x;				// membrane potential v
	std::vector<bool> fired;			// 1: neuron has fired, 0: neuron is inactive
	std::vector<double> I_syn;			// Synaptic input
};

namespace models {
	class ModelA : public NeuronModels {
	public:
		ModelA(size_t num_neurons, double k, double tau_C, double beta, int h,
			const double x_0, const double tau_x, unsigned int refrac_time)
			: NeuronModels{ num_neurons, k, tau_C, beta, h }, refrac(num_neurons), x_0{ x_0 }, tau_x{ tau_x }, refrac_time{ refrac_time }
		{
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<ModelA>(my_num_neurons, k, tau_C, beta, h,
				x_0, tau_x, refrac_time);
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return refrac[i];
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
				fired[i] = f;							// Decide whether a neuron fires depending on its firing rate
				refrac[i] = f ? refrac_time : 0;		// After having fired, a neuron is in a refractory state
			}
			// Neuron now/still in refractory state
			else {
				fired[i] = false;						// Set neuron inactive
				--refrac[i];							// Decrease refractory time
			}
		}

		void init_neurons() final {
			for (size_t i = 0; i < x.size(); ++i) {
				x[i] = random_number_distribution(random_number_generator);
				const bool f = theta(x[i]);
				fired[i] = f;							// Decide whether a neuron fires depending on its firing rate
				refrac[i] = f ? refrac_time : 0;		// After having fired, a neuron is in a refractory state
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

		double x_0;			// Background or resting activity
		double tau_x;		// Decay time of firing rate in msec
		unsigned int refrac_time; // Length of refractory period in msec. After an action potential a neuron cannot fire for this time

		// Random number generator for this class (C++11)
		std::mt19937& random_number_generator{ RandomHolder<ModelA>::get_random_generator() };
		// Random number distribution used together with "random_number_generator" (C++11)
		// Uniform distribution for interval [0, 1]
		std::uniform_real_distribution<double> random_number_distribution{ 0.0, nextafter(1.0, 2.0) };
	};

	class IzhikevichModel : public NeuronModels {
	public:
		IzhikevichModel(size_t num_neurons, double k, double tau_C, double beta, int h,
			const double a = 0.1, const double b = 0.2, const double c = -65., const double d = 2.,
			const double V_spike = 30., const double k1 = 0.04, const double k2 = 5., const double k3 = 140.)
			: NeuronModels{ num_neurons, k, tau_C, beta, h }, u(num_neurons), a{ a }, b{ b }, c{ c }, d{ d }, V_spike{ V_spike }, k1{ k1 }, k2{ k2 }, k3{ k3 }
		{
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<IzhikevichModel>(my_num_neurons, k, tau_C, beta, h,
				a, b, c, d, V_spike, k1, k2, k3);
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return u[i];
		}

	protected:
		void update_activity(const size_t i) final {
			for (int integration_steps = 0; integration_steps < h; ++integration_steps) {
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
			for (auto i = 0; i < x.size(); ++i) {
				x[i] = c;
				u[i] = iter_refrac(b * c, x[i]);
				fired[i] =x[i] >= V_spike;
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
		FitzHughNagumoModel(size_t num_neurons, double k, double tau_C, double beta, int h,
			const double a = 0.7, const double b = 0.8, const double phi = 0.08)
			: NeuronModels{ num_neurons, k, tau_C, beta, h }, w(num_neurons), a{ a }, b{ b }, phi{ phi }
		{
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<FitzHughNagumoModel>(my_num_neurons, k, tau_C, beta, h,
				a, b, phi);
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return w[i];
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
		AEIFModel(size_t num_neurons, double k, double tau_C, double beta, int h,
			const double C = 281., const double g_L = 30., const double E_L = -70.6, const double V_T = -50.4,
			const double d_T = 2., const double tau_w = 144., const double a = 4., const double b = 0.0805, const double V_peak = 20.)
			: NeuronModels{ num_neurons, k, tau_C, beta, h }, w(num_neurons), C{ C }, g_L{ g_L }, E_L{ E_L }, V_T{ V_T }, d_T{ d_T }, tau_w{ tau_w }, a{ a }, b{ b }, V_peak{ V_peak }
		{
			init_neurons();
		}

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final {
			return std::make_unique<AEIFModel>(my_num_neurons, k, tau_C, beta, h,
				C, g_L, E_L, V_T, d_T, tau_w, a, b, V_peak);
		}

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final {
			return w[i];
		}

	protected:
		void update_activity(const size_t i) final {
			for (int integration_steps = 0; integration_steps < h; ++integration_steps) {
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
			for (int i = 0; i < x.size(); ++i) {
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

		double C;	 // membrance capacitance
		double g_L;	 // leak conductance
		double E_L;	 // leak reversal potential
		double V_T;	 // spike threshold
		double d_T;	 // slope factor
		double tau_w; // adaptation time constant
		double a;	 // subthreshold
		double b;	 // spike-triggered adaptation

		double V_peak; // spike trigger
	};

} // namespace models
