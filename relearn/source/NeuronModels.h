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
#include "ModelParameter.h"
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
	 * Map of (MPI rank; FiringNeuronIds)
	 * The MPI rank specifies the corresponding process
	 */
	using MapFiringNeuronIds = std::map<int, FiringNeuronIds>;

	NeuronModels(double k, double tau_C, double beta, unsigned int h);

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
	[[nodiscard]] virtual std::vector<ModelParameter> get_parameter();

	/**
	 * Resizes the vectors and initializes their values
	 */
	virtual void init(size_t num_neurons);

	/**
	 * Returns the name of the model
	 */
	[[nodiscard]] virtual std::string name() = 0;

protected:
	virtual void update_activity(const size_t i) = 0;

	virtual void init_neurons() = 0;

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
		explicit ModelA(
			double k = NeuronModels::default_k, 
			double tau_C = NeuronModels::default_tau_C, 
			double beta = NeuronModels::default_beta, 
			unsigned int h = NeuronModels::default_h, 
			double x_0 = ModelA::default_x_0,
			double tau_x = ModelA::default_tau_x,
			unsigned int refrac_time = ModelA::default_refrac_time);

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final;

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final;

		[[nodiscard]] std::vector<ModelParameter> get_parameter() final;

		[[nodiscard]] virtual std::string name();

		void init(size_t num_neurons) final;

	protected:
		void update_activity(const size_t i) final;

		void init_neurons() final;

	private:
		[[nodiscard]] double iter_x(const double x, const double I_syn) const noexcept;

		[[nodiscard]] bool theta(const double x);

		static constexpr double default_x_0{ 0.05 };
		static constexpr double default_tau_x{ 5.0 };
		static constexpr unsigned int default_refrac_time{ 4 };

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
		explicit IzhikevichModel(
			double k = NeuronModels::default_k, 
			double tau_C = NeuronModels::default_tau_C, 
			double beta = NeuronModels::default_beta, 
			unsigned int h = NeuronModels::default_h, 
			double a = IzhikevichModel::default_a,
			double b = IzhikevichModel::default_b,
			double c = IzhikevichModel::default_c,
			double d = IzhikevichModel::default_d,
			double V_spike = IzhikevichModel::default_V_spike,
			double k1 = IzhikevichModel::default_k1,
			double k2 = IzhikevichModel::default_k2,
			double k3 = IzhikevichModel::default_k3);

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final;

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final;

		[[nodiscard]] std::vector<ModelParameter> get_parameter() final;

		[[nodiscard]] virtual std::string name();

		void init(size_t num_neurons) final;

	protected:
		void update_activity(const size_t i) final;

		void init_neurons() final;

	private:
		[[nodiscard]] double iter_x(const double x, const double u, const double I_syn) const noexcept;

		[[nodiscard]] double iter_refrac(const double u, const double x) const noexcept;

		[[nodiscard]] bool spiked(const double x) const noexcept;

		static constexpr double default_a{ 0.1 };
		static constexpr double default_b{ 0.2 };
		static constexpr double default_c{ -65.0 };
		static constexpr double default_d{ 2.0 };
		static constexpr double default_V_spike{ 30.0 };
		static constexpr double default_k1{ 0.04 };
		static constexpr double default_k2{ 5.0 };
		static constexpr double default_k3{ 140.0 };

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
		explicit FitzHughNagumoModel(
			double k = NeuronModels::default_k, 
			double tau_C = NeuronModels::default_tau_C, 
			double beta = NeuronModels::default_beta,
			unsigned int h = NeuronModels::default_h,
			const double a = FitzHughNagumoModel::default_a,
			const double b = FitzHughNagumoModel::default_b,
			const double phi = FitzHughNagumoModel::default_phi);

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final;

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final;

		[[nodiscard]] std::vector<ModelParameter> get_parameter() final;

		[[nodiscard]] virtual std::string name();

		void init(size_t num_neurons) final;

	protected:
		void update_activity(const size_t i) final;

		void init_neurons() final;

	private:
		[[nodiscard]] static double iter_x(const double x, const double w, const double I_syn) noexcept;

		[[nodiscard]] double iter_refrac(const double w, const double x) const noexcept;

		[[nodiscard]] static bool spiked(const double x, const double w) noexcept;

		static constexpr double default_a{ 0.7 };
		static constexpr double default_b{ 0.8 };
		static constexpr double default_phi{ 0.08 };

		std::vector<double> w; // recovery variable

		double a;
		double b;
		double phi;
	};

	class AEIFModel : public NeuronModels {
	public:
		explicit AEIFModel(
			double k = NeuronModels::default_k, 
			double tau_C = NeuronModels::default_tau_C, 
			double beta = NeuronModels::default_beta, 
			unsigned int h = NeuronModels::default_h, 
			const double C = AEIFModel::default_C,
			const double g_L = AEIFModel::default_g_L,
			const double E_L = AEIFModel::default_E_L,
			const double V_T = AEIFModel::default_V_T,
			const double d_T = AEIFModel::default_d_T,
			const double tau_w = AEIFModel::default_tau_w,
			const double a = AEIFModel::default_a,
			const double b = AEIFModel::default_b,
			const double V_peak = AEIFModel::default_V_peak);

		[[nodiscard]] std::unique_ptr<NeuronModels> clone() const final;

		[[nodiscard]] double get_secondary_variable(const size_t i) const noexcept final;

		[[nodiscard]] std::vector<ModelParameter> get_parameter() final;

		[[nodiscard]] virtual std::string name();

		void init(size_t num_neurons) final;

	protected:
		void update_activity(const size_t i) final;

		void init_neurons() final;

	private:
		[[nodiscard]] double f(const double x) const noexcept;

		[[nodiscard]] double iter_x(const double x, const double w, const double I_syn) const noexcept;

		[[nodiscard]] double iter_refrac(const double w, const double x) const noexcept;

		static constexpr double default_C{ 281.0 };
		static constexpr double default_g_L{ 30.0 };
		static constexpr double default_E_L{ -70.6 };
		static constexpr double default_V_T{ -50.4 };
		static constexpr double default_d_T{ 2.0 };
		static constexpr double default_tau_w{ 144.0 };
		static constexpr double default_a{ 4.0 };
		static constexpr double default_b{ 0.0805 };
		static constexpr double default_V_peak{ 20.0 };

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
