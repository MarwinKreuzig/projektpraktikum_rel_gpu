/*
 * File:   NeuronModels.h
 * Author: naveau
 *
 * Created on September 26, 2014, 9:31 PM
 */

#ifndef NEURONMODELS_H
#define	NEURONMODELS_H

#include <cstddef>
#include <random>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include <memory>
#include "NetworkGraph.h"
#include "MPIInfos.h"
#include "LogMessages.h"
#include "Timers.h"
#include "Random.h"

namespace models
{
	class Model_Ifc
	{
	public:
		virtual void update_activity(double &x, double &refrac, double &I_syn, unsigned short &fired, const double h) = 0;
		virtual void init_neurons(std::vector<double> &x, std::vector<double> &refrac, std::vector<unsigned short> &fired) = 0;

		template <typename T, typename... Ts>
		[[nodiscard]] static std::unique_ptr<T> create(Ts... args) { return std::make_unique<T>(args...); }
	};

	class ModelA : public Model_Ifc
	{
	public:
		ModelA(const double x_0, const double tau_x, const double refrac_time) : x_0{x_0}, tau_x{tau_x}, refrac_time{refrac_time} {}

		void update_activity(double &x, double &refrac, double &I_syn, unsigned short &fired, const double h) final
		{
			for (int integration_steps = 0; integration_steps < h; integration_steps++)
			{
				// Update the membrane potential
				x += iter_x(x, I_syn) / h;
			}

			// Neuron ready to fire again
			if (refrac == 0)
			{
				fired = static_cast<unsigned short>(theta(x));	   // Decide whether a neuron fires depending on its firing rate
				refrac = static_cast<double>(fired * refrac_time); // After having fired, a neuron is in a refractory state
			}
			// Neuron now/still in refractory state
			else
			{
				fired = 0; // Set neuron inactive
				--refrac;  // Decrease refractory time
			}
		}

		void init_neurons(std::vector<double> &x, std::vector<double> &refrac, std::vector<unsigned short> &fired) final
		{
			for (size_t i = 0; i < x.size(); ++i)
			{
				x[i] = random_number_distribution(random_number_generator);
				fired[i] = static_cast<unsigned short>(theta(x[i]));
				refrac[i] = static_cast<double>(fired[i]) * refrac_time;
			}
		}

	private:
		[[nodiscard]] double iter_x(const double x, const double I_syn) const { return ((x_0 - x) / tau_x + I_syn); }

		[[nodiscard]] bool theta(const double x)
		{
			// 1: fire, 0: inactive
			const double threshold = random_number_distribution(random_number_generator);
			return x >= threshold;
		}

		double x_0;			// Background or resting activity
		double tau_x;		// Decay time of firing rate in msec
		double refrac_time; // Length of refractory period in msec. After an action potential a neuron cannot fire for this time

		// Random number generator for this class (C++11)
		std::mt19937 &random_number_generator{RandomHolder<ModelA>::get_random_generator()};
		// Random number distribution used together with "random_number_generator" (C++11)
		// Uniform distribution for interval [0, 1]
		std::uniform_real_distribution<double> random_number_distribution{0.0, nextafter(1.0, 2.0)};
	};

	class IzhikevichModel : public Model_Ifc
	{
	public:
		IzhikevichModel() = default;
		IzhikevichModel(const double a, const double b, const double c, const double d) : a{a}, b{b}, c{c}, d{d} {}

		void update_activity(double &x, double &refrac, double &I_syn, unsigned short &fired, const double h) final
		{
			for (int integration_steps = 0; integration_steps < h; ++integration_steps)
			{
				x += iter_x(x, refrac, I_syn) / h;
				refrac += iter_refrac(refrac, x) / h;

				if (spiked(x))
				{
					fired = 1;
					x = c;
					refrac += d;
					break;
				}
			}
		}

		void init_neurons(std::vector<double> &x, std::vector<double> &refrac, std::vector<unsigned short> &fired) final
		{
			for (auto i = 0; i < x.size(); ++i)
			{
				x[i] = c;
				refrac[i] = iter_refrac(b * c, x[i]);
				fired[i] = static_cast<unsigned short>(x[i] >= V_spike);
			}
		}

	private:
		[[nodiscard]] static double iter_x(const double x, const double refrac, const double I_syn) { return k1 * x * x + k2 * x + k3 - refrac + I_syn; }
		[[nodiscard]] double iter_refrac(const double refrac, const double x) const { return a * (b * x - refrac); }

		[[nodiscard]] static bool spiked(const double x) { return x >= V_spike; }

		static constexpr double k1{0.04};
		static constexpr double k2{5.};
		static constexpr double k3{140.};

		static constexpr double V_spike{30.};

		double a{0.1};	// time-scale of membrane recovery u
		double b{0.2};	// sensitivity of membrane recovery to membrane potential v (x)
		double c{-65.}; // after-spike reset value for membrane potential v (x)
		double d{2.};	// after-spike reset of membrane recovery u
	};

	class FitzHughNagumoModel : public Model_Ifc
	{
	public:
		FitzHughNagumoModel() = default;
		FitzHughNagumoModel(const double a, const double b, const double phi) : a{a}, b{b}, phi{phi} {}

		void update_activity(double &x, double &refrac, double &I_syn, unsigned short &fired, const double h) final
		{
			fired = 0;

			// Update the membrane potential
			for (int integration_steps = 0; integration_steps < h; ++integration_steps)
			{
				x += iter_x(x, refrac, I_syn) / h;
				refrac += iter_refrac(refrac, x) / h;

				if (x >= 1.8)
				{
					fired = 1;
				}
			}
		}

		void init_neurons(std::vector<double> &x, std::vector<double> &refrac, std::vector<unsigned short> &fired) final
		{
			for (auto i = 0; i < x.size(); ++i)
			{
				x[i] = -1.2;
				refrac[i] = iter_refrac(-.6, x[i]);
				fired[i] = static_cast<unsigned short>(x[i] >= 1.8);
			}
		}

	private:
		[[nodiscard]] double iter_x(const double x, const double refrac, const double I_syn) const { return x - x * x * x / 3 - refrac + I_syn; }
		[[nodiscard]] double iter_refrac(const double refrac, const double x) const { return phi * (x + a - b * refrac); };

		double a{0.7};
		double b{0.8};
		double phi{0.08};
	};

	class AEIFModel : public Model_Ifc
	{
	public:
		AEIFModel() = default;
		AEIFModel(const double C, const double g_L, const double E_L, const double V_T, const double d_T, const double tau_w, const double a, const double b) : C{C}, g_L{g_L}, E_L{E_L}, V_T{V_T}, d_T{d_T}, tau_w{tau_w}, a{a}, b{b} {}

		void update_activity(double &x, double &refrac, double &I_syn, unsigned short &fired, const double h) final
		{
			for (int integration_steps = 0; integration_steps < h; ++integration_steps)
			{
				x += iter_x(x, refrac, I_syn) / h;
				refrac += iter_refrac(refrac, x) / h;

				if (x >= V_peak)
				{
					fired = 1;
					x = E_L;
					refrac += b;
					break;
				}
			}
		}

		void init_neurons(std::vector<double> &x, std::vector<double> &refrac, std::vector<unsigned short> &fired) final
		{
			for (int i = 0; i < x.size(); ++i)
			{
				x[i] = E_L;
				refrac[i] = iter_refrac(0, x[i]);
				fired[i] = static_cast<unsigned short>(x[i] >= V_peak);
			}
		}

	private:
		[[nodiscard]] double f(const double x) const { return -g_L * (x - E_L) + g_L * d_T * exp((x - V_T) / d_T); }
		[[nodiscard]] double iter_x(const double x, const double refrac, const double I_syn) const { return (f(x) - refrac + I_syn) / C; }
		[[nodiscard]] double iter_refrac(const double refrac, const double x) const { return (a * (x - E_L) - refrac) / tau_w; }

		constexpr static double V_peak{20.}; // spike trigger

		double C{281.};		// membrance capacitance
		double g_L{30.};	// leak conductance
		double E_L{-70.6};	// leak reversal potential
		double V_T{-50.4};	// spike threshold
		double d_T{2.};		// slope factor
		double tau_w{144.}; // adaptation time constant
		double a{4.};		// subthreshold
		double b{0.0805};	// spike-triggered adaptation
	};

} // namespace models

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
		[[nodiscard]] size_t size() const noexcept { return neuron_ids.size(); }

		// Resize the number of neuron ids
		void resize(size_t size) { neuron_ids.resize(size); }

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
		[[nodiscard]] size_t get_neuron_id(size_t neuron_id_index) const { return neuron_ids[neuron_id_index]; }

		// Get pointer to data
		[[nodiscard]] size_t* get_neuron_ids() noexcept { return neuron_ids.data(); }

		[[nodiscard]] size_t  get_neuron_ids_size_in_bytes() const noexcept { return neuron_ids.size() * sizeof(size_t); }

	private:
		std::vector<size_t> neuron_ids;  // Firing neuron ids
										 // This vector is used as MPI communication buffer
	};

	/**
	 * Map of (MPI rank; FiringNeuronIds)
	 * The MPI rank specifies the corresponding process
	 */
	using MapFiringNeuronIds = std::map<int, FiringNeuronIds>;

	NeuronModels(size_t num_neurons, double x_0, double tau_x, double k, double tau_C, double beta, int h, double refrac_time);
	NeuronModels(size_t num_neurons, double k, double tau_C, double beta, int h, std::unique_ptr<models::Model_Ifc> model);

	~NeuronModels() = default;

	NeuronModels(const NeuronModels& other) = delete;
	NeuronModels& operator=(const NeuronModels& other) = delete;

	NeuronModels(NeuronModels&& other) = default;
	//	: random_number_generator(other.random_number_generator)
	//{
	//	my_num_neurons = other.my_num_neurons;
	//	x_0 = other.x_0;
	//	tau_x = other.tau_x;
	//	k = other.k;
	//	tau_C = other.tau_C;
	//	beta = other.beta;
	//	refrac_time = other.refrac_time;
	//	h = other.h;
	//	x = std::move(other.x);
	//	fired = std::move(other.fired);
	//	refrac = std::move(other.refrac);
	//	I_syn = std::move(other.I_syn);
	//	random_number_distribution = std::move(other.random_number_distribution);
	//}
	NeuronModels& operator=(NeuronModels&& other) = default;

	[[nodiscard]] double get_beta() const noexcept { return beta; }

	[[nodiscard]] bool get_fired(size_t i) const { return static_cast<bool>(fired[i]); }

	[[nodiscard]] double get_x(size_t i) const { return x[i]; }

	[[nodiscard]] const std::vector<double>& get_x() const noexcept {
		return x;
	}

	[[nodiscard]] int get_refrac(size_t i) const { return u[i]; }

	/* Performs one iteration step of update in electrical activity */
	void update_electrical_activity(const NetworkGraph& network_graph, std::vector<double>& C);

private:
	// My local number of neurons
	size_t my_num_neurons;

	std::unique_ptr<models::Model_Ifc> model;

	// // Model parameters for all neurons
	double k;           // Proportionality factor for synapses in Hz
	double tau_C;       // Decay time of calcium
	double beta;        // Increase in calcium each time a neuron fires
	int    h;           // Precision for Euler integration

	// // Variables for each neuron where the array index denotes the neuron ID
	std::vector<double> x;			   // membrane potential v
	std::vector<double> u;			   // membrane recovery u
	std::vector<unsigned short> fired; // 1: neuron has fired, 0: neuron is inactive
	std::vector<double> I_syn;		   // Synaptic input
};

#endif	/* NEURONMODELS_H */
