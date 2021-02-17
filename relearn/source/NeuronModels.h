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

#include "ModelParameter.h"

#include <map>
#include <memory>
#include <vector>

class NetworkGraph;
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
            if (const bool found = find(neuron_id); !found) {
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

    NeuronModels(double k, double tau_C, double beta, unsigned int h, double background_activity, double background_activity_mean, double background_activity_stddev);

    virtual ~NeuronModels() = default;

    NeuronModels(const NeuronModels& other) = delete;
    NeuronModels& operator=(const NeuronModels& other) = delete;

    NeuronModels(NeuronModels&& other) = default;
    NeuronModels& operator=(NeuronModels&& other) = default;

    template <typename T, typename... Ts, std::enable_if_t<std::is_base_of<NeuronModels, T>::value, int> = 0>
    [[nodiscard]] static std::unique_ptr<T> create(Ts... args) {
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
    virtual void update_electrical_activity_serial_initialize() {
    }

    virtual void update_activity(size_t i) = 0;

    virtual void init_neurons() = 0;

    static constexpr double default_k{ 0.03 };
    static constexpr double default_tau_C{ 10000 }; //5000;   //very old 60.0;
    static constexpr double default_beta{ 0.001 }; //very old 0.05;
    static constexpr unsigned int default_h{ 10 };

    static constexpr double default_base_background_activity{ 0.0 };
    static constexpr double default_background_activity_mean{ 0.0 };
    static constexpr double default_background_activity_stddev{ 0.0 };

    static constexpr double min_k{ 0.0 };
    static constexpr double min_tau_C{ 0 };
    static constexpr double min_beta{ 0.0 };
    static constexpr unsigned int min_h{ 0 };

    static constexpr double min_base_background_activity{ -10000.0 };
    static constexpr double min_background_activity_mean{ -10000.0 };
    static constexpr double min_background_activity_stddev{ 0.0 };

    static constexpr double max_k{ 1.0 };
    static constexpr double max_tau_C{ 10.0e+6 };
    static constexpr double max_beta{ 1.0 };
    static constexpr unsigned int max_h{ 1000 };

    static constexpr double max_base_background_activity{ 10000.0 };
    static constexpr double max_background_activity_mean{ 10000.0 };
    static constexpr double max_background_activity_stddev{ 10000.0 };

private:
    // My local number of neurons
    size_t my_num_neurons;

    // // Model parameters for all neurons
    double k; // Proportionality factor for synapses in Hz
    double tau_C; // Decay time of calcium
    double beta; // Increase in calcium each time a neuron fires
    unsigned int h; // Precision for Euler integration

    double base_background_activity;
    double background_activity_mean;
    double background_activity_stddev;

    // Variables for each neuron where the array index denotes the neuron ID
    std::vector<double> I_syn; // Synaptic input
    std::vector<double> x; // membrane potential v
    std::vector<bool> fired; // true: neuron has fired, false: neuron is inactive

protected:
    [[nodiscard]] double get_k() const noexcept {
        return k;
    }

    [[nodiscard]] double get_tau_C() const noexcept {
        return tau_C;
    }

    [[nodiscard]] unsigned int get_h() const noexcept {
        return h;
    }

    [[nodiscard]] double get_base_background_activity() const noexcept {
        return base_background_activity;
    }

    [[nodiscard]] double get_background_activity_mean() const noexcept {
        return background_activity_mean;
    }

    [[nodiscard]] double get_background_activity_stddev() const noexcept {
        return background_activity_stddev;
    }

    [[nodiscard]] double get_I_syn(size_t i) const noexcept {
        return I_syn[i];
    }

    [[nodiscard]] size_t get_num_neurons() const noexcept {
        return my_num_neurons;
    }

    void set_x(size_t i, double new_value) noexcept {
        x[i] = new_value;
    }

    void set_fired(size_t i, bool new_value) noexcept {
        fired[i] = new_value;
    }

private:
    [[nodiscard]] static MapFiringNeuronIds update_electrical_activity_prepare_receiving_spikes(const MapFiringNeuronIds& firing_neuron_ids_outgoing);

    static void update_electrical_activity_exchange_neuron_ids(const MapFiringNeuronIds& firing_neuron_ids_outgoing, MapFiringNeuronIds& firing_neuron_ids_incoming);

    [[nodiscard]] MapFiringNeuronIds update_electrical_activity_prepare_sending_spikes(const NetworkGraph& network_graph);

    void update_electrical_activity_update_activity_and_calcium(std::vector<double>& C);

    void update_electrical_activity_calculate_input(const NetworkGraph& network_graph, const MapFiringNeuronIds& firing_neuron_ids_incoming);

    void update_electrical_activity_calculate_background();
};

namespace models {
class ModelA : public NeuronModels {
public:
    explicit ModelA(
        double k = NeuronModels::default_k,
        double tau_C = NeuronModels::default_tau_C,
        double beta = NeuronModels::default_beta,
        unsigned int h = NeuronModels::default_h,
        double background_activity = NeuronModels::default_base_background_activity,
        double background_activity_mean = NeuronModels::default_background_activity_mean,
        double background_activity_stddev = NeuronModels::default_background_activity_stddev,
        double x_0 = ModelA::default_x_0,
        double tau_x = ModelA::default_tau_x,
        unsigned int refrac_time = ModelA::default_refrac_time);

    [[nodiscard]] std::unique_ptr<NeuronModels> clone() const final;

    [[nodiscard]] double get_secondary_variable(size_t i) const noexcept final;

    [[nodiscard]] std::vector<ModelParameter> get_parameter() final;

    [[nodiscard]] std::string name() final;

    void init(size_t num_neurons) final;

protected:
    void update_electrical_activity_serial_initialize() final;

    void update_activity(size_t i) final;

    void init_neurons() final;

private:
    [[nodiscard]] double iter_x(double x, double I_syn) const noexcept;

    static constexpr double default_x_0{ 0.05 };
    static constexpr double default_tau_x{ 5.0 };
    static constexpr unsigned int default_refrac_time{ 4 };

    static constexpr double min_x_0{ 0.0 };
    static constexpr double min_tau_x{ 0.0 };
    static constexpr unsigned int min_refrac_time{ 0 };

    static constexpr double max_x_0{ 1.0 };
    static constexpr double max_tau_x{ 1000.0 };
    static constexpr unsigned int max_refrac_time{ 1000 };

    std::vector<unsigned int> refrac; // refractory time

    std::vector<double> theta_values;

    double x_0; // Background or resting activity
    double tau_x; // Decay time of firing rate in msec
    unsigned int refrac_time; // Length of refractory period in msec. After an action potential a neuron cannot fire for this time
};

class IzhikevichModel : public NeuronModels {
public:
    explicit IzhikevichModel(
        double k = NeuronModels::default_k,
        double tau_C = NeuronModels::default_tau_C,
        double beta = NeuronModels::default_beta,
        unsigned int h = NeuronModels::default_h,
        double background_activity = NeuronModels::default_base_background_activity,
        double background_activity_mean = NeuronModels::default_background_activity_mean,
        double background_activity_stddev = NeuronModels::default_background_activity_stddev,
        double a = IzhikevichModel::default_a,
        double b = IzhikevichModel::default_b,
        double c = IzhikevichModel::default_c,
        double d = IzhikevichModel::default_d,
        double V_spike = IzhikevichModel::default_V_spike,
        double k1 = IzhikevichModel::default_k1,
        double k2 = IzhikevichModel::default_k2,
        double k3 = IzhikevichModel::default_k3);

    [[nodiscard]] std::unique_ptr<NeuronModels> clone() const final;

    [[nodiscard]] double get_secondary_variable(size_t i) const noexcept final;

    [[nodiscard]] std::vector<ModelParameter> get_parameter() final;

    [[nodiscard]] std::string name() final;

    void init(size_t num_neurons) final;

protected:
    void update_activity(size_t i) final;

    void init_neurons() final;

private:
    [[nodiscard]] double iter_x(double x, double u, double I_syn) const noexcept;

    [[nodiscard]] double iter_refrac(double u, double x) const noexcept;

    [[nodiscard]] bool spiked(double x) const noexcept;

    static constexpr double default_a{ 0.1 };
    static constexpr double default_b{ 0.2 };
    static constexpr double default_c{ -65.0 };
    static constexpr double default_d{ 2.0 };
    static constexpr double default_V_spike{ 30.0 };
    static constexpr double default_k1{ 0.04 };
    static constexpr double default_k2{ 5.0 };
    static constexpr double default_k3{ 140.0 };

    static constexpr double min_a{ 0.0 };
    static constexpr double min_b{ 0.0 };
    static constexpr double min_c{ -150.0 };
    static constexpr double min_d{ 0.0 };
    static constexpr double min_V_spike{ 0.0 };
    static constexpr double min_k1{ 0.0 };
    static constexpr double min_k2{ 0.0 };
    static constexpr double min_k3{ 50.0 };

    static constexpr double max_a{ 1.0 };
    static constexpr double max_b{ 1.0 };
    static constexpr double max_c{ -50.0 };
    static constexpr double max_d{ 10.0 };
    static constexpr double max_V_spike{ 100.0 };
    static constexpr double max_k1{ 1.0 };
    static constexpr double max_k2{ 10.0 };
    static constexpr double max_k3{ 200.0 };

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
        double background_activity = NeuronModels::default_base_background_activity,
        double background_activity_mean = NeuronModels::default_background_activity_mean,
        double background_activity_stddev = NeuronModels::default_background_activity_stddev,
        double a = FitzHughNagumoModel::default_a,
        double b = FitzHughNagumoModel::default_b,
        double phi = FitzHughNagumoModel::default_phi);

    [[nodiscard]] std::unique_ptr<NeuronModels> clone() const final;

    [[nodiscard]] double get_secondary_variable(size_t i) const noexcept final;

    [[nodiscard]] std::vector<ModelParameter> get_parameter() final;

    [[nodiscard]] std::string name() final;

    void init(size_t num_neurons) final;

protected:
    void update_activity(size_t i) final;

    void init_neurons() final;

private:
    [[nodiscard]] static double iter_x(double x, double w, double I_syn) noexcept;

    [[nodiscard]] double iter_refrac(double w, double x) const noexcept;

    [[nodiscard]] static bool spiked(double x, double w) noexcept;

    static constexpr double default_a{ 0.7 };
    static constexpr double default_b{ 0.8 };
    static constexpr double default_phi{ 0.08 };

    static constexpr double min_a{ 0.7 };
    static constexpr double min_b{ 0.8 };
    static constexpr double min_phi{ 0.08 };

    static constexpr double max_a{ 0.7 };
    static constexpr double max_b{ 0.8 };
    static constexpr double max_phi{ 0.08 };

    static constexpr double init_x{ -1.2 };
    static constexpr double init_w{ -0.6 };

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
        double background_activity = NeuronModels::default_base_background_activity,
        double background_activity_mean = NeuronModels::default_background_activity_mean,
        double background_activity_stddev = NeuronModels::default_background_activity_stddev,
        double C = AEIFModel::default_C,
        double g_L = AEIFModel::default_g_L,
        double E_L = AEIFModel::default_E_L,
        double V_T = AEIFModel::default_V_T,
        double d_T = AEIFModel::default_d_T,
        double tau_w = AEIFModel::default_tau_w,
        double a = AEIFModel::default_a,
        double b = AEIFModel::default_b,
        double V_peak = AEIFModel::default_V_peak);

    [[nodiscard]] std::unique_ptr<NeuronModels> clone() const final;

    [[nodiscard]] double get_secondary_variable(size_t i) const noexcept final;

    [[nodiscard]] std::vector<ModelParameter> get_parameter() final;

    [[nodiscard]] std::string name() final;

    void init(size_t num_neurons) final;

protected:
    void update_activity(size_t i) final;

    void init_neurons() final;

private:
    [[nodiscard]] double f(double x) const noexcept;

    [[nodiscard]] double iter_x(double x, double w, double I_syn) const noexcept;

    [[nodiscard]] double iter_refrac(double w, double x) const noexcept;

    static constexpr double default_C{ 281.0 };
    static constexpr double default_g_L{ 30.0 };
    static constexpr double default_E_L{ -70.6 };
    static constexpr double default_V_T{ -50.4 };
    static constexpr double default_d_T{ 2.0 };
    static constexpr double default_tau_w{ 144.0 };
    static constexpr double default_a{ 4.0 };
    static constexpr double default_b{ 0.0805 };
    static constexpr double default_V_peak{ 20.0 };

    static constexpr double min_C{ 100.0 };
    static constexpr double min_g_L{ 0.0 };
    static constexpr double min_E_L{ -150.0 };
    static constexpr double min_V_T{ -150.0 };
    static constexpr double min_d_T{ 0.0 };
    static constexpr double min_tau_w{ 100.0 };
    static constexpr double min_a{ 0.0 };
    static constexpr double min_b{ 0.0 };
    static constexpr double min_V_peak{ 0.0 };

    static constexpr double max_C{ 500.0 };
    static constexpr double max_g_L{ 100.0 };
    static constexpr double max_E_L{ -20.0 };
    static constexpr double max_V_T{ 0.0 };
    static constexpr double max_d_T{ 10.0 };
    static constexpr double max_tau_w{ 200.0 };
    static constexpr double max_a{ 10.0 };
    static constexpr double max_b{ 0.3 };
    static constexpr double max_V_peak{ 70.0 };

    std::vector<double> w; // adaption variable

    double C; // membrance capacitance
    double g_L; // leak conductance
    double E_L; // leak reversal potential
    double V_T; // spike threshold
    double d_T; // slope factor
    double tau_w; // adaptation time constant
    double a; // subthreshold
    double b; // spike-triggered adaptation

    double V_peak; // spike trigger
};

} // namespace models
