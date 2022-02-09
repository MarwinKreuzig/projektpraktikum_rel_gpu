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

#include "../../mpi/CommunicationMap.h"
#include "../../util/RelearnException.h"
#include "../../util/TaggedID.h"
#include "../UpdateStatus.h"
#include "ModelParameter.h"

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

class NetworkGraph;
class NeuronMonitor;

/**
 * This class provides the basic interface for every neuron model, that is, the rules by which a neuron spikes.
 * The calculations should focus solely on the spiking behavior, and should not account for any plasticity changes.
 * The object itself stores only the local portion of the neuron population.
 * This class performs communication with MPI.
 */
class NeuronModel {
    friend class NeuronMonitor;

public:
    /**
     * @brief Construcs a new instance of type NeuronModel with 0 neurons and default values for all parameters
     */
    NeuronModel() = default;

    /**
     * @brief Construcs a new instance of type NeuronModel with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters agains the min and max values defined below in order to allow other values besides in the GUI
     * @param k The factor by which the input of a neighboring spiking neuron is weighted
     * @param tau_C The dampening factor by which the Calcium decreases in Neurons. @Todo(Fabian): Move that parameter into Neurons
     * @param beta The factor by which the Calcium is increased whenever a neuron spikes in Neurons. @Todo(Fabian): Move that parameter into Neurons
     * @param h The step size for the numerical integration
     * @param base_background_activity The base background activity that all neurons are exited with. Is only used if background_activity_stddev > 0.0
     * @param background_activity_mean The mean of background activity taht all neurons are exited with. Is only used if background_activity_stddev > 0.0
     * @param background_activity_stddev The standard deviation of background activity taht all neurons are exited with. Is only used if background_activity_stddev > 0.0
     *
     * If background_activity_stddev > 0.0, all neurons are exited with
     *      base_background_activity + N(background_activity_mean, background_activity_stddev)
     */
    NeuronModel(
        double k,
        double tau_C,
        double beta,
        unsigned int h,
        double base_background_activity,
        double background_activity_mean,
        double background_activity_stddev);

    virtual ~NeuronModel() = default;

    NeuronModel(const NeuronModel& other) = delete;
    NeuronModel& operator=(const NeuronModel& other) = delete;

    NeuronModel(NeuronModel&& other) = default;
    NeuronModel& operator=(NeuronModel&& other) = default;

    /**
     * @brief Creates an object of type T wrapped inside an std::unique_ptr
     * @param ...args The arguments that shall be passed to the constructor of T
     * @tparam T The type of NeuronModel that shall be constructed, must inherit from NeuronModel
     * @tparam ...Ts The types of parameters for the constructor of T
     * @return A new instance of type T wrapped inside an std::unique_ptr
     */
    template <typename T, typename... Ts, std::enable_if_t<std::is_base_of<NeuronModel, T>::value, int> = 0>
    [[nodiscard]] static std::unique_ptr<T> create(Ts... args) {
        return std::make_unique<T>(args...);
    }

    /**
     * @brief Provides a way to clone the current NeuronModel, i.e., all parameters.
     *      The returned object shares all parameters, but has 0 neurons.
     *      Because of inheritance-shenanigans, the return value might need to be casted
     * @return A new instance of the class with the same parameters wrapped inside an std::unique_ptr
     */
    [[nodiscard]] virtual std::unique_ptr<NeuronModel> clone() const = 0;

    /**
     * @brief Returns beta (The factor by which the Calcium is increased whenever a neuron spikes in Neurons)
     * @return Beta (The factor by which the Calcium is increased whenever a neuron spikes in Neurons)
     */
    [[nodiscard]] double get_beta() const noexcept {
        return beta;
    }

    /**
     * @brief Returns a bool that indicates if the neuron with the passed local id spiked in the current simulation step
     * @param neuron_id The local neuron id that should be queried
     * @exception Throws a RelearnException if neuron_id is too large
     * @return True iff the neuron spiked
     */
    [[nodiscard]] bool get_fired(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_local_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::get_fired: id is too large: {}", neuron_id);
        return fired[local_neuron_id] == 1;
    }

    /**
     * @brief Returns a vector of flags that indicate if the neuron with the local id spiked in the current simulation step
     * @return A constant reference to the vector of flags. It is not invalidated by calls to other methods
     */
    [[nodiscard]] const std::vector<char>& get_fired() const noexcept {
        return fired;
    }

    /**
     * @brief Returns a double that indicates the neuron's membrane potential in the current simulation step
     * @param neuron_id The local neuron id that should be queried
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The neuron's membrane potential
     */
    [[nodiscard]] double get_x(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_local_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::get_x: id is too large: {}", neuron_id);
        return x[local_neuron_id];
    }

    /**
     * @brief Returns a vector of doubles that indicate the neurons' respective membrane potential in the current simulation step
     * @return A constant reference to the vector of doubles. It is not invalidated by calls to other methods
     */
    [[nodiscard]] const std::vector<double>& get_x() const noexcept {
        return x;
    }

    /**
     * @brief Returns k (The factor by which the input of a neighboring spiking neuron is weighted)
     * @return k (The factor by which the input of a neighboring spiking neuron is weighted)
     */
    [[nodiscard]] double get_k() const noexcept {
        return k;
    }

    /**
     * @brief Returns tau_C (The dampening factor by which the Calcium decreases in Neurons)
     * @return tau_C (The dampening factor by which the Calcium decreases in Neurons)
     */
    [[nodiscard]] double get_tau_C() const noexcept {
        return tau_C;
    }

    /**
     * @brief Returns the numerical integration's step size
     * @return The step size
     */
    [[nodiscard]] unsigned int get_h() const noexcept {
        return h;
    }

    /**
     * @brief Returns the base background activity
     * @return The base background activity
     */
    [[nodiscard]] double get_base_background_activity() const noexcept {
        return base_background_activity;
    }

    /**
     * @brief Returns the mean background activity
     * @return The mean background activity
     */
    [[nodiscard]] double get_background_activity_mean() const noexcept {
        return background_activity_mean;
    }

    /**
     * @brief Returns the standard deviation of the background activity
     * @return The standard deviation of the background activity
     */
    [[nodiscard]] double get_background_activity_stddev() const noexcept {
        return background_activity_stddev;
    }

    /**
     * @brief Returns the synaptic input the specified neuron receives in the current simulation step
     * @param neuron_id The local neuron id for the neuron that should be queried
     * @exception Throws a RelearnException if neuron_id is too large
     * @return A double that indicates the synaptic input for the specified neuron
     */
    [[nodiscard]] double get_I_syn(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_local_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::get_I_syn: id is too large: {}", neuron_id);
        return I_syn[local_neuron_id];
    }

    /**
     * @brief Returns a vector of doubles that indicate the neurons' respective synaptic input in the current simulation step
     * @return A constant reference to the vector of doubles. It is not invalidated by calls to other methods
     */
    [[nodiscard]] const std::vector<double>& get_I_syn() const noexcept {
        return I_syn;
    }

    /**
     * @brief Returns the number of neurons that are stored in the object
     * @return The number of neurons that are stored in the object
     */
    [[nodiscard]] size_t get_num_neurons() const noexcept {
        return number_local_neurons;
    }

    /**
     * @brief Returns the secondary variable used for computation of the electrical activity.
     *      The meaning of the variable can vary between classes that inherit from NeuronModels
     * @param neuron_id The local neuron id for the neuron that should be queried
     * @exception Throws a RelearnException if neuron_id is too large
     * @return A double that indicates the secondary variable for the specified neuron
     */
    [[nodiscard]] virtual double get_secondary_variable(const NeuronID& neuron_id) const = 0;

    /**
     * @brief Performs one step of simulating the electrical activity for all neurons.
     *      This method performs communication via MPI.
     * @param network_graph The network graph that specifies which neurons are connected. Is used to determine which spikes effect the local portion.
     * @param disable_flags A vector of flags that specify which neurons should be left alone during the update
     */
    void update_electrical_activity(const NetworkGraph& network_graph, const std::vector<UpdateStatus>& disable_flags);

    /**
     * @brief Returns a vector with an std::unique_ptr for each class inherited from NeuronModels which can be cloned
     * @return A vector with all inherited classes
     */
    [[nodiscard]] static std::vector<std::unique_ptr<NeuronModel>> get_models();

    /**
     * @brief Returns a vector with all adjustable ModelParameter
     * @return A vector with all adjustable ModelParameter
     */
    [[nodiscard]] virtual std::vector<ModelParameter> get_parameter();

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     *      Sets the initial membrane potential and initial synaptic inputs to 0.0 and fired to false
     * @param number_neurons The number of local neurons to store in this class
     */
    virtual void init(size_t number_neurons);

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    virtual void create_neurons(size_t creation_count);

    /**
     * @brief Returns the name of the current model
     * @return The name of the current model
     */
    [[nodiscard]] virtual std::string name() = 0;

    /**
     * @brief Performs all required steps to disable all neurons that are specified.
     *      Disables incrementally, i.e., previously disabled neurons are not enabled.
     * @param neuron_ids The local neuron ids that should be disabled
     * @exception Throws a RelearnException if a specified id is too large
     */
    virtual void disable_neurons(const std::vector<NeuronID>& neuron_ids) {
        for (const NeuronID& neuron_id : neuron_ids) {
            const auto local_neuron_id = neuron_id.get_local_id();

            RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::disable_neurons: There is a too large id: {} vs {}", neuron_id, number_local_neurons);
            fired[local_neuron_id] = 0;
        }
    }

    static constexpr double default_k{ 0.03 };
    static constexpr double default_tau_C{ 10000 }; // 5000;   //very old 60.0;
    static constexpr double default_beta{ 0.001 }; // very old 0.05;
    static constexpr unsigned int default_h{ 10 };

    static constexpr double default_base_background_activity{ 0.0 };
    static constexpr double default_background_activity_mean{ 0.0 };
    static constexpr double default_background_activity_stddev{ 0.0 };

    static constexpr double min_k{ 0.0 };
    static constexpr double min_tau_C{ 0 };
    static constexpr double min_beta{ 0.0 };
    static constexpr unsigned int min_h{ 1 };

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

protected:
    /**
     * @brief This method is called as part of the update of the electrical activity.
     *      It provides a hook to initialize variables serially
     * @param disable_flags The local neuron ids that should be skipped
     */
    virtual void update_electrical_activity_serial_initialize(const std::vector<UpdateStatus>& disable_flags) {
    }

    /**
     * @brief Provides a hook to update the activity of the neuron with the passed id
     *      If OpenMP is activated, this is called in parallel for multiple ids
     * @param neuron_id The local neuron id that should be updated
     */
    virtual void update_activity(const NeuronID& neuron_id) = 0;

    /**
     * @brief Provides a hook to initialize all neurons with local id in [start_id, end_id)
     *      This method exists because of the order of operations when creating neurons
     * @param start_id The first local neuron id to initialize
     * @param end_id The next to last local neuron id to initialize
     */
    virtual void init_neurons(size_t start_id, size_t end_id) = 0;

    /**
     * @brief Sets the membrane potential for the specified neuron. Does not perform bound-checking
     * @param neuron_id The local neuron id
     * @param new_value The new membrane potential
     */
    void set_x(const NeuronID& neuron_id, const double new_value) noexcept {
        const auto local_neuron_id = neuron_id.get_local_id();
        x[local_neuron_id] = new_value;
    }

    /**
     * @brief Sets if a neuron fired for the specified neuron. Does not perform bound-checking
     * @param neuron_id The local neuron id
     * @param new_value True iff the neuron fired in the current simulation step
     */
    void set_fired(const NeuronID& neuron_id, const char new_value) noexcept {
        const auto local_neuron_id = neuron_id.get_local_id();
        fired[local_neuron_id] = new_value;
    }

private:
    [[nodiscard]] CommunicationMap<NeuronID> update_electrical_activity_prepare_sending_spikes(const NetworkGraph& network_graph, const std::vector<UpdateStatus>& disable_flags);

    void update_electrical_activity_update_activity(const std::vector<UpdateStatus>& disable_flags);

    void update_electrical_activity_calculate_input(const NetworkGraph& network_graph, const CommunicationMap<NeuronID>& firing_neuron_ids_incoming, const std::vector<UpdateStatus>& disable_flags);

    void update_electrical_activity_calculate_background(const std::vector<UpdateStatus>& disable_flags);

    // My local number of neurons
    size_t number_local_neurons{ 0 };

    // Model parameters for all neurons
    double k{ default_k }; // Proportionality factor for synapses in Hz
    double tau_C{ default_tau_C }; // Decay time of calcium
    double beta{ default_beta }; // Increase in calcium each time a neuron fires
    unsigned int h{ default_h }; // Precision for Euler integration

    double base_background_activity{ default_base_background_activity };
    double background_activity_mean{ default_background_activity_mean };
    double background_activity_stddev{ default_background_activity_stddev };

    // Variables for each neuron where the array index denotes the local neuron ID
    std::vector<double> I_syn{}; // Synaptic input
    std::vector<double> x{}; // membrane potential v
    std::vector<char> fired{}; // 0: neuron is inactive, != 0: neuron fired
};

namespace models {
/**
 * This class inherits from NeuronModel and implements a poisson spiking model
 */
class PoissonModel : public NeuronModel {
public:
    /**
     * @brief Construcs a new instance of type PoissonModel with 0 neurons and default values for all parameters
     */
    PoissonModel() = default;

    /**
     * @brief Construcs a new instance of type PoissonModel with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters agains the min and max values defined below in order to allow other values besides in the GUI
     * @param k See NeuronModel(...)
     * @param tau_C See NeuronModel(...)
     * @param beta See NeuronModel(...)
     * @param h See NeuronModel(...)
     * @param base_background_activity See NeuronModel(...)
     * @param background_activity_mean See NeuronModel(...)
     * @param background_activity_stddev See NeuronModel(...)
     * @param x_0 The resting membrane potential
     * @param tau_x The dampening factor by which the membrane potential decreases
     * @param refrac_time The number of steps a neuron doesn't spike after spiking
     */
    PoissonModel(
        double k,
        double tau_C,
        double beta,
        unsigned int h,
        double base_background_activity,
        double background_activity_mean,
        double background_activity_stddev,
        double x_0,
        double tau_x,
        unsigned int refrac_time);

    /**
     * @brief Clones this instance and creates a new PoissonModel with the same parameters and 0 local neurons
     */
    [[nodiscard]] std::unique_ptr<NeuronModel> clone() const final;

    /**
     * @brief Returns the refrac time (The number of steps a neuron doesn't spike after spiking)
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The refrac time (The number of steps a neuron doesn't spike after spiking)
     */
    [[nodiscard]] double get_secondary_variable(const NeuronID& neuron_id) const final {
        const auto local_neuron_id = neuron_id.get_local_id();

        RelearnException::check(local_neuron_id < get_num_neurons(), "PoissonModel::get_secondary_variable: id is too large: {}", neuron_id);
        return refrac[local_neuron_id];
    }

    /**
     * @brief Returns a vector with all adjustable ModelParameter for this class and NeuronModel
     * @return A vector with all adjustable ModelParameter
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() final;

    /**
     * @brief Returns the name of this model
     * @return The name of this model
     */
    [[nodiscard]] std::string name() final;

    /**
     * @brief Returns x_0 (The resting membrane potential)
     * @return x_0 (The resting membrane potential)
     */
    [[nodiscard]] double get_x_0() const noexcept {
        return x_0;
    }

    /**
     * @brief Returns tau_x (The dampening factor by which the membrane potential decreases)
     * @return tau_x (The dampening factor by which the membrane potential decreases)
     */
    [[nodiscard]] double get_tau_x() const noexcept {
        return tau_x;
    }

    /**
     * @brief Returns refrac_time (The number of steps a neuron doesn't spike after spiking)
     * @return refrac_time (The number of steps a neuron doesn't spike after spiking)
     */
    [[nodiscard]] unsigned int get_refrac_time() const noexcept {
        return refrac_time;
    }

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     *      Sets the initial refrac counter to 0
     * @param number_neurons The number of local neurons to store in this class
     */
    void init(size_t number_neurons) final;

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void create_neurons(size_t creation_count) final;

    static constexpr double default_x_0{ 0.05 };
    static constexpr double default_tau_x{ 5.0 };
    static constexpr unsigned int default_refrac_time{ 4 };

    static constexpr double min_x_0{ 0.0 };
    static constexpr double min_tau_x{ 0.0 };
    static constexpr unsigned int min_refrac_time{ 0 };

    static constexpr double max_x_0{ 1.0 };
    static constexpr double max_tau_x{ 1000.0 };
    static constexpr unsigned int max_refrac_time{ 1000 };

protected:
    void update_electrical_activity_serial_initialize(const std::vector<UpdateStatus>& disable_flags) final;

    void update_activity(const NeuronID& neuron_id) final;

    void init_neurons(size_t start_id, size_t end_id) final;

private:
    [[nodiscard]] double iter_x(const double x, const double I_syn) const noexcept {
        return ((x_0 - x) / tau_x + I_syn);
    }

    std::vector<unsigned int> refrac{}; // refractory time

    std::vector<double> theta_values{};

    double x_0{ default_x_0 }; // Background or resting activity
    double tau_x{ default_tau_x }; // Decay time of firing rate in msec
    unsigned int refrac_time{ default_refrac_time }; // Length of refractory period in msec. After an action potential a neuron cannot fire for this time
};

/**
 * This class inherits from NeuronModel and implements the spiking model from Izhikevich.
 * The differential equations are:
 *      d/dt v(t) = k1 * v(t)^2 + k2 * v(t) + k3 - u(t) + I_syn
 *      d/dt u(t) = a * (b * x - u(t))
 * If v(t) >= V_spike:
 *      v(t) = c
 *      u(t) += d
 */
class IzhikevichModel : public NeuronModel {
public:
    /**
     * @brief Construcs a new instance of type IzhikevichModel with 0 neurons and default values for all parameters
     */
    IzhikevichModel() = default;

    /**
     * @brief Construcs a new instance of type IzhikevichModel with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters agains the min and max values defined below in order to allow other values besides in the GUI
     * @param k See NeuronModel(...)
     * @param tau_C See NeuronModel(...)
     * @param beta See NeuronModel(...)
     * @param h See NeuronModel(...)
     * @param base_background_activity See NeuronModel(...)
     * @param background_activity_mean See NeuronModel(...)
     * @param background_activity_stddev See NeuronModel(...)
     * @param a The dampening factor for u(t)
     * @param b The dampening factor for v(t) inside the equation for d/dt u(t)
     * @param c The reset activity
     * @param d The additional dampening for u(t) in case of spiking
     * @param V_spike The spiking threshold
     * @param k1 The factor for v(t)^2 inside the equation for d/dt v(t)
     * @param k2 The factor for v(t) inside the equation for d/dt v(t)
     * @param k3 The constant inside the equation for d/dt v(t)
     */
    IzhikevichModel(
        double k,
        double tau_C,
        double beta,
        unsigned int h,
        double base_background_activity,
        double background_activity_mean,
        double background_activity_stddev,
        double a,
        double b,
        double c,
        double d,
        double V_spike,
        double k1,
        double k2,
        double k3);

    /**
     * @brief Clones this instance and creates a new IzhikevichModel with the same parameters and 0 local neurons
     */
    [[nodiscard]] std::unique_ptr<NeuronModel> clone() const final;

    /**
     * @brief Returns the dampending variable u
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The dampending variable u
     */
    [[nodiscard]] double get_secondary_variable(const NeuronID& neuron_id) const final {
        const auto local_neuron_id = neuron_id.get_local_id();

        RelearnException::check(local_neuron_id < get_num_neurons(), "IzhikevichModel::get_secondary_variable: id is too large: {}", neuron_id);
        return u[local_neuron_id];
    }

    /**
     * @brief Returns a vector with all adjustable ModelParameter for this class and NeuronModel
     * @return A vector with all adjustable ModelParameter
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() final;

    /**
     * @brief Returns the name of this model
     * @return The name of this model
     */
    [[nodiscard]] std::string name() final;

    /**
     * @brief Returns a (The dampening factor for u(t))
     * @return a (The dampening factor for u(t))
     */
    [[nodiscard]] double get_a() const noexcept {
        return a;
    }

    /**
     * @brief Returns b (The dampening factor for v(t) inside the equation for d/dt u(t))
     * @return b (The dampening factor for v(t) inside the equation for d/dt u(t))
     */
    [[nodiscard]] double get_b() const noexcept {
        return b;
    }

    /**
     * @brief Returns c (The reset activity)
     * @return c (The reset activity)
     */
    [[nodiscard]] double get_c() const noexcept {
        return c;
    }

    /**
     * @brief Returns d (The additional dampening for u(t))
     * @return d (The additional dampening for u(t))
     */
    [[nodiscard]] double get_d() const noexcept {
        return d;
    }

    /**
     * @brief Returns V_spike (The spiking threshold)
     * @return V_spike (The spiking threshold)
     */
    [[nodiscard]] double get_V_spike() const noexcept {
        return V_spike;
    }

    /**
     * @brief Returns k1 (The factor for v(t)^2 inside the equation for d/dt v(t))
     * @return k1 (The factor for v(t)^2 inside the equation for d/dt v(t))
     */
    [[nodiscard]] double get_k1() const noexcept {
        return k1;
    }

    /**
     * @brief Returns k2 (The factor for v(t) inside the equation for d/dt v(t))
     * @return k2 (The factor for v(t) inside the equation for d/dt v(t))
     */
    [[nodiscard]] double get_k2() const noexcept {
        return k2;
    }

    /**
     * @brief Returns k3 (The constant inside the equation for d/dt v(t))
     * @return k3 (The constant inside the equation for d/dt v(t))
     */
    [[nodiscard]] double get_k3() const noexcept {
        return k3;
    }

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     * @param number_neurons The number of local neurons to store in this class
     */
    void init(size_t number_neurons) final;

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void create_neurons(size_t creation_count) final;

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

protected:
    void update_activity(const NeuronID& neuron_id) final;

    void init_neurons(size_t start_id, size_t end_id) final;

private:
    [[nodiscard]] double iter_x(double x, double u, double I_syn) const noexcept;

    [[nodiscard]] double iter_refrac(double u, double x) const noexcept;

    [[nodiscard]] bool spiked(double x) const noexcept;

    std::vector<double> u{}; // membrane recovery

    double a{ default_a }; // time-scale of membrane recovery u
    double b{ default_b }; // sensitivity of membrane recovery to membrane potential v (x)
    double c{ default_c }; // after-spike reset value for membrane potential v (x)
    double d{ default_d }; // after-spike reset of membrane recovery u

    double V_spike{ default_V_spike };

    double k1{ default_k1 };
    double k2{ default_k2 };
    double k3{ default_k3 };
};

/**
 * This class inherits from NeuronModel and implements the spiking model from Fitz, Hugh, Nagumo.
 * The differential equations are:
 *      d/dt v(t) = v(t) - (v(t)^3)/3 - w(t) + I_syn
 *      d/dt w(t) = phi * (v(t) + a - b * w(t))
 */
class FitzHughNagumoModel : public NeuronModel {
public:
    /**
     * @brief Construcs a new instance of type FitzHughNagumoModel with 0 neurons and default values for all parameters
     */
    FitzHughNagumoModel() = default;

    /**
     * @brief Construcs a new instance of type IzhikevichModel with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters agains the min and max values defined below in order to allow other values besides in the GUI
     * @param k See NeuronModel(...)
     * @param tau_C See NeuronModel(...)
     * @param beta See NeuronModel(...)
     * @param h See NeuronModel(...)
     * @param base_background_activity See NeuronModel(...)
     * @param background_activity_mean See NeuronModel(...)
     * @param background_activity_stddev See NeuronModel(...)
     * @param a The constant inside the equation for d/dt w(t)
     * @param b The dampening factor for w(t) inside the equation for d/dt w(t)
     * @param phi The dampening factor for w(t)
     */
    FitzHughNagumoModel(
        double k,
        double tau_C,
        double beta,
        unsigned int h,
        double base_background_activity,
        double background_activity_mean,
        double background_activity_stddev,
        double a,
        double b,
        double phi);

    /**
     * @brief Clones this instance and creates a new FitzHughNagumoModel with the same parameters and 0 local neurons
     */
    [[nodiscard]] std::unique_ptr<NeuronModel> clone() const final;

    /**
     * @brief Returns the dampending variable w
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The dampending variable w
     */
    [[nodiscard]] double get_secondary_variable(const NeuronID& neuron_id) const final {
        const auto local_neuron_id = neuron_id.get_local_id();

        RelearnException::check(local_neuron_id < get_num_neurons(), "In FitzHughNagumoModel::get_secondary_variable, id is too large");
        return w[local_neuron_id];
    }

    /**
     * @brief Returns a vector with all adjustable ModelParameter for this class and NeuronModel
     * @return A vector with all adjustable ModelParameter
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() final;

    /**
     * @brief Returns the name of this model
     * @return The name of this model
     */
    [[nodiscard]] std::string name() final;

    /**
     * @brief Returns k3 ()
     * @return k3 ()
     */
    [[nodiscard]] double get_a() const noexcept {
        return a;
    }

    /**
     * @brief Returns k3 ()
     * @return k3 ()
     */
    [[nodiscard]] double get_b() const noexcept {
        return b;
    }

    /**
     * @brief Returns k3 ()
     * @return k3 ()
     */
    [[nodiscard]] double get_phi() const noexcept {
        return phi;
    }

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     * @param number_neurons The number of local neurons to store in this class
     */
    void init(size_t number_neurons) final;

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void create_neurons(size_t creation_count) final;

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

protected:
    void update_activity(const NeuronID& neuron_id) final;

    void init_neurons(size_t start_id, size_t end_id) final;

private:
    [[nodiscard]] static double iter_x(double x, double w, double I_syn) noexcept;

    [[nodiscard]] double iter_refrac(double w, double x) const noexcept;

    [[nodiscard]] static bool spiked(double x, double w) noexcept;

    std::vector<double> w{}; // recovery variable

    double a{ default_a };
    double b{ default_b };
    double phi{ default_phi };
};

/**
 * This class inherits from NeuronModel and implements an exponential spiking model from Brette and Gerstner.
 * The differential equations are:
 *      d/dt v(t) = (-g_L * (v(t) - E_L) + g_L * d_T * exp((v(t) - V_T) / d_T) - w(t) + I_syn) / C
 *      d/dt w(t) = (a * (v(t) - E_L) - w(t)) / tau_W
 * If v(t) >= V_spike:
 *      v(t) = E_L
 *      w(t) += b
 */
class AEIFModel : public NeuronModel {
public:
    /**
     * @brief Construcs a new instance of type AEIFModel with 0 neurons and default values for all parameters
     */
    AEIFModel() = default;

    /**
     * @brief Construcs a new instance of type IzhikevichModel with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters agains the min and max values defined below in order to allow other values besides in the GUI
     * @param k See NeuronModel(...)
     * @param tau_C See NeuronModel(...)
     * @param beta See NeuronModel(...)
     * @param h See NeuronModel(...)
     * @param base_background_activity See NeuronModel(...)
     * @param background_activity_mean See NeuronModel(...)
     * @param background_activity_stddev See NeuronModel(...)
     * @param C The dampening factor for v(t) (membrane capacitance)
     * @param g_T The leak conductance
     * @param E_L The reset membrane potential (leak reversal potential)
     * @param V_T The spiking threshold in the equation
     * @param d_T The slope factor
     * @param tau_w The dampening factor for w(t)
     * @param a The subthreshold adaptation
     * @param b The additional dampening for w(t) in case of spiking
     * @param V_spike The spiking threshold in the spiking check
     */
    AEIFModel(
        double k,
        double tau_C,
        double beta,
        unsigned int h,
        double base_background_activity,
        double background_activity_mean,
        double background_activity_stddev,
        double C,
        double g_L,
        double E_L,
        double V_T,
        double d_T,
        double tau_w,
        double a,
        double b,
        double V_spike);

    /**
     * @brief Clones this instance and creates a new AEIFModel with the same parameters and 0 local neurons
     */
    [[nodiscard]] std::unique_ptr<NeuronModel> clone() const final;

    /**
     * @brief Returns the dampending variable w
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The dampending variable w
     */
    [[nodiscard]] double get_secondary_variable(const NeuronID& neuron_id) const final {
        const auto local_neuron_id = neuron_id.get_local_id();

        RelearnException::check(local_neuron_id < get_num_neurons(), "In AEIFModel::get_secondary_variable, id is too large");
        return w[local_neuron_id];
    }

    /**
     * @brief Returns a vector with all adjustable ModelParameter for this class and NeuronModel
     * @return A vector with all adjustable ModelParameter
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() final;

    /**
     * @brief Returns the name of this model
     * @return The name of this model
     */
    [[nodiscard]] std::string name() final;

    /**
     * @brief Returns C (The dampening factor for v(t) (membrane capacitance))
     * @return C (The dampening factor for v(t) (membrane capacitance))
     */
    [[nodiscard]] double get_C() const noexcept {
        return C;
    }

    /**
     * @brief Returns g_L (The leak conductance)
     * @return g_L (The leak conductance)
     */
    [[nodiscard]] double get_g_L() const noexcept {
        return g_L;
    }

    /**
     * @brief Returns E_L (The reset membrane potential (leak reversal potential))
     * @return E_L (The reset membrane potential (leak reversal potential))
     */
    [[nodiscard]] double get_E_L() const noexcept {
        return E_L;
    }

    /**
     * @brief Returns V_T (The spiking threshold in the equation)
     * @return V_T (The spiking threshold in the equation)
     */
    [[nodiscard]] double get_V_T() const noexcept {
        return V_T;
    }

    /**
     * @brief Returns d_T (The slope factor)
     * @return d_T (The slope factor)
     */
    [[nodiscard]] double get_d_T() const noexcept {
        return d_T;
    }

    /**
     * @brief Returns tau_w (The dampening factor for w(t))
     * @return tau_w (The dampening factor for w(t))
     */
    [[nodiscard]] double get_tau_w() const noexcept {
        return tau_w;
    }

    /**
     * @brief Returns a (The subthreshold adaptation)
     * @return a (The subthreshold adaptation)
     */
    [[nodiscard]] double get_a() const noexcept {
        return a;
    }

    /**
     * @brief Returns b (The additional dampening for w(t) in case of spiking)
     * @return b (The additional dampening for w(t) in case of spiking)
     */
    [[nodiscard]] double get_b() const noexcept {
        return b;
    }

    /**
     * @brief Returns V_spike (The spiking threshold in the spiking check)
     * @return V_spike (The spiking threshold in the spiking check)
     */
    [[nodiscard]] double get_V_spike() const noexcept {
        return V_spike;
    }

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     * @param number_neurons The number of local neurons to store in this class
     */
    void init(size_t number_neurons) final;

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void create_neurons(size_t creation_count) final;

    static constexpr double default_C{ 281.0 };
    static constexpr double default_g_L{ 30.0 };
    static constexpr double default_E_L{ -70.6 };
    static constexpr double default_V_T{ -50.4 };
    static constexpr double default_d_T{ 2.0 };
    static constexpr double default_tau_w{ 144.0 };
    static constexpr double default_a{ 4.0 };
    static constexpr double default_b{ 0.0805 };
    static constexpr double default_V_spike{ 20.0 };

    static constexpr double min_C{ 100.0 };
    static constexpr double min_g_L{ 0.0 };
    static constexpr double min_E_L{ -150.0 };
    static constexpr double min_V_T{ -150.0 };
    static constexpr double min_d_T{ 0.0 };
    static constexpr double min_tau_w{ 100.0 };
    static constexpr double min_a{ 0.0 };
    static constexpr double min_b{ 0.0 };
    static constexpr double min_V_spike{ 0.0 };

    static constexpr double max_C{ 500.0 };
    static constexpr double max_g_L{ 100.0 };
    static constexpr double max_E_L{ -20.0 };
    static constexpr double max_V_T{ 0.0 };
    static constexpr double max_d_T{ 10.0 };
    static constexpr double max_tau_w{ 200.0 };
    static constexpr double max_a{ 10.0 };
    static constexpr double max_b{ 0.3 };
    static constexpr double max_V_spike{ 70.0 };

protected:
    void update_activity(const NeuronID& neuron_id) final;

    void init_neurons(size_t start_id, size_t end_id) final;

private:
    [[nodiscard]] double f(double x) const noexcept;

    [[nodiscard]] double iter_x(double x, double w, double I_syn) const noexcept;

    [[nodiscard]] double iter_refrac(double w, double x) const noexcept;

    std::vector<double> w{}; // adaption variable

    double C{ default_C }; // membrance capacitance
    double g_L{ default_g_L }; // leak conductance
    double E_L{ default_E_L }; // leak reversal potential
    double V_T{ default_V_T }; // spike threshold
    double d_T{ default_d_T }; // slope factor
    double tau_w{ default_tau_w }; // adaptation time constant
    double a{ default_a }; // subthreshold
    double b{ default_b }; // spike-triggered adaptation

    double V_spike{ default_V_spike }; // spike trigger
};

} // namespace models
