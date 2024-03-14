/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronModel.h"

namespace models {
/**
 * This class inherits from NeuronModel and implements the spiking model from Izhikevich.
 * The differential equations are:
 *      d/dt v(t) = k1 * v(t)^2 + k2 * v(t) + k3 - u(t) + input
 *      d/dt u(t) = a * (b * x - u(t))
 * If v(t) >= V_spike:
 *      v(t) = c
 *      u(t) += d
 */
class IzhikevichModel : public NeuronModel {
    friend class AdapterNeuronModel<IzhikevichModel>;

public:
    /**
     * @brief Constructs a new instance of type IzhikevichModel with 0 neurons and default values for all parameters
     */
    IzhikevichModel() = default;

    /**
     * @brief Constructs a new instance of type IzhikevichModel with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters against the min and max values defined below in order to allow other values besides in the GUI
     * @param h See NeuronModel(...)
     * @param synaptic_input_calculator See NeuronModel(...)
     * @param background_activity_calculator See NeuronModel(...)
     * @param stimulus_calculator See NeuronModel(...)
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
        unsigned int h,
        std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
        std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator,
        std::unique_ptr<Stimulus>&& stimulus_calculator,
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
    [[nodiscard]] virtual std::unique_ptr<NeuronModel> clone() const final;

    /**
     * @brief Returns the dampening variable u
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The dampening variable u
     */
    [[nodiscard]] virtual double get_secondary_variable(const NeuronID neuron_id) const final {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < get_number_neurons(), "IzhikevichModel::get_secondary_variable: id is too large: {}", neuron_id);
        return u[local_neuron_id];
    }

    double iter_refraction(double, double) const noexcept;

    /**
     * @brief Returns a vector with all adjustable ModelParameter for this class and NeuronModel
     * @return A vector with all adjustable ModelParameter
     */
    [[nodiscard]] virtual std::vector<ModelParameter> get_parameter() final;

    /**
     * @brief Returns the name of this model
     * @return The name of this model
     */
    [[nodiscard]] virtual std::string name() final;

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
    void virtual init(number_neurons_type number_neurons) override final;

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void virtual create_neurons(number_neurons_type creation_count) override final;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
            + u.capacity() * sizeof(double);
        footprint->emplace("IzhikevichModel", my_footprint);

        NeuronModel::record_memory_footprint(footprint);
    }

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

    virtual void update_activity(const step_type step) override final;

    virtual void update_activity_benchmark() override final;

    virtual void init_neurons(number_neurons_type start_id, number_neurons_type end_id) override final;

private:
    [[nodiscard]] bool spiked(double x) const noexcept;

    void update_activity_benchmark(NeuronID neuron_id);

    std::vector<double> u{}; // membrane recovery

    double a{ default_a }; // timescale of membrane recovery u
    double b{ default_b }; // sensitivity of membrane recovery to membrane potential v (x)
    double c{ default_c }; // after-spike reset value for membrane potential v (x)
    double d{ default_d }; // after-spike reset of membrane recovery u

    double V_spike{ default_V_spike };

    double k1{ default_k1 };
    double k2{ default_k2 };
    double k3{ default_k3 };
};

} // namespace models
