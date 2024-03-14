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
 * This class inherits from NeuronModel and implements an exponential spiking model from Brette and Gerstner.
 * The differential equations are:
 *      d/dt v(t) = (-g_L * (v(t) - E_L) + g_L * d_T * exp((v(t) - V_T) / d_T) - w(t) + input) / C
 *      d/dt w(t) = (a * (v(t) - E_L) - w(t)) / tau_W
 * If v(t) >= V_spike:
 *      v(t) = E_L
 *      w(t) += b
 */
class AEIFModel : public NeuronModel {
    friend class AdapterNeuronModel<AEIFModel>;

public:
    /**
     * @brief Constructs a new instance of type AEIFModel with 0 neurons and default values for all parameters
     */
    AEIFModel() = default;

    /**
     * @brief Constructs a new instance of type IzhikevichModel with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters against the min and max values defined below in order to allow other values besides in the GUI
     * @param h See NeuronModel(...)
     * @param synaptic_input_calculator See NeuronModel(...)
     * @param background_activity_calculator See NeuronModel(...)
     * @param stimulus_calculator See NeuronModel(...)
     * @param C The dampening factor for v(t) (membrane capacitance)
     * @param g_T The leak conductance
     * @param E_L The reset membrane potential (leak reversal potential)
     * @param V_T The spiking threshold in the equation
     * @param d_T The slope factor
     * @param tau_w The dampening factor for w(t)
     * @param a The sub-threshold adaptation
     * @param b The additional dampening for w(t) in case of spiking
     * @param V_spike The spiking threshold in the spiking check
     */
    AEIFModel(
        unsigned int h,
        std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
        std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator,
        std::unique_ptr<Stimulus>&& stimulus_calculator,
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
    [[nodiscard]] virtual std::unique_ptr<NeuronModel> clone() const final;

    /**
     * @brief Returns the dampening variable w
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The dampening variable w
     */
    [[nodiscard]] virtual double get_secondary_variable(const NeuronID neuron_id) const final {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < get_number_neurons(), "In AEIFModel::get_secondary_variable, id is too large");
        return w[local_neuron_id];
    }

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
     * @brief Returns a (The sub-threshold adaptation)
     * @return a (The sub-threshold adaptation)
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
    virtual void init(number_neurons_type number_neurons) final;

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    virtual void create_neurons(number_neurons_type creation_count) override final;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
            + w.capacity() * sizeof(double);
        footprint->emplace("AEIFModel", my_footprint);

        NeuronModel::record_memory_footprint(footprint);
    }

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
    virtual void update_activity(const step_type step) final;

    virtual void update_activity_benchmark() final;

    virtual void init_neurons(number_neurons_type start_id, number_neurons_type end_id) final;

private:
    [[nodiscard]] double f(double x) const noexcept;

    [[nodiscard]] double iter_x(double x, double w, double input) const noexcept;

    [[nodiscard]] double iter_refraction(double w, double x) const noexcept;

    void update_activity_benchmark(NeuronID neuron_id);

    std::vector<double> w{}; // adaption variable

    double C{ default_C }; // membrane capacitance
    double g_L{ default_g_L }; // leak conductance
    double E_L{ default_E_L }; // leak reversal potential
    double V_T{ default_V_T }; // spike threshold
    double d_T{ default_d_T }; // slope factor
    double tau_w{ default_tau_w }; // adaptation time constant
    double a{ default_a }; // sub-threshold
    double b{ default_b }; // spike-triggered adaptation

    double V_spike{ default_V_spike }; // spike trigger
};

} // namespace models
