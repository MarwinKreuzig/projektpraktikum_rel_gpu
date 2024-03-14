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
 * This class inherits from NeuronModel and implements the spiking model from Fitz, Hugh, Nagumo.
 * The differential equations are:
 *      d/dt v(t) = v(t) - (v(t)^3)/3 - w(t) + input
 *      d/dt w(t) = phi * (v(t) + a - b * w(t))
 */
class FitzHughNagumoModel : public NeuronModel {
    friend class AdapterNeuronModel<FitzHughNagumoModel>;

public:
    /**
     * @brief Constructs a new instance of type FitzHughNagumoModel with 0 neurons and default values for all parameters
     */
    FitzHughNagumoModel() = default;

    /**
     * @brief Constructs a new instance of type IzhikevichModel with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters against the min and max values defined below in order to allow other values besides in the GUI
     * @param h See NeuronModel(...)
     * @param synaptic_input_calculator See NeuronModel(...)
     * @param background_activity_calculator See NeuronModel(...)
     * @param stimulus_calculator See NeuronModel(...)
     * @param a The constant inside the equation for d/dt w(t)
     * @param b The dampening factor for w(t) inside the equation for d/dt w(t)
     * @param phi The dampening factor for w(t)
     */
    FitzHughNagumoModel(
        unsigned int h,
        std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
        std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator,
        std::unique_ptr<Stimulus>&& stimulus_calculator,
        double a,
        double b,
        double phi);

    /**
     * @brief Clones this instance and creates a new FitzHughNagumoModel with the same parameters and 0 local neurons
     */
    [[nodiscard]] std::unique_ptr<NeuronModel> clone() const final;

    /**
     * @brief Returns the dampening variable w
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The dampening variable w
     */
    [[nodiscard]] double get_secondary_variable(const NeuronID neuron_id) const final {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < get_number_neurons(), "In FitzHughNagumoModel::get_secondary_variable, id is too large");
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
    void init(number_neurons_type number_neurons) override final;

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void create_neurons(number_neurons_type creation_count) override final;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
            + w.capacity() * sizeof(double);
        footprint->emplace("FitzHughNagumoModel", my_footprint);

        NeuronModel::record_memory_footprint(footprint);
    }

    static constexpr double default_a{ 0.7 };
    static constexpr double default_b{ 0.8 };
    static constexpr double default_phi{ 0.08 };

    static constexpr double min_a{ 0.6 };
    static constexpr double min_b{ 0.7 };
    static constexpr double min_phi{ 0.07 };

    static constexpr double max_a{ 0.8 };
    static constexpr double max_b{ 0.9 };
    static constexpr double max_phi{ 0.09 };

    static constexpr double init_x{ -1.2 };
    static constexpr double init_w{ -0.6 };

protected:
    void update_activity(const step_type step) final;

    void update_activity_benchmark() final;

    void init_neurons(number_neurons_type start_id, number_neurons_type end_id) final;

private:
    [[nodiscard]] static double iter_x(double x, double w, double input) noexcept;

    [[nodiscard]] double iter_refraction(double w, double x) const noexcept;

    [[nodiscard]] static bool spiked(double x, double w) noexcept;

    void update_activity_benchmark(NeuronID neuron_id);

    std::vector<double> w{}; // recovery variable

    double a{ default_a };
    double b{ default_b };
    double phi{ default_phi };
};

} // namespace models
