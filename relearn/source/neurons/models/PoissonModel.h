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
 * This class inherits from NeuronModel and implements a poisson spiking model
 */
class PoissonModel : public NeuronModel {
    friend class AdapterNeuronModel<PoissonModel>;

public:
    /**
     * @brief Constructs a new instance of type PoissonModel with 0 neurons and default values for all parameters
     */
    PoissonModel() = default;

    /**
     * @brief Constructs a new instance of type PoissonModel with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters against the min and max values defined below in order to allow other values besides in the GUI
     * @param h See NeuronModel(...)
     * @param synaptic_input_calculator See NeuronModel(...)
     * @param background_activity_calculator See NeuronModel(...)
     * @param stimulus_calculator See NeuronModel(...)
     * @param x_0 The resting membrane potential
     * @param tau_x The dampening factor by which the membrane potential decreases
     * @param refractory_time The number of steps a neuron doesn't spike after spiking
     */
    PoissonModel(
        unsigned int h,
        std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
        std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator,
        std::unique_ptr<Stimulus>&& stimulus_calculator,
        double x_0,
        double tau_x,
        unsigned int refractory_time);

    /**
     * @brief Clones this instance and creates a new PoissonModel with the same parameters and 0 local neurons
     */
    [[nodiscard]] std::unique_ptr<NeuronModel> clone() const final;

    /**
     * @brief Returns the refractory_time time (The number of steps a neuron doesn't spike after spiking)
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The refractory_time time (The number of steps a neuron doesn't spike after spiking)
     */
    [[nodiscard]] double get_secondary_variable(const NeuronID neuron_id) const final {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < get_number_neurons(), "PoissonModel::get_secondary_variable: id is too large: {}", neuron_id);
        return refractory_time[local_neuron_id];
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
     * @brief Returns refractory_period (The number of steps a neuron doesn't spike after spiking)
     * @return refractory_period (The number of steps a neuron doesn't spike after spiking)
     */
    [[nodiscard]] unsigned int get_refractory_time() const noexcept {
        return refractory_period;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(NeuronModel)
            + refractory_time.capacity() * sizeof(unsigned int);
        footprint->emplace("PoissonModel", my_footprint);

        NeuronModel::record_memory_footprint(footprint);
    }

    static constexpr double default_x_0{ 0.05 };
    static constexpr double default_tau_x{ 5.0 };
    static constexpr unsigned int default_refractory_period{ 4 }; // In Sebastian's work: 4

    static constexpr double min_x_0{ 0.0 };
    static constexpr double min_tau_x{ 0.0 };
    static constexpr unsigned int min_refractory_time{ 0 };

    static constexpr double max_x_0{ 1.0 };
    static constexpr double max_tau_x{ 1000.0 };
    static constexpr unsigned int max_refractory_time{ 1000 };

protected:
    void update_activity_cpu() final;

    void update_activity_benchmark() final;

    void init_neurons_cpu(number_neurons_type start_id, number_neurons_type end_id) final;

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     *      Sets the initial refractory_time counter to 0
     * @param number_neurons The number of local neurons to store in this class
     */
    void init_cpu(number_neurons_type number_neurons) override final;

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void create_neurons_cpu(number_neurons_type creation_count) final;

private:
    [[nodiscard]] double iter_x(const double x, const double input) const noexcept {
        return ((x_0 - x) / tau_x + input);
    }

    void update_activity_benchmark(NeuronID neuron_id);

    std::vector<unsigned int> refractory_time{}; // refractory time

    double x_0{ default_x_0 }; // Background or resting activity
    double tau_x{ default_tau_x }; // Decay time of firing rate in msec
    unsigned int refractory_period{ default_refractory_period }; // Length of refractory period in msec. After an action potential a neuron cannot fire for this time
};

} // namespace models
