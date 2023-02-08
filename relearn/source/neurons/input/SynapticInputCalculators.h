#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/input/SynapticInputCalculator.h"

class NetworkGraph;

/**
 * This class returns the input from the synapses directly
 */
class LinearSynapticInputCalculator : public SynapticInputCalculator {
public:
    /**
     * @brief Constructs a new instance of type LinearSynapticInputCalculator with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters against the min and max values defined below in order to allow other values besides in the GUI
     * @param synapse_conductance The factor by which the input of a neighboring spiking neuron is weighted
     */
    LinearSynapticInputCalculator(const double synapse_conductance, std::unique_ptr<TransmissionDelayer>&& transmission_delayer)
        : SynapticInputCalculator(synapse_conductance, std::move(transmission_delayer)) { }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<SynapticInputCalculator> clone() const final {
        return std::make_unique<LinearSynapticInputCalculator>(get_synapse_conductance(), get_transmission_delayer()->clone());
    }

protected:
    void update_synaptic_input(const NetworkGraph& network_graph_static, const NetworkGraph& network_graph_plastic, std::span<const FiredStatus> fired, std::span<const UpdateStatus> disable_flags) override;
};

/**
 * This class returns the input from the synapses when applying a logarithm
 */
class LogarithmicSynapticInputCalculator : public SynapticInputCalculator {
public:
    /**
     * @brief Constructs a new instance of type LogarithmicSynapticInputCalculator with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters against the min and max values defined below in order to allow other values besides in the GUI
     * @param synapse_conductance The factor by which the input of a neighboring spiking neuron is weighted
     * @param scaling_factor The factor that scales the logarithmic input
     */
    LogarithmicSynapticInputCalculator(const double synapse_conductance, std::unique_ptr<TransmissionDelayer>&& transmission_delayer, const double scaling_factor)
        : SynapticInputCalculator(synapse_conductance, std::move(transmission_delayer))
        , scale_factor(scaling_factor) { }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<SynapticInputCalculator> clone() const final {
        return std::make_unique<LogarithmicSynapticInputCalculator>(get_synapse_conductance(), get_transmission_delayer()->clone(), get_scale_factor());
    }

    /**
     * @brief Returns the currently used scale factor for the logarithm
     * @return The scale factor
     */
    [[nodiscard]] double get_scale_factor() const noexcept {
        return scale_factor;
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() override {
        auto base_params = SynapticInputCalculator::get_parameter();
        base_params.emplace_back(Parameter<double>{ "scale_factor", scale_factor, min_scaling, max_scaling });
        return base_params;
    }

    static constexpr double default_scaling{ 1.0 };

    static constexpr double min_scaling{ 0.0 };
    static constexpr double max_scaling{ 100.0 };

protected:
    void update_synaptic_input(const NetworkGraph& network_graph_static, const NetworkGraph& network_graph_plastic, std::span<const FiredStatus> fired, std::span<const UpdateStatus> disable_flags) override;

private:
    double scale_factor{ default_scaling };
};

class HyperbolicTangentSynapticInputCalculator : public SynapticInputCalculator {
public:
    /**
     * @brief Construcs a new instance of type HyperbolicTangentSynapticInputCalculator with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters agains the min and max values defined below in order to allow other values besides in the GUI
     * @param synapse_conductance The factor by which the input of a neighboring spiking neuron is weighted
     * @param scaling_factor The factor that scales the hyperbolic tanget input
     */
    HyperbolicTangentSynapticInputCalculator(const double synapse_conductance, std::unique_ptr<TransmissionDelayer>&& transmission_delayer, const double scaling_factor)
        : SynapticInputCalculator(synapse_conductance, std::move(transmission_delayer))
        , scale_factor(scaling_factor) { }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<SynapticInputCalculator> clone() const final {
        return std::make_unique<HyperbolicTangentSynapticInputCalculator>(get_synapse_conductance(),get_transmission_delayer()->clone(), get_scale_factor());
    }

    /**
     * @brief Returns the currently used scale factor for the hyperbolic tangent
     * @return The scale factor
     */
    [[nodiscard]] double get_scale_factor() const noexcept {
        return scale_factor;
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() override {
        auto base_params = SynapticInputCalculator::get_parameter();
        base_params.emplace_back(Parameter<double>{ "scale_factor", scale_factor, min_scaling, max_scaling });
        return base_params;
    }

    static constexpr double default_scaling{ 1.0 };

    static constexpr double min_scaling{ 0.0 };
    static constexpr double max_scaling{ 100.0 };

protected:
    void update_synaptic_input(const NetworkGraph& network_graph_static, const NetworkGraph& network_graph_plastic, std::span<const FiredStatus> fired, std::span<const UpdateStatus> disable_flags) override;

private:
    double scale_factor{ default_scaling };
};
