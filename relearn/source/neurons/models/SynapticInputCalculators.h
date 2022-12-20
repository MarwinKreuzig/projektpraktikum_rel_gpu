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

#include "neurons/models/SynapticInputCalculator.h"

class NetworkGraph;

/**
 * This class returns the input from the synapses directly
 */
class LinearSynapticInputCalculator : public SynapticInputCalculator {
public:
    /**
     * @brief Constructs a new instance of type LinearSynapticInputCalculator with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters against the min and max values defined below in order to allow other values besides in the GUI
     * @param k The factor by which the input of a neighboring spiking neuron is weighted
     */
    LinearSynapticInputCalculator(const double k)
        : SynapticInputCalculator(k){};

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<SynapticInputCalculator> clone() const final {
        return std::make_unique<LinearSynapticInputCalculator>(get_k());
    }

protected:
    void update_synaptic_input(const NetworkGraph& network_graph_static, const NetworkGraph& network_graph_plastic, const std::vector<FiredStatus>& fired, const std::vector<UpdateStatus>& disable_flags) override;
};

/**
 * This class returns the input from the synapses when applying a logarithm
 */
class LogarithmicSynapticInputCalculator : public SynapticInputCalculator {
public:
    /**
     * @brief Constructs a new instance of type LogarithmicSynapticInputCalculator with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters against the min and max values defined below in order to allow other values besides in the GUI
     * @param k The factor by which the input of a neighboring spiking neuron is weighted
     */
    LogarithmicSynapticInputCalculator(const double k)
        : SynapticInputCalculator(k){};

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<SynapticInputCalculator> clone() const final {
        return std::make_unique<LogarithmicSynapticInputCalculator>(get_k());
    }

protected:
    void update_synaptic_input(const NetworkGraph& network_graph_static, const NetworkGraph& network_graph_plastic, const std::vector<FiredStatus>& fired, const std::vector<UpdateStatus>& disable_flags) override;
};
