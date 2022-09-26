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
     * @brief Construcs a new instance of type LinearSynapticInputCalculator with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters agains the min and max values defined below in order to allow other values besides in the GUI
     * @param k The factor by which the input of a neighboring spiking neuron is weighted
     * @param base_background_activity The base background activity that all neurons are exited with.
     * @param background_activity_mean The mean of background activity that all neurons are exited with. Is only used if background_activity_stddev > 0.0
     * @param background_activity_stddev The standard deviation of background activity that all neurons are exited with. Is only used if background_activity_stddev > 0.0
     *
     * If background_activity_stddev > 0.0, all neurons are excited with
     *      base_background_activity + N(background_activity_mean, background_activity_stddev)
     * otherwise, all neurons are excited with
     *      base_background_activity
     */
    LinearSynapticInputCalculator(const double k, const double base_background_activity, const double background_activity_mean, const double background_activity_stddev)
        : SynapticInputCalculator(k, base_background_activity, background_activity_mean, background_activity_stddev){};

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<SynapticInputCalculator> clone() const final {
        return std::make_unique<LinearSynapticInputCalculator>(get_k(), get_base_background_activity(), get_background_activity_mean(), get_background_activity_stddev());
    }

protected:
    void update_synaptic_input(const NetworkGraph& network_graph, const std::vector<FiredStatus> fired, const std::vector<UpdateStatus>& disable_flags) override;
};

/**
 * This class returns the input from the synapses when applying a logarithm
 */
class LogarithmicSynapticInputCalculator : public SynapticInputCalculator {
public:
    /**
     * @brief Construcs a new instance of type LogarithmicSynapticInputCalculator with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters agains the min and max values defined below in order to allow other values besides in the GUI
     * @param k The factor by which the input of a neighboring spiking neuron is weighted
     * @param base_background_activity The base background activity that all neurons are exited with.
     * @param background_activity_mean The mean of background activity that all neurons are exited with. Is only used if background_activity_stddev > 0.0
     * @param background_activity_stddev The standard deviation of background activity that all neurons are exited with. Is only used if background_activity_stddev > 0.0
     *
     * If background_activity_stddev > 0.0, all neurons are excited with
     *      base_background_activity + N(background_activity_mean, background_activity_stddev)
     * otherwise, all neurons are excited with
     *      base_background_activity
     */
    LogarithmicSynapticInputCalculator(const double k, const double base_background_activity, const double background_activity_mean, const double background_activity_stddev)
        : SynapticInputCalculator(k, base_background_activity, background_activity_mean, background_activity_stddev){};

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<SynapticInputCalculator> clone() const final {
        return std::make_unique<LogarithmicSynapticInputCalculator>(get_k(), get_base_background_activity(), get_background_activity_mean(), get_background_activity_stddev());
    }

protected:
    void update_synaptic_input(const NetworkGraph& network_graph, const std::vector<FiredStatus> fired, const std::vector<UpdateStatus>& disable_flags) override;
};
