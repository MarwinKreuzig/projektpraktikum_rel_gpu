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

#include "neurons/FiredStatus.h"
#include "neurons/UpdateStatus.h"
#include "neurons/models/FiredStatusCommunicator.h"
#include "neurons/models/ModelParameter.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"

#include <memory>
#include <vector>

class NetworkGraph;
class NeuronMonitor;

class SynapticInputCalculator {
    friend class NeuronMonitor;

public:
    /**
     * @brief Construcs a new instance of type NeuronModel with 0 neurons and the passed values for all parameters.
     *      Does not check the parameters agains the min and max values defined below in order to allow other values besides in the GUI
     * @param k The factor by which the input of a neighboring spiking neuron is weighted
     * @param h The step size for the numerical integration
     * @param base_background_activity The base background activity that all neurons are exited with. Is only used if background_activity_stddev > 0.0
     * @param background_activity_mean The mean of background activity taht all neurons are exited with. Is only used if background_activity_stddev > 0.0
     * @param background_activity_stddev The standard deviation of background activity that all neurons are exited with. Is only used if background_activity_stddev > 0.0
     *
     * If background_activity_stddev > 0.0, all neurons are exited with
     *      base_background_activity + N(background_activity_mean, background_activity_stddev)
     */
    SynapticInputCalculator(const double k, const double base_background_activity, const double background_activity_mean, const double background_activity_stddev)
        : k(k)
        , base_background_activity(base_background_activity)
        , background_activity_mean(background_activity_mean)
        , background_activity_stddev(background_activity_stddev) { }

    [[nodiscard]] virtual std::unique_ptr<SynapticInputCalculator> clone() const = 0;

    void init(size_t number_neurons);

    void create_neurons(size_t creation_count);

    void update_input(const NetworkGraph& network_graph, const std::vector<FiredStatus> fired, const std::vector<UpdateStatus>& disable_flags) {
        fired_status_comm->set_local_fired_status(fired, disable_flags, network_graph);
        fired_status_comm->exchange_fired_status();

        update_background_activity(disable_flags);
        update_synaptic_input(network_graph, fired, disable_flags);
    }

    [[nodiscard]] double get_background_activity(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::get_x: id is too large: {}", neuron_id);
        return background_activity[local_neuron_id];
    }

    [[nodiscard]] const std::vector<double>& get_background_activity() const noexcept {
        return background_activity;
    }

    [[nodiscard]] double get_synaptic_input(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::get_x: id is too large: {}", neuron_id);
        return synaptic_input[local_neuron_id];
    }

    [[nodiscard]] const std::vector<double>& get_synaptic_input() const noexcept {
        return synaptic_input;
    }

    [[nodiscard]] const std::unique_ptr<FiredStatusCommunicator>& get_fired_status_communicator() const {
        return fired_status_comm;
    }

    /**
     * @brief Returns k (The factor by which the input of a neighboring spiking neuron is weighted)
     * @return k (The factor by which the input of a neighboring spiking neuron is weighted)
     */
    [[nodiscard]] double get_k() const noexcept {
        return k;
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
     * @brief Returns the number of neurons that are stored in the object
     * @return The number of neurons that are stored in the object
     */
    [[nodiscard]] size_t get_number_neurons() const noexcept {
        return number_local_neurons;
    }

    [[nodiscard]] std::vector<ModelParameter> get_parameter() {
        return {
            Parameter<double>{ "k", k, min_k, max_k },
            Parameter<double>{ "Base background activity", base_background_activity, min_base_background_activity, max_base_background_activity },
            Parameter<double>{ "Background activity mean", background_activity_mean, min_background_activity_mean, max_background_activity_mean },
            Parameter<double>{ "Background activity standard deviation", background_activity_stddev, min_background_activity_stddev, max_background_activity_stddev },
        };
    }

    static constexpr double default_k{ 0.03 };
    static constexpr double default_base_background_activity{ 0.0 };
    static constexpr double default_background_activity_mean{ 0.0 };
    static constexpr double default_background_activity_stddev{ 0.0 };

    static constexpr double min_k{ 0.0 };
    static constexpr double min_base_background_activity{ -10000.0 };
    static constexpr double min_background_activity_mean{ -10000.0 };
    static constexpr double min_background_activity_stddev{ 0.0 };

    static constexpr double max_k{ 10.0 };
    static constexpr double max_base_background_activity{ 10000.0 };
    static constexpr double max_background_activity_mean{ 10000.0 };
    static constexpr double max_background_activity_stddev{ 10000.0 };

protected:
    virtual void update_synaptic_input(const NetworkGraph& network_graph, const std::vector<FiredStatus> fired, const std::vector<UpdateStatus>& disable_flags) = 0;

    virtual void update_background_activity(const std::vector<UpdateStatus>& disable_flags) = 0;

    void set_synaptic_input(const size_t neuron_id, const double value) {
        RelearnException::check(neuron_id < number_local_neurons, "SynapticInputCalculator::set_synaptic_input: neuron_id was too large: {} vs {}", neuron_id, number_local_neurons);
        synaptic_input[neuron_id] = value;
    }

    void set_background_activity(const size_t neuron_id, const double value) {
        RelearnException::check(neuron_id < number_local_neurons, "SynapticInputCalculator::set_background_activity: neuron_id was too large: {} vs {}", neuron_id, number_local_neurons);
        background_activity[neuron_id] = value;
    }

    [[nodiscard]] std::vector<double>& get_inner_background_activity() noexcept {
        return background_activity;
    }

    [[nodiscard]] std::vector<double>& get_inner_synaptic_input() noexcept {
        return synaptic_input;
    }

private:
    size_t number_local_neurons{};

    double k{ default_k }; // Proportionality factor for synapses in Hz

    double base_background_activity{ default_base_background_activity };
    double background_activity_mean{ default_background_activity_mean };
    double background_activity_stddev{ default_background_activity_stddev };

    std::vector<double> synaptic_input{};
    std::vector<double> background_activity{};

    std::unique_ptr<FiredStatusCommunicator> fired_status_comm{};
};