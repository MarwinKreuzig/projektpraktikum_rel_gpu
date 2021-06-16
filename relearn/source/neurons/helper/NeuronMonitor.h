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

#include "../Neurons.h"
#include "../models/NeuronModels.h"

#include <memory>
#include <vector>

/**
 * An object of type NeuronInformation functions as a snapshot of one neuron at one point in the simulation.
 * It stores all necessary superficial informations as a plain-old-data class.
 */
class NeuronInformation {
    double calcium{};
    double x{};
    bool fired{};
    double secondary{};
    double I_sync{};

    double axons{};
    double axons_connected{};
    double dendrites_exc{};
    double dendrites_exc_connected{};
    double dendrites_inh{};
    double dendrites_inh_connected{};

public:
    /**
     * @brief Constructs a NeuronInformation that holds the arguments in one class
     * @param c The current calcium concentration
     * @param x The current membrane potential
     * @param f The current fire status
     * @param s The current secondary variable of the model
     * @param i The current synaptic input
     * @param ax The current number of axonal elements
     * @param ax_c The current number of connected axonal elements
     * @param de The current number of excitatory dendritic elements
     * @param de_c The current number of connected excitatory dendritic elements
     * @param di The current number of inhibitory dendritic elements
     * @param di_c The current number of connected inhibitory dendritic elements
     */
    NeuronInformation(double c, double x, bool f, double s, double i,
        double ax, double ax_c, double de, double de_c, double di, double di_c) noexcept
        : calcium(c)
        , x(x)
        , fired(f)
        , secondary(s)
        , I_sync(i)
        , axons(ax)
        , axons_connected(ax_c)
        , dendrites_exc(de)
        , dendrites_exc_connected(de_c)
        , dendrites_inh(di)
        , dendrites_inh_connected(di_c) {
    }

    /**
     * @brief Returns the stored calcium concentration
     * @return The stored calcium concentration
     */
    [[nodiscard]] double get_calcium() const noexcept {
        return calcium;
    }

    /**
     * @brief Returns the stored membrane potential
     * @return The stored membrane potential
     */
    [[nodiscard]] double get_x() const noexcept {
        return x;
    }

    /**
     * @brief Returns the stored fire status
     * @return The stored fire status
     */
    [[nodiscard]] bool get_fired() const noexcept {
        return fired;
    }

    /**
     * @brief Returns the stored secondary variable of the model
     * @return The stored secondary variable of the model
     */
    [[nodiscard]] double get_secondary() const noexcept {
        return secondary;
    }

    /**
     * @brief Returns the stored synaptic input
     * @return The stored synaptic input
     */
    [[nodiscard]] double get_I_sync() const noexcept {
        return I_sync;
    }

    /**
     * @brief Returns the stored number of axonal elements
     * @return The stored number of axonal elements
     */
    [[nodiscard]] double get_axons() const noexcept {
        return axons;
    }

    /**
     * @brief Returns the stored number of connected axonal elements
     * @return The stored number of connected axonal elements
     */
    [[nodiscard]] double get_axons_connected() const noexcept {
        return axons_connected;
    }

    /**
     * @brief Returns the stored number of excitatory dendritic elements
     * @return The stored number of excitatory dendritic elements
     */
    [[nodiscard]] double get_dendrites_exc() const noexcept {
        return dendrites_exc;
    }

    /**
     * @brief Returns the stored number of connected excitatory dendritic elements
     * @return The stored number of connected excitatory dendritic elements
     */
    [[nodiscard]] double get_dendrites_exc_connected() const noexcept {
        return dendrites_exc_connected;
    }

    /**
     * @brief Returns the stored number of inhibitory dendritic elements
     * @return The stored number of inhibitory dendritic elements
     */
    [[nodiscard]] double get_dendrites_inh() const noexcept {
        return dendrites_inh;
    }

    /**
     * @brief Returns the stored number of connected inhibitory dendritic elements
     * @return The stored number of connected inhibitory dendritic elements
     */
    [[nodiscard]] double get_dendrites_inh_connected() const noexcept {
        return dendrites_inh_connected;
    }
};

/**
 * An object of type NeuronMonitor monitors a specified neuron throughout the simulation.
 * It automatically gathers all necessary data, however, it only does so if there is space left.
 * 
 * Offers the following static members:
 * neurons_to_monitor - an std::shared_ptr to the neurons to monitor. Has to be set before a call to record_data()
 * max_steps - the number of pre-allocated slots for NeuronInformations
 * current_step - The current step for each monitor. Has to be incremented from the outside
 */
class NeuronMonitor {
    size_t target_neuron_id{};

    std::vector<NeuronInformation> informations{};

public:
    static inline std::shared_ptr<Neurons> neurons_to_monitor{};
    static inline size_t max_steps = 0;
    static inline size_t current_step = 0;

    /**
     * @brief Constructs a NeuronMonitor that monitors the specified neuron
     * @param num_neurons The local neuron id for the object to monitor
     */
    explicit NeuronMonitor(const size_t neuron_id) noexcept
        : target_neuron_id(neuron_id) {
        informations.reserve(max_steps);
    }
    ~NeuronMonitor() = default;

    NeuronMonitor(const NeuronMonitor& other) noexcept = delete;
    NeuronMonitor& operator=(const NeuronMonitor& other) noexcept = delete;

    NeuronMonitor(NeuronMonitor&& other) noexcept = default;
    NeuronMonitor& operator=(NeuronMonitor&& other) noexcept = default;

    /**
     * @brief Returns the local neuron id which is monitored
     * @return The neuron id
     */
    [[nodiscard]] size_t get_target_id() const noexcept {
        return target_neuron_id;
    }

    /**
     * @brief Captures the current state of the monitored neuron
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons or if the std::shared_ptr is empty
     */
    void record_data() {
        if (current_step >= informations.size()) {
            return;
        }

        RelearnException::check(neurons_to_monitor.operator bool(), "In NeuronMonitor, the shared pointer is empty");
        RelearnException::check(target_neuron_id < neurons_to_monitor->num_neurons, "In NeuronMonitor, the target id is too large for the neurons class");

        const double& calcium = neurons_to_monitor->calcium[target_neuron_id];
        const double& x = neurons_to_monitor->neuron_model->x[target_neuron_id];
        const bool& fired = neurons_to_monitor->neuron_model->fired[target_neuron_id];
        const double& secondary = neurons_to_monitor->neuron_model->get_secondary_variable(target_neuron_id);
        const double& I_sync = neurons_to_monitor->neuron_model->I_syn[target_neuron_id];

        const double& axons = neurons_to_monitor->axons->cnts[target_neuron_id];
        const unsigned int& axons_connected = neurons_to_monitor->axons->connected_cnts[target_neuron_id];
        const double& dendrites_exc = neurons_to_monitor->dendrites_exc->cnts[target_neuron_id];
        const unsigned int& dendrites_exc_connected = neurons_to_monitor->dendrites_exc->connected_cnts[target_neuron_id];
        const double& dendrites_inh = neurons_to_monitor->dendrites_inh->cnts[target_neuron_id];
        const unsigned int& dendrites_inh_connected = neurons_to_monitor->dendrites_inh->connected_cnts[target_neuron_id];

        informations.emplace_back(calcium, x, fired, secondary, I_sync, axons, axons_connected, dendrites_exc, dendrites_exc_connected, dendrites_inh, dendrites_inh_connected);
    }

    /**
     * @brief Increases the capacity for stored NeuronInformations
     * @param neuron_id The amount by which the storage should be increased
     */
    void increase_monitoring_capacity(const size_t increase_by) noexcept {
        informations.reserve(informations.size() + increase_by);
    }

    /**
     * @brief Returns the stored informations. The reference is invalidated by calls to increase_monitor_capacity
     * @return An std::vector of NeuronInformation
     */
    [[nodiscard]] const std::vector<NeuronInformation>& get_informations() const noexcept {
        return informations;
    }
};
