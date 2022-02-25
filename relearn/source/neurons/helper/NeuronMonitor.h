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

#include "neurons/Neurons.h"
#include "neurons/models/NeuronModels.h"

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
    NeuronInformation(const double c, const double x, const bool f, const double s, const double i,
        const double ax, const double ax_c, const double de, const double de_c, const double di, const double di_c) noexcept
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
 * Offers the following static member:
 * neurons_to_monitor - an std::shared_ptr to the neurons to monitor. Has to be set before a call to record_data()
 */
class NeuronMonitor {
    NeuronID target_neuron_id{ NeuronID::uninitialized_id() };

    std::vector<NeuronInformation> informations{};

public:
    static inline std::shared_ptr<Neurons> neurons_to_monitor{};

    /**
     * @brief Constructs a NeuronMonitor that monitors the specified neuron
     * @param neuron_id The local neuron id for the object to monitor
     */
    explicit NeuronMonitor(const NeuronID& neuron_id) noexcept
        : target_neuron_id(neuron_id) {
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
    [[nodiscard]] NeuronID get_target_id() const noexcept {
        return target_neuron_id;
    }

    /**
     * @brief Captures the current state of the monitored neuron
     * @exception Throws a ReleanException if neuron_id is larger or equal to the number of neurons or if the std::shared_ptr is empty
     */
    void record_data() {
        RelearnException::check(neurons_to_monitor.operator bool(), "NeuronMonitor::record_data: The shared pointer is empty");
        
        const auto local_neuron_id = target_neuron_id.get_local_id();
        RelearnException::check(local_neuron_id < neurons_to_monitor->number_neurons, "NeuronMonitor::record_data: The target id is too large for the neurons class");

        const double& calcium = neurons_to_monitor->calcium[local_neuron_id];
        const double& x = neurons_to_monitor->neuron_model->x[local_neuron_id];
        const bool& fired = neurons_to_monitor->neuron_model->fired[local_neuron_id] == FiredStatus::Fired;
        const double& secondary = neurons_to_monitor->neuron_model->get_secondary_variable(target_neuron_id);
        const double& I_sync = neurons_to_monitor->neuron_model->I_syn[local_neuron_id];

        const double& axons = neurons_to_monitor->axons->grown_elements[local_neuron_id];
        const unsigned int& axons_connected = neurons_to_monitor->axons->connected_elements[local_neuron_id];
        const double& dendrites_exc = neurons_to_monitor->dendrites_exc->grown_elements[local_neuron_id];
        const unsigned int& dendrites_exc_connected = neurons_to_monitor->dendrites_exc->connected_elements[local_neuron_id];
        const double& dendrites_inh = neurons_to_monitor->dendrites_inh->grown_elements[local_neuron_id];
        const unsigned int& dendrites_inh_connected = neurons_to_monitor->dendrites_inh->connected_elements[local_neuron_id];

        informations.emplace_back(calcium, x, fired, secondary, I_sync, axons, axons_connected, dendrites_exc, dendrites_exc_connected, dendrites_inh, dendrites_inh_connected);
    }

    /**
     * @brief Increases the capacity for stored NeuronInformations by reserving
     * @param neuron_id The amount by which the storage should be increased
     */
    void increase_monitoring_capacity(const size_t increase_by) noexcept {
        informations.reserve(informations.size() + increase_by);
    }

    /**
     * @brief Clears the recorded data
     */
    void clear() noexcept {
        informations.clear();
    }

    /**
     * @brief Returns the stored informations
     * @return An std::vector of NeuronInformation
     */
    [[nodiscard]] const std::vector<NeuronInformation>& get_informations() const noexcept {
        return informations;
    }
};
