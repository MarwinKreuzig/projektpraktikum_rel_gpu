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

#include "ElementType.h"
#include "MPIWrapper.h"
#include "ModelParameter.h"
#include "RelearnException.h"
#include "SignalType.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

class NeuronMonitor;

class SynapticElements {
    friend class NeuronMonitor;

public:
    SynapticElements(ElementType type, double min_C_level_to_grow,
        double C_target = SynapticElements::default_C_target,
        double nu = SynapticElements::default_nu,
        double vacant_retract_ratio = SynapticElements::default_vacant_retract_ratio)
        : type(type)
        , min_C_level_to_grow(min_C_level_to_grow)
        , C_target(C_target)
        , nu(nu)
        , vacant_retract_ratio(vacant_retract_ratio) {
    }

    SynapticElements(const SynapticElements& other) = delete;
    SynapticElements(SynapticElements&& other) = default;

    SynapticElements& operator=(const SynapticElements& other) = delete;
    SynapticElements& operator=(SynapticElements&& other) = default;

    ~SynapticElements() = default;

    void init(size_t number_neurons) {
        size = number_neurons;
        cnts.resize(size, 0.0);
        connected_cnts.resize(size, 0);
        delta_cnts.resize(size, 0.0);
        signal_types.resize(size);
    }

    std::vector<ModelParameter> get_parameter() {
        return {
            Parameter<double>{ "Minimum calcium to grow", min_C_level_to_grow, SynapticElements::min_min_C_level_to_grow, SynapticElements::max_min_C_level_to_grow },
            Parameter<double>{ "Target calcium", C_target, SynapticElements::min_C_target, SynapticElements::max_C_target },
            Parameter<double>{ "nu", nu, SynapticElements::min_nu, SynapticElements::max_nu },
            Parameter<double>{ "Vacant synapse retract ratio", vacant_retract_ratio, SynapticElements::min_vacant_retract_ratio, SynapticElements::max_vacant_retract_ratio },
        };
    }

    [[nodiscard]] const std::vector<double>& get_cnts() const noexcept {
        return cnts;
    }

    [[nodiscard]] const std::vector<unsigned int>& get_connected_cnts() const noexcept {
        return connected_cnts;
    }

    [[nodiscard]] const std::vector<double>& get_delta_cnts() const noexcept {
        return delta_cnts;
    }

    [[nodiscard]] const std::vector<SignalType>& get_signal_types() const noexcept {
        return signal_types;
    }

    void update_cnt(size_t neuron_id, double delta) {
        RelearnException::check(neuron_id < cnts.size(), "Synaptic elements, update_cnt out of bounds");
        cnts[neuron_id] += delta;
        RelearnException::check(cnts[neuron_id] >= 0.0, "Synaptic elements, update_cnt was negative");
    }

    void update_conn_cnt(size_t neuron_id, int delta) {
        RelearnException::check(neuron_id < connected_cnts.size(), "Synaptic elements, update_conn_cnt out of bounds");
        if (delta < 0) {
            const unsigned int abs_delta = -delta;
            RelearnException::check(connected_cnts[neuron_id] >= abs_delta, "%u: %d", neuron_id, delta);
        }

        connected_cnts[neuron_id] += delta;
    }

    void update_delta_cnt(size_t neuron_id, double delta) {
        RelearnException::check(neuron_id < delta_cnts.size(), "Synaptic elements, update_delta_cnt out of bounds");
        delta_cnts[neuron_id] += delta;
        RelearnException::check(delta_cnts[neuron_id] >= 0.0, "Synaptic elements, delta cnts is negative");
    }

    void set_signal_type(size_t neuron_id, SignalType type) {
        RelearnException::check(neuron_id < signal_types.size(), "Synaptic elements, set_signal_type out of bounds");
        signal_types[neuron_id] = type;
    }

    [[nodiscard]] double get_cnt(size_t neuron_id) const {
        RelearnException::check(neuron_id < cnts.size(), "Synaptic elements, get_cnt out of bounds");
        return cnts[neuron_id];
    }

    [[nodiscard]] unsigned int get_connected_cnt(size_t neuron_id) const {
        RelearnException::check(neuron_id < connected_cnts.size(), "Synaptic elements, get_connected_cnt out of bounds");
        return connected_cnts[neuron_id];
    }

    [[nodiscard]] double get_delta_cnt(size_t neuron_id) const {
        RelearnException::check(neuron_id < delta_cnts.size(), "Synaptic elements, get_delta_cnt out of bounds");
        return delta_cnts[neuron_id];
    }

    [[nodiscard]] SignalType get_signal_type(size_t neuron_id) const {
        RelearnException::check(neuron_id < signal_types.size(), "Synaptic elements, get_signal_type out of bounds");
        return signal_types[neuron_id];
    }

    [[nodiscard]] ElementType get_element_type() const noexcept {
        return type;
    }

    /**
	 * Updates the number of synaptic elements for neuron "neuron_id"
	 * Returns the number of synapses to be deleted as a consequence of deleting synaptic elements
	 *
	 * Synaptic elements are deleted based on "delta_cnts" in the following way:
	 * 1. Delete vacant elements
	 * 2. Delete bound elements
	 */
    [[nodiscard]] unsigned int update_number_elements(size_t neuron_id);

    void update_number_elements_delta(const std::vector<double>& calcium) noexcept {
        // For my neurons
        for (size_t i = 0; i < this->size; ++i) {
            const auto inc = gaussian_growth_curve(calcium[i], min_C_level_to_grow, C_target, nu);
            delta_cnts[i] += inc;
        }
    }

private:
    [[nodiscard]] static double gaussian_growth_curve(double Ca, double eta, double epsilon, double growth_rate) noexcept {
        /**
		 * gaussian_growth_curve generates a gaussian curve that is compressed by
		 * growth-factor nu and intersects the x-axis at
		 * eta (left intersection) and epsilon (right intersection).
		 * xi and zeta are helper variables that directly follow from eta and epsilon.
		 * See Butz and van Ooyen, 2013 PloS Comp Biol, Equation 4.
		 */

        const auto xi = (eta + epsilon) / 2;
        const auto zeta = (eta - epsilon) / (2 * sqrt(-log(0.5)));

        const auto dz = growth_rate * (2 * exp(-pow((Ca - xi) / zeta, 2)) - 1);
        return dz;
    }

public:
    static constexpr double default_C_target{ 0.7 }; // gold 0.5;
    static constexpr double default_eta_Axons{ 0.4 }; // gold 0.0;
    static constexpr double default_eta_Dendrites_exc{ 0.1 }; // gold 0.0;
    static constexpr double default_eta_Dendrites_inh{ 0.0 }; // gold 0.0;
    static constexpr double default_nu{ 1e-5 }; // gold 1e-5;
    static constexpr double default_vacant_retract_ratio{ 0 };

    static constexpr double min_min_C_level_to_grow{ 0.0 };
    static constexpr double min_C_target{ 0.0 };
    static constexpr double min_nu{ 0.0 };
    static constexpr double min_vacant_retract_ratio{ 0.0 };

    static constexpr double max_min_C_level_to_grow{ 10.0 };
    static constexpr double max_C_target{ 100.0 };
    static constexpr double max_nu{ 1.0 };
    static constexpr double max_vacant_retract_ratio{ 1.0 };

private:
    ElementType type; // Denotes the type of all synaptic elements, which is AXON or DENDRITE
    size_t size = 0;
    std::vector<double> cnts;
    std::vector<double> delta_cnts; // Keeps track of changes in number of elements until those changes are applied in next connectivity update
    std::vector<unsigned int> connected_cnts;
    std::vector<SignalType> signal_types; // Signal type of synaptic elements, i.e., EXCITATORY or INHIBITORY.
        // Note: Given that currently exc. and inh. dendrites are in different objects, this would only be needed for axons.
        //       A more memory-efficient solution would be to use a different class for axons which has the signal_types array.

    // Parameters
    double min_C_level_to_grow; // Minimum level of calcium needed for elements to grow
    double C_target; // Desired calcium level (possible extension of the model: Give all neurons individual C_target values!)
    double nu; // Growth rate for synaptic elements in ms^-1. Needs to be much smaller than 1 to separate activity and structural dynamics.
    double vacant_retract_ratio; // Percentage of how many vacant synaptic elements should be deleted during each connectivity update
};
