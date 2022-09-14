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

#include "TargetCalciumDecay.h"
#include "neurons/FiredStatus.h"
#include "neurons/UpdateStatus.h"
#include "util/TaggedID.h"

#include <functional>
#include <vector>

class NeuronMonitor;

/**
 * This class focuses on calculating the inter-cellular calcium concentration of the neurons.
 * It offers the functionality for neuron-dependent target values and an overall reduction thereof.
 */
class CalciumCalculator {
    friend class NeuronMonitor;

public:
    /**
     * @brief Sets the type of target value decay
     * @param new_decay_type The new decay type
     */
    constexpr void set_decay_type(const TargetCalciumDecay new_decay_type) noexcept {
        decay_type = new_decay_type;
    }

    /**
     * @brief Returns the type of target value decay
     * @return The decay type
     */
    [[nodiscard]] constexpr TargetCalciumDecay get_decay_type() const noexcept {
        return decay_type;
    }

    /**
     * @brief Sets the amount of target value decay
     * @param new_decay_amount The new decay amount
     */
    constexpr void set_decay_amount(const double new_decay_amount) noexcept {
        decay_amount = new_decay_amount;
    }

    /**
     * @brief Returns the amount of target value decay
     * @return The decay amount
     */
    [[nodiscard]] constexpr double get_decay_amount() const noexcept {
        return decay_amount;
    }

    /**
     * @brief Sets the steps of target value decay
     * @param new_decay_step The new decay steps, i.e., a decay happens every new_decay_step steps
     */
    constexpr void set_decay_step(const size_t new_decay_step) noexcept {
        decay_step = new_decay_step;
    }

    /**
     * @brief Returns the steps of target value decay
     * @return The new decay steps
     */
    [[nodiscard]] constexpr size_t get_decay_step() const noexcept {
        return decay_step;
    }

    /**
     * @brief Sets beta, the constant by which the calcium increases every time a neuron spikes
     * @param new_beta The new value for beta
     */
    constexpr void set_beta(const double new_beta) noexcept {
        beta = new_beta;
    }

    /**
     * @brief Returns beta, increase-in-calcium constant
     * @return beta
     */
    [[nodiscard]] constexpr double get_beta() const noexcept {
        return beta;
    }

    /**
     * @brief Sets the dampening factor for the calcium decrease (the decay constant)
     * @param new_tau_C The dampening factor
     */
    constexpr void set_tau_C(const double new_tau_C) noexcept {
        tau_C = new_tau_C;
    }

    /**
     * @brief Returns tau_C (The dampening factor by which the calcium decreases)
     * @return the dampening factor
     */
    [[nodiscard]] constexpr double get_tau_C() const noexcept {
        return tau_C;
    }

    /**
     * @brief Sets the numerical integration's step size
     * @param new_h The new step size
     */
    constexpr void set_h(const unsigned int new_h) noexcept {
        h = new_h;
    }

    /**
     * @brief Returns the numerical integration's step size
     * @return The step size
     */
    [[nodiscard]] constexpr unsigned int get_h() const noexcept {
        return h;
    }

    /**
     * @brief Returns the inter-cellular calcium concentration
     * @return The calcium values
     */
    [[nodiscard]] constexpr const std::vector<double>& get_calcium() const noexcept {
        return calcium;
    }

    /**
     * @brief Returns the target calcium values
     * @return The target calcium values
     */
    [[nodiscard]] constexpr const std::vector<double>& get_target_calcium() const noexcept {
        return target_calcium;
    }

    /**
     * @brief Sets the function that is used to determine the initial calcium value of the neurons
     * @param calculator The function that maps neuron id to initial calcium value
     */
    void set_initial_calcium_calculator(std::function<double(int, NeuronID::value_type)> initiator) noexcept {
        initial_calcium_initiator = std::move(initiator);
    }

    /**
     * @brief Sets the function that is used to determine the target calcium value of the neurons
     * @param calculator The function that maps neuron id to target calcium value
     */
    void set_target_calcium_calculator(std::function<double(int, NeuronID::value_type)> calculator) noexcept {
        target_calcium_calculator = std::move(calculator);
    }

    /**
     * @brief Initializes the given number of neurons, uses the previously passed functions to determine the initial and target values
     * @param number_neurons The number of neurons, must be > 0
     * @exception Throws a RelearnException if any of the functions is empty or number_neurons == 0
     */
    void init(size_t number_neurons);

    /**
     * @brief Creates the given number of neurons, uses the previously passed functions to determine the initial and target values
     * @param number_neurons The number of neurons, must be > 0
     * @exception Throws a RelearnException if any of the functions is empty or number_neurons == 0
     */
    void create_neurons(size_t number_neurons);

    /**
     * @brief Updates the calcium values for each neuron
     * @param step The current update step
     * @param disable_flags Indicates if a neuron is to be updated
     * @param fired_status Indicates if a neuron fired
     * @exception Throws a RelearnException if the size of the vectors doesn't match the size of the stored vectors
     */
    void update_calcium(size_t step, const std::vector<UpdateStatus>& disable_flags, const std::vector<FiredStatus>& fired_status);

    static constexpr double default_tau_C{ 10000 }; // In Sebastians work: 5000
    static constexpr double default_beta{ 0.001 }; // In Sebastians work: 0.001
    static constexpr unsigned int default_h{ 10 };

    static constexpr double min_tau_C{ 0 };
    static constexpr double min_beta{ 0.0 };
    static constexpr unsigned int min_h{ 1 };

    static constexpr double max_tau_C{ 10.0e+6 };
    static constexpr double max_beta{ 1.0 };
    static constexpr unsigned int max_h{ 1000 };

private:
    std::function<double(int, NeuronID::value_type)> initial_calcium_initiator{};
    std::function<double(int, NeuronID::value_type)> target_calcium_calculator{};

    std::vector<double> calcium{};
    std::vector<double> target_calcium{};

    double beta{ default_beta };
    double tau_C{ default_tau_C }; // Decay time of calcium
    unsigned int h{ default_h }; // Precision for Euler integration

    TargetCalciumDecay decay_type{ TargetCalciumDecay::None };
    double decay_amount{ 0.0 };
    size_t decay_step{ 1000 };
};
