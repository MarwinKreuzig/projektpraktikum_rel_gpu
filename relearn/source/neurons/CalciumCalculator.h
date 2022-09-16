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
#include "util/RelearnException.h"
#include "util/TaggedID.h"

#include <functional>
#include <limits>
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
     * @brief Constructs a new object with the given parameters
     * @param decay_type The type of decay (abolute, relative, none)
     * @param decay_amount The amount of decay
     * @param decay_step The steps when the decay occurs
     * @exception Throws a RelearnException if
     *      (a) The decay_type is Relative, but the amount is not from [0, 1) and the step is not larger than 0
     *      (b) The decay_type is Absolute, but the amount is not from (0, inf) and the step is not larger than 0
     */
    explicit CalciumCalculator(const TargetCalciumDecay decay_type = TargetCalciumDecay::None, const double decay_amount = 0.1, const size_t decay_step = 1000)
        : decay_type(decay_type)
        , decay_amount(decay_amount)
        , decay_step(decay_step) {

        if (decay_type == TargetCalciumDecay::Absolute) {
            RelearnException::check(decay_amount > 0, "CalciumCalculator::CalciumCalculator: The decay type is absolute, but the amount was not larger than 0! {}", decay_amount);
            RelearnException::check(decay_step > 0, "CalciumCalculator::CalciumCalculator: The decay type is absolute, but the step is 0!");
        } else if (decay_type == TargetCalciumDecay::Relative) {
            RelearnException::check(decay_amount >= 0 && decay_amount < 1.0, "CalciumCalculator::CalciumCalculator: The decay type is relative, but the amount was not from [0, 1)! {}", decay_amount);
            RelearnException::check(decay_step > 0, "CalciumCalculator::CalciumCalculator: The decay type is relative, but the step is 0!");
        }
    }

    /**
     * @brief Returns the type of target value decay
     * @return The decay type
     */
    [[nodiscard]] constexpr TargetCalciumDecay get_decay_type() const noexcept {
        return decay_type;
    }

    /**
     * @brief Returns the amount of target value decay
     * @return The decay amount
     */
    [[nodiscard]] constexpr double get_decay_amount() const noexcept {
        return decay_amount;
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
     * @exception Throws a RelearnException if the new value is not in the given interval by minimum and maximum
     */
    constexpr void set_beta(const double new_beta) {
        RelearnException::check(min_beta <= new_beta, "CalciumCalculator::set_beta: new_beta was smaller than the minimum: {} vs {}", new_beta, min_beta);
        RelearnException::check(new_beta <= max_beta, "CalciumCalculator::set_beta: new_beta was larger than the maximum: {} vs {}", new_beta, max_beta);
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
     * @exception Throws a RelearnException if the new value is not in the given interval by minimum and maximum
     */
    constexpr void set_tau_C(const double new_tau_C) {
        RelearnException::check(min_tau_C <= new_tau_C, "CalciumCalculator::set_tau_C: new_tau_C was smaller than the minimum: {} vs {}", new_tau_C, min_tau_C);
        RelearnException::check(new_tau_C <= max_tau_C, "CalciumCalculator::set_tau_C: new_tau_C was larger than the maximum: {} vs {}", new_tau_C, max_tau_C);
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
     * @exception Throws a RelearnException if the new value is not in the given interval by minimum and maximum
     */
    constexpr void set_h(const unsigned int new_h) {
        RelearnException::check(min_h <= new_h, "CalciumCalculator::set_h: new_h was smaller than the minimum: {} vs {}", new_h, min_h);
        RelearnException::check(new_h <= max_h, "CalciumCalculator::set_h: new_h was larger than the maximum: {} vs {}", new_h, max_h);
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
     * @brief Sets the function that is used to determine the initial calcium value of the neurons.
     *      When calling init(...), the initial calcium calculator must not be empty. It can be so inbetween.
     * @param calculator The function that maps neuron id to initial calcium value
     */
    void set_initial_calcium_calculator(std::function<double(int, NeuronID::value_type)> initiator) noexcept {
        initial_calcium_initiator = std::move(initiator);
    }

    /**
     * @brief Sets the function that is used to determine the target calcium value of the neurons
     *      When calling init(...), the target calcium calculator must not be empty. It can be so inbetween.
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

    static constexpr double default_C_target{ 0.7 }; // In Sebastians work: 0.5

    static constexpr double default_tau_C{ 10000 }; // In Sebastians work: 5000
    static constexpr double default_beta{ 0.001 }; // In Sebastians work: 0.001
    static constexpr unsigned int default_h{ 10 };

    static constexpr double min_tau_C{ 0 };
    static constexpr double min_beta{ 0.0 };
    static constexpr unsigned int min_h{ 1 };

    static constexpr double max_tau_C{ 10.0e+6 };
    static constexpr double max_beta{ 1.0 };
    static constexpr unsigned int max_h{ std::numeric_limits<unsigned int>::max() };

private:
    void update_current_calcium(const std::vector<UpdateStatus>& disable_flags, const std::vector<FiredStatus>& fired_status) noexcept;

    void update_target_calcium(size_t step, const std::vector<UpdateStatus>& disable_flags) noexcept;

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
