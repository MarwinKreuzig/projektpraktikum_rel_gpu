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

class CalciumCalculator {
    friend class NeuronMonitor;

public:
    constexpr void set_decay_type(const TargetCalciumDecay new_decay_type) noexcept {
        decay_type = new_decay_type;
    }

    [[nodiscard]] constexpr TargetCalciumDecay get_decay_type() const noexcept {
        return decay_type;
    }

    constexpr void set_decay_amount(const double new_decay_amount) noexcept {
        decay_amount = new_decay_amount;
    }

    [[nodiscard]] constexpr double get_decay_amount() const noexcept {
        return decay_amount;
    }

    constexpr void set_decay_step(const size_t new_decay_step) noexcept {
        decay_step = new_decay_step;
    }

    [[nodiscard]] constexpr size_t get_decay_step() const noexcept {
        return decay_step;
    }

    constexpr void set_beta(const double new_beta) noexcept {
        beta = new_beta;
    }

    /**
     * @brief Returns beta (The factor by which the Calcium is increased whenever a neuron spikes)
     * @return Beta (The factor by which the Calcium is increased whenever a neuron spikes)
     */
    [[nodiscard]] constexpr double get_beta() const noexcept {
        return beta;
    }

    constexpr void set_tau_C(const double new_tau_C) noexcept {
        tau_C = new_tau_C;
    }

    /**
     * @brief Returns tau_C (The dampening factor by which the Calcium decreases)
     * @return tau_C (The dampening factor by which the Calcium decreases)
     */
    [[nodiscard]] constexpr double get_tau_C() const noexcept {
        return tau_C;
    }

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

    [[nodiscard]] constexpr const std::vector<double>& get_calcium() const noexcept {
        return calcium;
    }

    [[nodiscard]] constexpr const std::vector<double>& get_target_calcium() const noexcept {
        return target_calcium;
    }

    void set_initial_calcium_calculator(std::function<double(int, NeuronID::value_type)> initiator) noexcept {
        initial_calcium_initiator = std::move(initiator);
    }

    void set_target_calcium_calculator(std::function<double(int, NeuronID::value_type)> calculator) noexcept {
        target_calcium_calculator = std::move(calculator);
    }

    void init(size_t number_neurons);

    void create_neurons(size_t number_neurons);

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
