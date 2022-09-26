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

#include "neurons/models/BackgroundActivityCalculator.h"

#include "util/Random.h"
#include "util/Timers.h"

/**
 * This class provides no input whatsoever.
 */
class NullBackgroundActivityCalculator : public BackgroundActivityCalculator {
public:
    /**
     * @brief Constructs a new object of type NullBackgroundActivityCalculator
     */
    NullBackgroundActivityCalculator() = default;

    virtual ~NullBackgroundActivityCalculator() = default;

    /**
     * @brief This activity calculator does not provide any input
     * @param disable_flags Unused
     */
    void update_input([[maybe_unused]] const std::vector<UpdateStatus>& disable_flags) override {
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<BackgroundActivityCalculator> clone() const override {
        return std::make_unique<NullBackgroundActivityCalculator>();
    }
};

/**
 * This class provides a constant input
 */
class ConstantBackgroundActivityCalculator : public BackgroundActivityCalculator {
public:
    /**
     * @brief Constructs a new object with the given constant input
     * @brief input The base input
     */
    ConstantBackgroundActivityCalculator(const double input) noexcept
        : BackgroundActivityCalculator()
        , base_input(input) {
    }

    virtual ~ConstantBackgroundActivityCalculator() = default;

    /**
     * @brief This activity calculator does not provide any input
     * @param disable_flags Unused
     */
    void update_input(const std::vector<UpdateStatus>& disable_flags) override {
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
            "ConstantBackgroundActivityCalculator::update_input: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
        for (size_t neuron_id = 0U; neuron_id < number_neurons; neuron_id++) {
            const auto input = disable_flags[neuron_id] == UpdateStatus::Disabled ? 0.0 : base_input;
            set_background_activity(neuron_id, input);
        }
        Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<BackgroundActivityCalculator> clone() const override {
        return std::make_unique<ConstantBackgroundActivityCalculator>(base_input);
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() override {
        auto parameters = BackgroundActivityCalculator::get_parameter();
        parameters.emplace_back(Parameter<double>("Base background activity", base_input, BackgroundActivityCalculator::min_base_background_activity, BackgroundActivityCalculator::max_base_background_activity));

        return parameters;
    }

private:
    double base_input{ default_base_background_activity };
};

/**
 * This class provides a normally distributed input, i.e., according to some N(expected, standard deviation) + base
 */
class NormalBackgroundActivityCalculator : public BackgroundActivityCalculator {
public:
    /**
     * @brief Constructs a new object with the given normal input, i.e.,
     *      from base + N(mean, stddev)
     * @brief base The base input
     * @brief mean The mean input
     * @brief stddev The standard deviation, must be > 0.0
     * @exception Throws a RelearnException if stddev <= 0.0
     */
    NormalBackgroundActivityCalculator(const double base, const double mean, const double stddev)
        : BackgroundActivityCalculator()
        , base_input(base)
        , mean_input(mean)
        , stddev_input(stddev) {
        RelearnException::check(stddev > 0.0, "NormalBackgroundActivityCalculator::NormalBackgroundActivityCalculator: stddev was: {}", stddev);
    }

    virtual ~NormalBackgroundActivityCalculator() = default;

    /**
     * @brief This activity calculator does not provide any input
     * @param disable_flags Unused
     */
    void update_input(const std::vector<UpdateStatus>& disable_flags) override {
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
            "NormalBackgroundActivityCalculator::update_input: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
        for (size_t neuron_id = 0U; neuron_id < number_neurons; neuron_id++) {
            const auto input = disable_flags[neuron_id] == UpdateStatus::Disabled ? 0.0 : base_input + RandomHolder::get_random_normal_double(RandomHolderKey::BackgroundActivity, mean_input, stddev_input);
            set_background_activity(neuron_id, input);
        }
        Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<BackgroundActivityCalculator> clone() const override {
        return std::make_unique<NormalBackgroundActivityCalculator>(base_input, mean_input, stddev_input);
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() override {
        auto parameters = BackgroundActivityCalculator::get_parameter();
        parameters.emplace_back(Parameter<double>("Base background activity", base_input, BackgroundActivityCalculator::min_base_background_activity, BackgroundActivityCalculator::max_base_background_activity));
        parameters.emplace_back(Parameter<double>("Mean background activity", mean_input, BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean));
        parameters.emplace_back(Parameter<double>("Stddev background activity", stddev_input, BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev));

        return parameters;
    }

private:
    double base_input{ default_base_background_activity };
    double mean_input{ default_background_activity_mean };
    double stddev_input{ default_background_activity_stddev };
};
