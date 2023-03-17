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

#include "neurons/input/BackgroundActivityCalculator.h"

#include "io/InteractiveNeuronIO.h"
#include "util/Random.h"
#include "util/Timers.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <utility>

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
     * @param step The current update step
     */
    void update_input([[maybe_unused]] const step_type step) override {
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
     * @brief Updates the input, providing constant or 0 input depending on the disable_flags
     * @param step The current update step
     */
    void update_input([[maybe_unused]] const step_type step) override {
        const auto& disable_flags = extra_infos->get_disable_flags();
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
            "ConstantBackgroundActivityCalculator::update_input: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
        for (number_neurons_type neuron_id = 0U; neuron_id < number_neurons; neuron_id++) {
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
 * This class provides a normally distributed input, i.e., according to some N(expected, standard deviation)
 */
class NormalBackgroundActivityCalculator : public BackgroundActivityCalculator {
public:
    /**
     * @brief Constructs a new object with the given normal input, i.e.,
     *      from N(mean, stddev)
     * @brief mean The mean input
     * @brief stddev The standard deviation, must be > 0.0
     * @exception Throws a RelearnException if stddev <= 0.0
     */
    NormalBackgroundActivityCalculator(const double mean, const double stddev)
        : BackgroundActivityCalculator()
        , mean_input(mean)
        , stddev_input(stddev) {
        RelearnException::check(stddev > 0.0, "NormalBackgroundActivityCalculator::NormalBackgroundActivityCalculator: stddev was: {}", stddev);
    }

    virtual ~NormalBackgroundActivityCalculator() = default;

    /**
     * @brief Updates the input, providing normal or 0 input depending on the status of the neuron in the extra infos
     * @param step The current update step
     */
    void update_input([[maybe_unused]] const step_type step) override {
        const auto& disable_flags = extra_infos->get_disable_flags();
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
            "NormalBackgroundActivityCalculator::update_input: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
        for (number_neurons_type neuron_id = 0U; neuron_id < number_neurons; neuron_id++) {
            const auto input = disable_flags[neuron_id] == UpdateStatus::Disabled ? 0.0 : RandomHolder::get_random_normal_double(RandomHolderKey::BackgroundActivity, mean_input, stddev_input);
            set_background_activity(neuron_id, input);
        }
        Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<BackgroundActivityCalculator> clone() const override {
        return std::make_unique<NormalBackgroundActivityCalculator>(mean_input, stddev_input);
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() override {
        auto parameters = BackgroundActivityCalculator::get_parameter();
        parameters.emplace_back(Parameter<double>("Mean background activity", mean_input, BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean));
        parameters.emplace_back(Parameter<double>("Stddev background activity", stddev_input, BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev));

        return parameters;
    }

private:
    double mean_input{ default_background_activity_mean };
    double stddev_input{ default_background_activity_stddev };
};

/**
 * This class provides a normally distributed input, i.e., according to some N(expected, standard deviation).
 * However, it draws all input at the initialization phase and only returns pointers into that memory;
 * this speeds the update up enormously, but ignores if a neuron is disabled.
 */
class FastNormalBackgroundActivityCalculator : public BackgroundActivityCalculator {
public:
    /**
     * @brief Constructs a new object with the given normal input, i.e., from N(mean, stddev).
     *      It draws number_neurons*multiplier inputs initially and than just returns pointers.
     * @brief mean The mean input
     * @brief stddev The standard deviation, must be > 0.0
     * @param multiplier The factor how many more values should be drawn
     * @exception Throws a RelearnException if stddev <= 0.0
     */
    FastNormalBackgroundActivityCalculator(const double mean, const double stddev, const size_t multiplier)
        : BackgroundActivityCalculator()
        , mean_input(mean)
        , stddev_input(stddev)
        , multiplier(multiplier) {
        RelearnException::check(stddev > 0.0, "FastNormalBackgroundActivityCalculator::FastNormalBackgroundActivityCalculator: stddev was: {}", stddev);
        RelearnException::check(multiplier > 0, "FastNormalBackgroundActivityCalculator::FastNormalBackgroundActivityCalculator: multiplier was: 0", stddev);
    }

    virtual ~FastNormalBackgroundActivityCalculator() = default;

    /**
     * @brief Updates the input, providing constant to all neurons to speed up the calculations.
     * @param step The current update step
     */
    void update_input([[maybe_unused]] const step_type step) override {
        const auto number_neurons = get_number_neurons();
        const auto& disable_flags = extra_infos->get_disable_flags();
        RelearnException::check(disable_flags.size() == number_neurons,
            "FastNormalBackgroundActivityCalculator::update_input: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);

        const auto min_offset = 0;
        const auto max_offset = pre_drawn_values.size() - number_neurons;

        const auto new_offset = RandomHolder::get_random_uniform_integer<size_t>(RandomHolderKey::BackgroundActivity, min_offset, max_offset);
        offset = new_offset;

        Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
    }

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0
     */
    void init(const number_neurons_type number_neurons) override {
        BackgroundActivityCalculator::init(number_neurons);

        pre_drawn_values.resize(number_neurons * multiplier);

        for (NeuronID::value_type neuron_id = 0U; neuron_id < number_neurons * multiplier; neuron_id++) {
            const auto input = RandomHolder::get_random_normal_double(RandomHolderKey::BackgroundActivity, mean_input, stddev_input);
            pre_drawn_values[neuron_id] = input;
        }
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(const number_neurons_type number_neurons) override {
        const auto previous_number_neurons = get_number_neurons();

        BackgroundActivityCalculator::create_neurons(number_neurons);

        const auto now_number_neurons = get_number_neurons();
        pre_drawn_values.resize(now_number_neurons * multiplier);

        for (NeuronID::value_type neuron_id = previous_number_neurons * multiplier; neuron_id < now_number_neurons * multiplier; neuron_id++) {
            const auto input = RandomHolder::get_random_normal_double(RandomHolderKey::BackgroundActivity, mean_input, stddev_input);
            pre_drawn_values[neuron_id] = input;
        }
    }

    /**
     * @brief Returns the calculated background activity for the given neuron. Changes after calls to update_input(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The background activity for the given neuron
     */
    double get_background_activity(const NeuronID neuron_id) const override {
        const auto number_neurons = get_number_neurons();
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_neurons, "FastNormalBackgroundActivityCalculator::get_background_activity: id is too large: {}", neuron_id);
        return pre_drawn_values[offset + local_neuron_id];
    }

    /**
     * @brief Returns the calculated background activity for all. Changes after calls to update_input(...)
     * @return The background activity for all neurons
     */
    std::span<const double> get_background_activity() const noexcept override {
        const auto number_neurons = get_number_neurons();

        const auto pointer = pre_drawn_values.data();

        return std::span<const double>{ pointer + offset, number_neurons };
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<BackgroundActivityCalculator> clone() const override {
        return std::make_unique<FastNormalBackgroundActivityCalculator>(mean_input, stddev_input, multiplier);
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter() override {
        auto parameters = BackgroundActivityCalculator::get_parameter();
        parameters.emplace_back(Parameter<double>("Mean background activity", mean_input, BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean));
        parameters.emplace_back(Parameter<double>("Stddev background activity", stddev_input, BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev));

        return parameters;
    }

private:
    double mean_input{ default_background_activity_mean };
    double stddev_input{ default_background_activity_stddev };
    size_t multiplier{ 1 };
    size_t offset{ 0 };

    std::vector<double> pre_drawn_values{};
};
