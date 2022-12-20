/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_background_activity.h"

#include "tagged_id/tagged_id_adapter.h"

#include "neurons/input/BackgroundActivityCalculator.h"
#include "neurons/input/BackgroundActivityCalculators.h"

#include <algorithm>
#include <memory>

void test_background_equality(const std::unique_ptr<BackgroundActivityCalculator>& background_calculator) {
    const auto number_neurons = background_calculator->get_number_neurons();
    const auto& inputs = background_calculator->get_background_activity();

    ASSERT_EQ(inputs.size(), number_neurons);

    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        const NeuronID id{ neuron_id };

        const auto input = background_calculator->get_background_activity(id);
        ASSERT_EQ(inputs[neuron_id], input);
    }
}

void test_background_exceptions(const std::unique_ptr<BackgroundActivityCalculator>& background_calculator) {
    const auto number_neurons = background_calculator->get_number_neurons();

    for (size_t it = 0; it < number_neurons + 100; it++) {
        const NeuronID id{ it + number_neurons };
        ASSERT_THROW(const auto input = background_calculator->get_background_activity(id), RelearnException);
    }
}

void test_init_create(const std::unique_ptr<BackgroundActivityCalculator>& background_calculator, const size_t number_init_neurons, const size_t number_create_neurons) {
    ASSERT_EQ(background_calculator->get_number_neurons(), 0);
    ASSERT_TRUE(background_calculator->get_background_activity().empty());

    test_background_exceptions(background_calculator);

    auto first_clone = background_calculator->clone();
    ASSERT_EQ(first_clone->get_number_neurons(), 0);
    ASSERT_TRUE(first_clone->get_background_activity().empty());

    ASSERT_THROW(background_calculator->init(0), RelearnException);
    background_calculator->init(number_init_neurons);

    ASSERT_EQ(background_calculator->get_number_neurons(), number_init_neurons);
    ASSERT_EQ(background_calculator->get_background_activity().size(), number_init_neurons);
    ASSERT_EQ(first_clone->get_number_neurons(), 0);
    ASSERT_TRUE(first_clone->get_background_activity().empty());

    test_background_equality(background_calculator);
    test_background_exceptions(background_calculator);

    for (size_t neuron_id = 0; neuron_id < number_init_neurons; neuron_id++) {
        const NeuronID id{ neuron_id };

        const auto input = background_calculator->get_background_activity(id);
        ASSERT_EQ(input, 0.0);
    }

    auto second_clone = background_calculator->clone();
    ASSERT_EQ(second_clone->get_number_neurons(), 0);
    ASSERT_TRUE(second_clone->get_background_activity().empty());

    ASSERT_THROW(background_calculator->create_neurons(0), RelearnException);
    background_calculator->create_neurons(number_create_neurons);

    ASSERT_EQ(background_calculator->get_number_neurons(), number_init_neurons + number_create_neurons);
    ASSERT_EQ(background_calculator->get_background_activity().size(), number_init_neurons + number_create_neurons);

    test_background_equality(background_calculator);
    test_background_exceptions(background_calculator);

    for (size_t neuron_id = 0; neuron_id < number_init_neurons + number_create_neurons; neuron_id++) {
        const NeuronID id{ neuron_id };

        const auto input = background_calculator->get_background_activity(id);
        ASSERT_EQ(input, 0.0);
    }

    ASSERT_EQ(first_clone->get_number_neurons(), 0);
    ASSERT_TRUE(first_clone->get_background_activity().empty());

    ASSERT_EQ(second_clone->get_number_neurons(), 0);
    ASSERT_TRUE(second_clone->get_background_activity().empty());
}

TEST_F(BackgroundActivityTest, testNullBackgroundActivityConstruct) {
    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NullBackgroundActivityCalculator>();

    const auto number_neurons_init = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = TaggedIdAdapter::get_random_number_neurons(mt);

    test_init_create(background_calculator, number_neurons_init, number_neurons_create);
}

TEST_F(BackgroundActivityTest, testConstantBackgroundActivityConstruct) {
    const auto constant_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_base_background_activity, BackgroundActivityCalculator::max_base_background_activity, mt);

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<ConstantBackgroundActivityCalculator>(constant_background);

    const auto number_neurons_init = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = TaggedIdAdapter::get_random_number_neurons(mt);

    test_init_create(background_calculator, number_neurons_init, number_neurons_create);

    const auto& parameters = background_calculator->get_parameter();
    ASSERT_EQ(parameters.size(), 1);

    ModelParameter mp = parameters[0];
    Parameter<double> param1 = std::get<Parameter<double>>(mp);

    ASSERT_EQ(param1.min(), BackgroundActivityCalculator::min_base_background_activity);
    ASSERT_EQ(param1.max(), BackgroundActivityCalculator::max_base_background_activity);
    ASSERT_EQ(param1.value(), constant_background);
}

TEST_F(BackgroundActivityTest, testNormalBackgroundActivityConstruct) {
    const auto mean_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean, mt);
    const auto stddev_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev, mt);

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NormalBackgroundActivityCalculator>(mean_background, stddev_background);

    const auto number_neurons_init = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = TaggedIdAdapter::get_random_number_neurons(mt);

    test_init_create(background_calculator, number_neurons_init, number_neurons_create);

    const auto& parameters = background_calculator->get_parameter();
    ASSERT_EQ(parameters.size(), 2);

    ModelParameter mp1 = parameters[0];
    Parameter<double> param1 = std::get<Parameter<double>>(mp1);

    ASSERT_EQ(param1.min(), BackgroundActivityCalculator::min_background_activity_mean);
    ASSERT_EQ(param1.max(), BackgroundActivityCalculator::max_background_activity_mean);
    ASSERT_EQ(param1.value(), mean_background);

    ModelParameter mp2 = parameters[1];
    Parameter<double> param2 = std::get<Parameter<double>>(mp2);

    ASSERT_EQ(param2.min(), BackgroundActivityCalculator::min_background_activity_stddev);
    ASSERT_EQ(param2.max(), BackgroundActivityCalculator::max_background_activity_stddev);
    ASSERT_EQ(param2.value(), stddev_background);
}

TEST_F(BackgroundActivityTest, testNullBackgroundActivityUpdate) {
    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NullBackgroundActivityCalculator>();

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    background_calculator->init(number_neurons);

    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);
    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);
    background_calculator->update_input(step, update_status);

    test_background_equality(background_calculator);

    const auto& background_input = background_calculator->get_background_activity();
    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        ASSERT_EQ(background_input[neuron_id], 0.0);
    }
}

TEST_F(BackgroundActivityTest, testConstantBackgroundActivityUpdate) {
    const auto constant_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_base_background_activity, BackgroundActivityCalculator::max_base_background_activity, mt);
    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<ConstantBackgroundActivityCalculator>(constant_background);

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    background_calculator->init(number_neurons);

    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);
    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);
    background_calculator->update_input(step, update_status);

    test_background_equality(background_calculator);

    const auto& background_input = background_calculator->get_background_activity();
    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(background_input[neuron_id], 0.0);
        } else {
            ASSERT_EQ(background_input[neuron_id], constant_background);
        }
    }
}

TEST_F(BackgroundActivityTest, testNormalBackgroundActivityUpdate) {
    const auto mean_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean, mt);
    const auto stddev_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev, mt);

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NormalBackgroundActivityCalculator>(mean_background, stddev_background);

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    background_calculator->init(number_neurons);

    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);
    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);
    background_calculator->update_input(step, update_status);

    test_background_equality(background_calculator);

    std::vector<double> background_values{};
    background_values.reserve(number_neurons);

    auto number_enabled_neurons = 0;
    const auto& background_input = background_calculator->get_background_activity();
    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(background_input[neuron_id], 0.0);
        } else {
            background_values.emplace_back(background_input[neuron_id] - mean_background);
            number_enabled_neurons++;
        }
    }

    const auto summed_background = std::reduce(background_values.begin(), background_values.end());

    if (std::abs(summed_background) >= eps * number_enabled_neurons) {
        std::cerr << "The total variance was: " << std::abs(summed_background) << '\n';
        std::cerr << "That's more than " << eps << " * " << number_enabled_neurons << '\n';
        // TODO(future): Insert some statistical test here
    }
}
