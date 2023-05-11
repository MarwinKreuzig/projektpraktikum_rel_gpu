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

#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/input/BackgroundActivityCalculator.h"
#include "neurons/input/BackgroundActivityCalculators.h"
#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"

#include <algorithm>
#include <memory>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

void test_background_equality(const std::unique_ptr<BackgroundActivityCalculator>& background_calculator) {
    const auto number_neurons = background_calculator->get_number_neurons();
    const auto& inputs = background_calculator->get_background_activity();

    ASSERT_EQ(inputs.size(), number_neurons);

    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto input = background_calculator->get_background_activity(neuron_id);
        ASSERT_EQ(inputs[neuron_id.get_neuron_id()], input);
    }
}

void test_background_exceptions(const std::unique_ptr<BackgroundActivityCalculator>& background_calculator) {
    const auto number_neurons = background_calculator->get_number_neurons();

    for (const auto& neuron_id : NeuronID::range_id(number_neurons) | ranges::views::transform(plus(number_neurons))) {
        ASSERT_THROW(const auto input = background_calculator->get_background_activity(NeuronID{ neuron_id }), RelearnException);
    }
}

void test_init_create(const std::unique_ptr<BackgroundActivityCalculator>& background_calculator, const size_t number_init_neurons, const size_t number_create_neurons, const bool check_input = true) {
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

    if (check_input) {
        for (const auto& neuron_id : NeuronID::range(number_init_neurons)) {
            const auto input = background_calculator->get_background_activity(neuron_id);
            ASSERT_EQ(input, 0.0);
        }
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

    if (check_input) {
        for (const auto& neuron_id : NeuronID::range(number_init_neurons + number_create_neurons)) {
            const auto input = background_calculator->get_background_activity(neuron_id);
            ASSERT_EQ(input, 0.0);
        }
    }

    ASSERT_EQ(first_clone->get_number_neurons(), 0);
    ASSERT_TRUE(first_clone->get_background_activity().empty());

    ASSERT_EQ(second_clone->get_number_neurons(), 0);
    ASSERT_TRUE(second_clone->get_background_activity().empty());
}

TEST_F(BackgroundActivityTest, testNullBackgroundActivityConstruct) {
    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NullBackgroundActivityCalculator>(std::make_unique<IdentityTransformation>(), 0, std::numeric_limits<RelearnTypes::step_type>::max());

    const auto number_neurons_init = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdAdapter::get_random_number_neurons(mt);

    test_init_create(background_calculator, number_neurons_init, number_neurons_create);
}

TEST_F(BackgroundActivityTest, testConstantBackgroundActivityConstruct) {
    const auto constant_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_base_background_activity, BackgroundActivityCalculator::max_base_background_activity, mt);

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<ConstantBackgroundActivityCalculator>(std::make_unique<IdentityTransformation>(),0, std::numeric_limits<RelearnTypes::step_type>::max(),constant_background);

    const auto number_neurons_init = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdAdapter::get_random_number_neurons(mt);

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

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NormalBackgroundActivityCalculator>(std::make_unique<IdentityTransformation>(),0, std::numeric_limits<RelearnTypes::step_type>::max(),mean_background, stddev_background);

    const auto number_neurons_init = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdAdapter::get_random_number_neurons(mt);

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

TEST_F(BackgroundActivityTest, testFastNormalBackgroundActivityConstruct) {
    const auto mean_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean, mt);
    const auto stddev_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev, mt);

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<FastNormalBackgroundActivityCalculator>(std::make_unique<IdentityTransformation>(),0, std::numeric_limits<RelearnTypes::step_type>::max(),mean_background, stddev_background, 5);

    const auto number_neurons_init = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = NeuronIdAdapter::get_random_number_neurons(mt);

    test_init_create(background_calculator, number_neurons_init, number_neurons_create, false);

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
    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NullBackgroundActivityCalculator>(std::make_unique<IdentityTransformation>(),0, std::numeric_limits<RelearnTypes::step_type>::max());

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    background_calculator->init(number_neurons);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    background_calculator->set_extra_infos(extra_info);

    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
            extra_info->set_disabled_neurons(std::vector{ NeuronID{ neuron_id } });
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);
    background_calculator->update_input(step);

    test_background_equality(background_calculator);

    const auto& background_input = background_calculator->get_background_activity();
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        ASSERT_EQ(background_input[neuron_id], 0.0);
    }
}

TEST_F(BackgroundActivityTest, testConstantBackgroundActivityUpdate) {
    const auto constant_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_base_background_activity, BackgroundActivityCalculator::max_base_background_activity, mt);
    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<ConstantBackgroundActivityCalculator>(std::make_unique<IdentityTransformation>(),0, std::numeric_limits<RelearnTypes::step_type>::max(),constant_background);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    background_calculator->init(number_neurons);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    background_calculator->set_extra_infos(extra_info);

    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
            extra_info->set_disabled_neurons(std::vector{ NeuronID{ neuron_id } });
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);
    background_calculator->update_input(step);

    test_background_equality(background_calculator);

    const auto& background_input = background_calculator->get_background_activity();
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
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

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<NormalBackgroundActivityCalculator>(std::make_unique<IdentityTransformation>(),0, std::numeric_limits<RelearnTypes::step_type>::max(),mean_background, stddev_background);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    background_calculator->init(number_neurons);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    background_calculator->set_extra_infos(extra_info);

    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
            extra_info->set_disabled_neurons(std::vector{ NeuronID{ neuron_id } });
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);
    background_calculator->update_input(step);

    test_background_equality(background_calculator);

    std::vector<double> background_values{};
    background_values.reserve(number_neurons);

    auto number_enabled_neurons = 0;
    const auto& background_input = background_calculator->get_background_activity();
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(background_input[neuron_id], 0.0);
        } else {
            background_values.emplace_back(background_input[neuron_id] - mean_background);
            number_enabled_neurons++;
        }
    }

    const auto summed_background = ranges::accumulate(background_values, 0.0);

    if (std::abs(summed_background) >= eps * number_enabled_neurons) {
        std::cerr << "The total variance was: " << std::abs(summed_background) << '\n';
        std::cerr << "That's more than " << eps << " * " << number_enabled_neurons << '\n';
        // TODO(future): Insert some statistical test here
    }
}

TEST_F(BackgroundActivityTest, testFastNormalBackgroundActivityUpdate) {
    const auto mean_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_mean, BackgroundActivityCalculator::max_background_activity_mean, mt);
    const auto stddev_background = RandomAdapter::get_random_double<double>(BackgroundActivityCalculator::min_background_activity_stddev, BackgroundActivityCalculator::max_background_activity_stddev, mt);

    std::unique_ptr<BackgroundActivityCalculator> background_calculator = std::make_unique<FastNormalBackgroundActivityCalculator>(std::make_unique<IdentityTransformation>(),0, std::numeric_limits<RelearnTypes::step_type>::max(),mean_background, stddev_background, 5);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    background_calculator->init(number_neurons);

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);

    background_calculator->set_extra_infos(extra_info);

    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);
    for (const auto neuron_id : NeuronID::range_id(number_neurons)) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
            extra_info->set_disabled_neurons(std::vector{ NeuronID{ neuron_id } });
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);
    background_calculator->update_input(step);

    test_background_equality(background_calculator);

    const auto& background_input = background_calculator->get_background_activity();

    const auto background_values = NeuronID::range_id(number_neurons)
        | ranges::views::filter(not_equal_to(UpdateStatus::Disabled), lookup(update_status))
        | ranges::views::transform([&background_input, mean_background](const auto neuron_id) { return background_input[neuron_id] - mean_background; })
        | ranges::to_vector;

    const auto number_enabled_neurons = ranges::size(background_values);
    const auto summed_background
        = ranges::accumulate(background_values, 0.0);

    if (std::abs(summed_background) >= eps * number_enabled_neurons) {
        std::cerr << "The total variance was: " << std::abs(summed_background) << '\n';
        std::cerr << "That's more than " << eps << " * " << number_enabled_neurons << '\n';
        // TODO(future): Insert some statistical test here
    }
}

TEST_F(BackgroundActivityTest, testIdentityTransformation) {
    IdentityTransformation transformation;

    const auto num_steps = RandomAdapter::get_random_integer(100, 9999999, mt);
    for(auto i=0;i<num_steps;i++) {
        const auto value = RandomAdapter::get_random_double(-100,100,mt);
        ASSERT_EQ(transformation.transform(i, value), value);
    }
}

TEST_F(BackgroundActivityTest, testLinearTransformation) {
    const auto num_steps = RandomAdapter::get_random_integer(100, 1000, mt);

    LinearTransformation transformation{-1.0/num_steps, 1, 0};

    const auto value = RandomAdapter::get_random_double(1.0,100.0,mt);
    auto before = value;
    for(auto i=1;i<num_steps;i++) {
        const auto transformed = transformation.transform(i, value);
        ASSERT_LT(transformed, value);
        ASSERT_LT(transformed, before);
        ASSERT_GT(transformed, 0);
        before = transformed;
    }

    ASSERT_EQ(transformation.transform(num_steps+1, value), 0);
}
