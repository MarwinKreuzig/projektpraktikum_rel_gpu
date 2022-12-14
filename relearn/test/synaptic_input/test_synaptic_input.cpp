/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_synaptic_input.h"

#include "RandomAdapter.h"

#include "network_graph/network_graph_adapter.h"
#include "tagged_id/tagged_id_adapter.h"

#include "neurons/models/SynapticInputCalculator.h"
#include "neurons/models/SynapticInputCalculators.h"

#include <memory>

void test_input_equality(const std::unique_ptr<SynapticInputCalculator>& input_calculator) {
    const auto number_neurons = input_calculator->get_number_neurons();
    const auto& inputs = input_calculator->get_synaptic_input();

    ASSERT_EQ(inputs.size(), number_neurons);

    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        const NeuronID id{ neuron_id };

        const auto input = input_calculator->get_synaptic_input(id);
        ASSERT_EQ(inputs[neuron_id], input);
    }
}

void test_input_exceptions(const std::unique_ptr<SynapticInputCalculator>& input_calculator) {
    const auto number_neurons = input_calculator->get_number_neurons();

    for (size_t it = 0; it < number_neurons + 100; it++) {
        const NeuronID id{ it + number_neurons };
        ASSERT_THROW(const auto input = input_calculator->get_synaptic_input(id), RelearnException);
    }
}

void test_constructor_clone(const std::unique_ptr<SynapticInputCalculator>& input_calculator, const double expected_value, const auto new_value) {
    const auto original_value = input_calculator->get_synapse_conductance();
    ASSERT_EQ(original_value, expected_value);

    const auto clone = input_calculator->clone();
    const auto cloned_value = clone->get_synapse_conductance();

    ASSERT_EQ(cloned_value, expected_value);

    auto parameter = input_calculator->get_parameter()[0];
    auto cast_parameter = std::get<Parameter<double>>(parameter);

    ASSERT_EQ(cast_parameter.value(), expected_value);

    cast_parameter.set_value(new_value);

    ASSERT_EQ(new_value, input_calculator->get_synapse_conductance());
    ASSERT_EQ(expected_value, clone->get_synapse_conductance());

    ASSERT_EQ(SynapticInputCalculator::min_conductance, cast_parameter.min());
    ASSERT_EQ(SynapticInputCalculator::max_conductance, cast_parameter.max());
}

void test_init_create(const std::unique_ptr<SynapticInputCalculator>& input_calculator, const size_t number_init_neurons, const size_t number_create_neurons) {
    ASSERT_EQ(input_calculator->get_number_neurons(), 0);
    ASSERT_TRUE(input_calculator->get_synaptic_input().empty());

    test_input_exceptions(input_calculator);

    auto first_clone = input_calculator->clone();
    ASSERT_EQ(first_clone->get_number_neurons(), 0);
    ASSERT_TRUE(first_clone->get_synaptic_input().empty());

    ASSERT_THROW(input_calculator->init(0), RelearnException);
    input_calculator->init(number_init_neurons);

    ASSERT_EQ(input_calculator->get_number_neurons(), number_init_neurons);
    ASSERT_EQ(input_calculator->get_synaptic_input().size(), number_init_neurons);
    ASSERT_EQ(first_clone->get_number_neurons(), 0);
    ASSERT_TRUE(first_clone->get_synaptic_input().empty());

    test_input_equality(input_calculator);
    test_input_exceptions(input_calculator);

    for (size_t neuron_id = 0; neuron_id < number_init_neurons; neuron_id++) {
        const NeuronID id{ neuron_id };

        const auto input = input_calculator->get_synaptic_input(id);
        ASSERT_EQ(input, 0.0);
    }

    auto second_clone = input_calculator->clone();
    ASSERT_EQ(second_clone->get_number_neurons(), 0);
    ASSERT_TRUE(second_clone->get_synaptic_input().empty());

    ASSERT_THROW(input_calculator->create_neurons(0), RelearnException);
    input_calculator->create_neurons(number_create_neurons);

    ASSERT_EQ(input_calculator->get_number_neurons(), number_init_neurons + number_create_neurons);
    ASSERT_EQ(input_calculator->get_synaptic_input().size(), number_init_neurons + number_create_neurons);

    test_input_equality(input_calculator);
    test_input_exceptions(input_calculator);

    for (size_t neuron_id = 0; neuron_id < number_init_neurons + number_create_neurons; neuron_id++) {
        const NeuronID id{ neuron_id };

        const auto input = input_calculator->get_synaptic_input(id);
        ASSERT_EQ(input, 0.0);
    }

    ASSERT_EQ(first_clone->get_number_neurons(), 0);
    ASSERT_TRUE(first_clone->get_synaptic_input().empty());

    ASSERT_EQ(second_clone->get_number_neurons(), 0);
    ASSERT_TRUE(second_clone->get_synaptic_input().empty());
}

TEST_F(SynapticInputTest, testLinearSynapticInputConstruct) {
    const auto random_k = RandomAdapter::get_random_double<double>(SynapticInputCalculator::min_conductance, SynapticInputCalculator::max_conductance, mt);

    std::unique_ptr<SynapticInputCalculator> input_calculator = std::make_unique<LinearSynapticInputCalculator>(random_k);
    ASSERT_EQ(input_calculator->get_synapse_conductance(), random_k);

    const auto new_k = RandomAdapter::get_random_double<double>(SynapticInputCalculator::min_conductance, SynapticInputCalculator::max_conductance, mt);
    test_constructor_clone(input_calculator, random_k, new_k);

    const auto number_neurons_init = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = TaggedIdAdapter::get_random_number_neurons(mt);

    test_init_create(input_calculator, number_neurons_init, number_neurons_create);
}

TEST_F(SynapticInputTest, testLinearSynapticInputUpdateEmptyGraph) {
    const auto random_k = RandomAdapter::get_random_double<double>(SynapticInputCalculator::min_conductance, SynapticInputCalculator::max_conductance, mt);
    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);

    std::unique_ptr<SynapticInputCalculator> input_calculator = std::make_unique<LinearSynapticInputCalculator>(random_k);
    input_calculator->init(number_neurons);

    NetworkGraph ng_plastic(number_neurons, MPIRank::root_rank());
    NetworkGraph ng_static(number_neurons, MPIRank::root_rank());
    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);
    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);

    input_calculator->update_input(0, ng_static, ng_plastic, fired_status, update_status);

    for (const auto& value : input_calculator->get_synaptic_input()) {
        ASSERT_EQ(0.0, value);
    }

    test_input_equality(input_calculator);
}

TEST_F(SynapticInputTest, testLinearSynapticInputUpdate) {
    const auto random_k = RandomAdapter::get_random_double<double>(SynapticInputCalculator::min_conductance, SynapticInputCalculator::max_conductance, mt);

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto num_synapses = NetworkGraphAdapter::get_random_number_synapses(mt) + number_neurons;

    std::unique_ptr<SynapticInputCalculator> input_calculator = std::make_unique<LinearSynapticInputCalculator>(random_k);
    input_calculator->init(number_neurons);

    NetworkGraph ng_plastic(number_neurons, MPIRank::root_rank());
    NetworkGraph ng_static(number_neurons, MPIRank::root_rank());
    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);
    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);

    for (size_t synapse_id = 0; synapse_id < num_synapses; synapse_id++) {
        const auto weight = std::abs(NetworkGraphAdapter::get_random_synapse_weight(mt));
        const auto source_id = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto target_id = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);

        ng_plastic.add_synapse(LocalSynapse(target_id, source_id, weight));
    }

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
        }
        if (RandomAdapter::get_random_bool(mt)) {
            fired_status[neuron_id] = FiredStatus::Fired;
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);

    input_calculator->update_input(step, ng_static, ng_plastic, fired_status, update_status);

    const auto& inputs = input_calculator->get_synaptic_input();

    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(inputs[neuron_id], 0.0);
            continue;
        }

        auto total_input = 0.0;

        for (const auto& [other_id, weight] : ng_plastic.get_all_in_edges(NeuronID(neuron_id))) {
            if (fired_status[other_id.get_neuron_id().get_neuron_id()] == FiredStatus::Inactive) {
                continue;
            }

            total_input += (weight * random_k);
        }

        ASSERT_EQ(total_input, inputs[neuron_id]);
    }

    test_input_equality(input_calculator);
}

TEST_F(SynapticInputTest, testLogarithmicSynapticInputConstruct) {
    const auto random_k = RandomAdapter::get_random_double<double>(SynapticInputCalculator::min_conductance, SynapticInputCalculator::max_conductance, mt);

    std::unique_ptr<SynapticInputCalculator> input_calculator = std::make_unique<LogarithmicSynapticInputCalculator>(random_k);
    ASSERT_EQ(input_calculator->get_synapse_conductance(), random_k);

    const auto new_k = RandomAdapter::get_random_double<double>(SynapticInputCalculator::min_conductance, SynapticInputCalculator::max_conductance, mt);
    test_constructor_clone(input_calculator, random_k, new_k);

    const auto number_neurons_init = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto number_neurons_create = TaggedIdAdapter::get_random_number_neurons(mt);

    test_init_create(input_calculator, number_neurons_init, number_neurons_create);
}

TEST_F(SynapticInputTest, testLogarithmicSynapticInputUpdateEmptyGraph) {
    const auto random_k = RandomAdapter::get_random_double<double>(SynapticInputCalculator::min_conductance, SynapticInputCalculator::max_conductance, mt);
    const auto number_neurons_init = TaggedIdAdapter::get_random_number_neurons(mt);

    std::unique_ptr<SynapticInputCalculator> input_calculator = std::make_unique<LogarithmicSynapticInputCalculator>(random_k);
    input_calculator->init(number_neurons_init);

    NetworkGraph ng_plastic(number_neurons_init, MPIRank::root_rank());
    NetworkGraph ng_static(number_neurons_init, MPIRank::root_rank());

    std::vector<FiredStatus> fired_status(number_neurons_init, FiredStatus::Inactive);
    std::vector<UpdateStatus> update_status(number_neurons_init, UpdateStatus::Enabled);

    input_calculator->update_input(0, ng_static, ng_plastic, fired_status, update_status);

    for (const auto& value : input_calculator->get_synaptic_input()) {
        ASSERT_EQ(0.0, value);
    }

    test_input_equality(input_calculator);
}

TEST_F(SynapticInputTest, testLogarithmicSynapticInputUpdate) {
    const auto random_k = RandomAdapter::get_random_double<double>(SynapticInputCalculator::min_conductance, SynapticInputCalculator::max_conductance, mt);

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto num_synapses = NetworkGraphAdapter::get_random_number_synapses(mt) + number_neurons;

    std::unique_ptr<SynapticInputCalculator> input_calculator = std::make_unique<LogarithmicSynapticInputCalculator>(random_k);
    input_calculator->init(number_neurons);

    NetworkGraph ng_static(number_neurons, MPIRank::root_rank());
    NetworkGraph ng_plastic(number_neurons, MPIRank::root_rank());

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);
    std::vector<UpdateStatus> update_status(number_neurons, UpdateStatus::Enabled);

    for (size_t synapse_id = 0; synapse_id < num_synapses; synapse_id++) {
        const auto weight = std::abs(NetworkGraphAdapter::get_random_synapse_weight(mt));
        const auto source_id = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);
        const auto target_id = TaggedIdAdapter::get_random_neuron_id(number_neurons, mt);

        ng_plastic.add_synapse(LocalSynapse(target_id, source_id, weight));
    }

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        if (RandomAdapter::get_random_bool(mt)) {
            update_status[neuron_id] = UpdateStatus::Disabled;
        }
        if (RandomAdapter::get_random_bool(mt)) {
            fired_status[neuron_id] = FiredStatus::Fired;
        }
    }

    const auto step = RandomAdapter::get_random_integer<RelearnTypes::step_type>(0, 1000000, mt);

    input_calculator->update_input(step, ng_static, ng_plastic, fired_status, update_status);

    const auto& inputs = input_calculator->get_synaptic_input();

    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        if (update_status[neuron_id] == UpdateStatus::Disabled) {
            ASSERT_EQ(inputs[neuron_id], 0.0);
            continue;
        }

        auto total_input = 0.0;

        for (const auto& [other_id, weight] : ng_plastic.get_all_in_edges(NeuronID(neuron_id))) {
            if (fired_status[other_id.get_neuron_id().get_neuron_id()] == FiredStatus::Inactive) {
                continue;
            }

            total_input += (weight * random_k);
        }

        const auto scaled_input = std::log(total_input + 1.0);

        ASSERT_EQ(scaled_input, inputs[neuron_id]);
    }

    test_input_equality(input_calculator);
}
