/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_synapse_creation_request.h"

#include "neurons/helper/SynapseCreationRequests.h"

TEST_F(SynapseCreationTest, testDefaultConstructor) {
    SynapseCreationRequest scr{};

    const auto& target_neuron_id = scr.get_target();
    const auto& source_neuron_id = scr.get_source();
    const auto& signal_type = scr.get_signal_type();

    ASSERT_FALSE(target_neuron_id.is_initialized());
    ASSERT_FALSE(source_neuron_id.is_initialized());

    ASSERT_EQ(signal_type, SignalType{});
}

TEST_F(SynapseCreationTest, testConstructor) {
    const auto& golden_target_neuron_id = get_random_neuron_id(10000);
    const auto& golden_source_neuron_id = get_random_neuron_id(10000);
    const auto& golden_signal_type = get_random_signal_type();

    SynapseCreationRequest scr{ golden_target_neuron_id, golden_source_neuron_id, golden_signal_type };

    const auto& target_neuron_id = scr.get_target();
    const auto& source_neuron_id = scr.get_source();
    const auto& signal_type = scr.get_signal_type();

    ASSERT_EQ(target_neuron_id, golden_target_neuron_id);
    ASSERT_EQ(source_neuron_id, golden_source_neuron_id);

    ASSERT_EQ(signal_type, golden_signal_type);
}

TEST_F(SynapseCreationTest, testConstructorException) {
    const auto& golden_signal_type = get_random_signal_type();

    const auto& dummy_neuron_id = get_random_neuron_id(10000);

    ASSERT_THROW(SynapseCreationRequest scr(NeuronID::virtual_id(), dummy_neuron_id, golden_signal_type), RelearnException);
    ASSERT_THROW(SynapseCreationRequest scr(NeuronID::uninitialized_id(), dummy_neuron_id, golden_signal_type), RelearnException);

    ASSERT_THROW(SynapseCreationRequest scr(dummy_neuron_id, NeuronID::virtual_id(), golden_signal_type), RelearnException);
    ASSERT_THROW(SynapseCreationRequest scr(dummy_neuron_id, NeuronID::uninitialized_id(), golden_signal_type), RelearnException);

    ASSERT_THROW(SynapseCreationRequest scr(NeuronID::virtual_id(), NeuronID::virtual_id(), golden_signal_type), RelearnException);
    ASSERT_THROW(SynapseCreationRequest scr(NeuronID::virtual_id(), NeuronID::uninitialized_id(), golden_signal_type), RelearnException);

    ASSERT_THROW(SynapseCreationRequest scr(NeuronID::uninitialized_id(), NeuronID::virtual_id(), golden_signal_type), RelearnException);
    ASSERT_THROW(SynapseCreationRequest scr(NeuronID::uninitialized_id(), NeuronID::uninitialized_id(), golden_signal_type), RelearnException);
}

TEST_F(SynapseCreationTest, testStructuredBinding) {
    const auto& golden_target_neuron_id = get_random_neuron_id(10000);
    const auto& golden_source_neuron_id = get_random_neuron_id(10000);
    const auto& golden_signal_type = get_random_signal_type();

    SynapseCreationRequest scr{ golden_target_neuron_id, golden_source_neuron_id, golden_signal_type };

    const auto& [target_neuron_id, source_neuron_id, signal_type] = scr;

    ASSERT_EQ(target_neuron_id, golden_target_neuron_id);
    ASSERT_EQ(source_neuron_id, golden_source_neuron_id);

    ASSERT_EQ(signal_type, golden_signal_type);
}
