/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_transmission_delay.h"

#include "neurons/neuron_types_adapter.h"
#include "tagged_id/tagged_id_adapter.h"

#include "neurons/input/TransmissionDelayer.h"

#include <unordered_set>
TEST_F(TransmissionDelayTest, testNoDelay) {
    TransmissionDelayer delayer;

    const auto num_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& fired_status = NeuronTypesAdapter::get_fired_status(num_neurons, mt);

    const auto delayed_fired_status = delayer.apply_delay(fired_status);
    ASSERT_EQ(fired_status, delayed_fired_status);
}

TEST_F(TransmissionDelayTest, testDelay) {

    const auto delay = RandomAdapter::get_random_integer(1U,50U, mt);
    TransmissionDelayer delayer(delay);

    const auto num_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    std::vector<std::vector<FiredStatus>> fired_status_list;
    for(int i=0;i<delay;i++) {
        const auto &fired_status = NeuronTypesAdapter::get_fired_status(num_neurons, mt);
        fired_status_list.push_back(fired_status);
    }

    std::vector<FiredStatus> empty;
    empty.resize(num_neurons, FiredStatus::Inactive);

    for(int i=0;i<delay;i++) {
        const auto& delayed = delayer.apply_delay(fired_status_list[i]);
        ASSERT_EQ(empty, delayed);
    }

    for(int i=0;i<delay;i++) {
        const auto &fired_status = NeuronTypesAdapter::get_fired_status(num_neurons, mt);
        const auto& delayed = delayer.apply_delay(fired_status);
        ASSERT_EQ(fired_status_list[i], delayed);
    }

}