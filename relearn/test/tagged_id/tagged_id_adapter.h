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

#include "RandomAdapter.h"

#include "util/TaggedID.h"

#include <random>

class TaggedIdAdapter {
public:
    constexpr static int upper_bound_num_neurons = 1000;

    static NeuronID::value_type get_random_number_neurons(std::mt19937& mt) {
        return RandomAdapter::get_random_integer<NeuronID::value_type>(1, upper_bound_num_neurons, mt);
    }

    static NeuronID get_random_neuron_id(std::mt19937& mt) {
        const auto value = RandomAdapter::get_random_integer<NeuronID::value_type>(0, upper_bound_num_neurons - 1, mt);
        return NeuronID{ value };
    }

    static NeuronID get_random_neuron_id(NeuronID::value_type number_neurons, std::mt19937& mt) {
        const auto value = RandomAdapter::get_random_integer<NeuronID::value_type>(0, number_neurons - 1, mt);
        return NeuronID{ value };
    }

    static NeuronID get_random_neuron_id(NeuronID::value_type number_neurons, NeuronID::value_type offset, std::mt19937& mt) {
        const auto value = RandomAdapter::get_random_integer<NeuronID::value_type>(offset, offset + number_neurons - 1, mt);
        return NeuronID{ value };
    }

    static NeuronID get_random_neuron_id(NeuronID::value_type number_neurons, NeuronID except, std::mt19937& mt) {
        NeuronID nid{};
        do {
            nid = get_random_neuron_id(number_neurons, mt);
        } while (nid == except);
        return nid;
    }
};
