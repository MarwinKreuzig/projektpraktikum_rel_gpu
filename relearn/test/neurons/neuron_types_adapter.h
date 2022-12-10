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

#include "neurons/ElementType.h"
#include "neurons/SignalType.h"
#include "neurons/helper/DistantNeuronRequests.h"

#include <random>

class NeuronTypesAdapter {
public:
    static ElementType get_random_element_type(std::mt19937& mt) noexcept {
        return RandomAdapter::get_random_bool(mt) ? ElementType::Axon : ElementType::Dendrite;
    }

    static SignalType get_random_signal_type(std::mt19937& mt) noexcept {
        return RandomAdapter::get_random_bool(mt) ? SignalType::Excitatory : SignalType::Inhibitory;
    }

    static DistantNeuronRequest::TargetNeuronType get_random_target_neuron_type(std::mt19937& mt) {
        const auto drawn = RandomAdapter::get_random_integer<int>(0, 2, mt);

        if (drawn == 0) {
            return DistantNeuronRequest::TargetNeuronType::BranchNode;
        }

        if (drawn == 1) {
            return DistantNeuronRequest::TargetNeuronType::Leaf;
        }

        return DistantNeuronRequest::TargetNeuronType::VirtualNode;
    }
};
