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

#include "neurons/enums/ElementType.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/SignalType.h"
#include "neurons/helper/DistantNeuronRequests.h"

#include <random>
#include <vector>

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

    static std::vector<FiredStatus> get_fired_status(size_t number_neurons, std::mt19937& mt) {
        const auto number_disabled = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);
        return get_fired_status(number_neurons, number_disabled, mt);
    }

    static std::vector<FiredStatus> get_fired_status(size_t number_neurons, size_t number_inactive, std::mt19937& mt) {
        std::vector<FiredStatus> status(number_inactive, FiredStatus::Inactive);
        status.resize(number_neurons, FiredStatus::Fired);

        RandomAdapter::shuffle(status.begin(), status.end(), mt);

        return status;
    }

    static std::vector<UpdateStatus> get_update_status(size_t number_neurons, std::mt19937& mt) {
        const auto number_disabled = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);
        return get_update_status(number_neurons, number_disabled, mt);
    }
    

    static std::vector<UpdateStatus> get_update_status(size_t number_neurons, size_t number_disabled, std::mt19937& mt) {
        std::vector<UpdateStatus> status(number_disabled, UpdateStatus::Disabled);
        status.resize(number_neurons, UpdateStatus::Enabled);

        RandomAdapter::shuffle(status.begin(), status.end(), mt);

        return status;
    }
};
