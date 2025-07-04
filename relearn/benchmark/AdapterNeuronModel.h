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

#include "enums/FiredStatus.h"
#include "util/NeuronID.h"

#include <range/v3/algorithm/fill.hpp>

#include <span>
#include <vector>

template <typename NeuronModelType>
class AdapterNeuronModel {
    NeuronModelType& model;

public:
    AdapterNeuronModel(NeuronModelType& neuron_model)
        : model(neuron_model) {
    }

    std::span<const double> get_background() {
        return model.get_background_activity();
    }

    std::span<const double> get_synaptic_input() {
        return model.get_synaptic_input();
    }

    std::span<const double> get_x() {
        return model.get_x();
    }

    void set_fired_status(const FiredStatus fs) {
        ranges::fill(model.fired, fs);
    }

    void update_activity_benchmark(const NeuronID id) {
        model.update_activity_benchmark(id);
    }

    void update_activity() {
        model.update_activity(0);
    }

    void update_activity_benchmark() {
        model.update_activity_benchmark();
    }
};
