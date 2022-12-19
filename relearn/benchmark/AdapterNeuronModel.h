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

#include "neurons/FiredStatus.h"

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
        for (auto& fired_status : model.fired) {
            fired_status = fs;
        }
    }

    void update_activity(NeuronID id) {
        model.update_activity(id);
    }

    void update_activity_benchmark(NeuronID id) {
        model.update_activity_benchmark(id);
    }
};
