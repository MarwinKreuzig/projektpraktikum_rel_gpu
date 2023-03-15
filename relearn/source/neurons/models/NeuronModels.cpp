/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronModels.h"

#include "Config.h"
#include "mpi/MPIWrapper.h"
#include "neurons/NetworkGraph.h"
#include "util/Random.h"
#include "util/Timers.h"

void NeuronModel::init(number_neurons_type number_neurons) {
    RelearnException::check(number_local_neurons == 0, "NeuronModel::init: Was already initialized");
    RelearnException::check(number_neurons > 0, "NeuronModel::init: Must initialize with more than 0 neurons");

    number_local_neurons = number_neurons;

    x.resize(number_neurons, 0.0);
    fired.resize(number_neurons, FiredStatus::Inactive);
    for (auto i = 0; i < number_fire_recorders; i++) {
        fired_recorder[i].resize(number_local_neurons, 0U);
    }

    input_calculator->init(number_neurons);
    background_calculator->init(number_neurons);
    stimulus_calculator->init(number_neurons);
}

void NeuronModel::create_neurons(number_neurons_type creation_count) {
    RelearnException::check(number_local_neurons > 0, "NeuronModel::create_neurons: Was not initialized");
    RelearnException::check(creation_count > 0, "NeuronModel::create_neurons: Must create more than 0 neurons");

    const auto current_size = number_local_neurons;
    const auto new_size = current_size + creation_count;
    number_local_neurons = new_size;

    x.resize(new_size, 0.0);
    fired.resize(new_size, FiredStatus::Inactive);
    for (auto i = 0; i < number_fire_recorders; i++) {
        fired_recorder[i].resize(new_size, 0U);
    }

    input_calculator->create_neurons(creation_count);
    background_calculator->create_neurons(creation_count);
    stimulus_calculator->create_neurons(creation_count);
}

void NeuronModel::update_electrical_activity(const step_type step) {
    input_calculator->update_input(step, fired);
    background_calculator->update_input(step);
    stimulus_calculator->update_stimulus(step);

    Timers::start(TimerRegion::CALC_ACTIVITY);
    update_activity();
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);
}

void NeuronModel::notify_of_plasticity_change(const step_type step) {
    input_calculator->notify_of_plasticity_change(step);
}

std::vector<std::unique_ptr<NeuronModel>> NeuronModel::get_models() {
    std::vector<std::unique_ptr<NeuronModel>> res;
    res.push_back(NeuronModel::create<models::PoissonModel>());
    res.push_back(NeuronModel::create<models::IzhikevichModel>());
    res.push_back(NeuronModel::create<models::FitzHughNagumoModel>());
    res.push_back(NeuronModel::create<models::AEIFModel>());
    return res;
}

std::vector<ModelParameter> NeuronModel::get_parameter() {
    auto parameters = input_calculator->get_parameter();
    auto other_parameters = background_calculator->get_parameter();

    parameters.insert(parameters.end(), other_parameters.begin(), other_parameters.end());
    parameters.emplace_back(Parameter<unsigned int>{ "Number integration steps", h, NeuronModel::min_h, NeuronModel::max_h });

    return parameters;
}
