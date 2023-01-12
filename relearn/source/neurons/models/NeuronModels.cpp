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
#include "neurons/Neurons.h"
#include "util/Random.h"
#include "util/Timers.h"

void NeuronModel::init(number_neurons_type number_neurons) {
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

void NeuronModel::update_electrical_activity(const step_type step, const NetworkGraph& network_graph_static, const NetworkGraph& network_graph_plastic, const std::span<const UpdateStatus> disable_flags) {
    input_calculator->update_input(step, network_graph_static, network_graph_plastic, fired, disable_flags);
    background_calculator->update_input(step, disable_flags);
    stimulus_calculator->update_stimulus(step, disable_flags);

    Timers::start(TimerRegion::CALC_ACTIVITY);
    update_activity(disable_flags);
    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);
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
