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
#include "FiredStatusCommunicationMap.h"
#include "mpi/MPIWrapper.h"
#include "neurons/NetworkGraph.h"
#include "neurons/Neurons.h"
#include "util/Random.h"
#include "util/Timers.h"

NeuronModel::NeuronModel(const double k, const unsigned int h, const double base_background_activity, const double background_activity_mean, const double background_activity_stddev)
    : k(k)
    , h(h)
    , base_background_activity(base_background_activity)
    , background_activity_mean(background_activity_mean)
    , background_activity_stddev(background_activity_stddev) {
}

void NeuronModel::init(size_t number_neurons) {
    number_local_neurons = number_neurons;

    x.resize(number_neurons, 0.0);
    fired.resize(number_neurons, FiredStatus::Inactive);
    fired_recorder.resize(number_neurons, 0);
    synaptic_input.resize(number_neurons, 0.0);
    background_activity.resize(number_neurons, 0.0);

    fired_status_comm = std::make_unique<FiredStatusCommunicationMap>(MPIWrapper::get_num_ranks(), number_neurons);
}

void NeuronModel::create_neurons(size_t creation_count) {
    const auto current_size = number_local_neurons;
    const auto new_size = current_size + creation_count;
    number_local_neurons = new_size;

    x.resize(new_size, 0.0);
    fired.resize(new_size, FiredStatus::Inactive);
    fired_recorder.resize(new_size, 0);
    synaptic_input.resize(new_size, 0.0);
    background_activity.resize(new_size, 0.0);

    fired_status_comm = std::make_unique<FiredStatusCommunicationMap>(MPIWrapper::get_num_ranks(), new_size);
}

void NeuronModel::update_electrical_activity(const NetworkGraph& network_graph, const std::vector<UpdateStatus>& disable_flags) {

    fired_status_comm->set_local_fired_status(fired, disable_flags, network_graph);
    fired_status_comm->exchange_fired_status();

    /**
     * Now fired contains spikes only from my own neurons
     * (spikes from local neurons)
     *
     * The incoming spikes of neurons from other ranks are in firing_neuron_ids_incoming
     * (spikes from neurons from other ranks)
     */

    update_electrical_activity_serial_initialize(disable_flags);

    update_electrical_activity_calculate_background(disable_flags);
    update_electrical_activity_calculate_input(network_graph, disable_flags);
    update_electrical_activity_update_activity(disable_flags);
}

void NeuronModel::update_electrical_activity_update_activity(const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_ACTIVITY);

    // For my neurons
#pragma omp parallel for shared(disable_flags) default(none)
    for (auto neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        NeuronID converted_id{ neuron_id };
        update_activity(converted_id);
    }

    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);
}

void NeuronModel::update_electrical_activity_calculate_input(const NetworkGraph& network_graph, const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_INPUT);

#pragma omp parallel for shared(network_graph, disable_flags, std::ranges::binary_search) default(none)
    for (auto neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        NeuronID id{ neuron_id };
        /**
         * Determine synaptic input from neurons connected to me
         */

        // Walk through the local in-edges of my neuron
        const NetworkGraph::LocalEdges& local_in_edges = network_graph.get_local_in_edges(id);

        auto total_input = 0.0;
        for (const auto& [src_neuron_id, edge_val] : local_in_edges) {
            const auto spike = fired[src_neuron_id.get_neuron_id()];
            if (spike == FiredStatus::Fired) {
                total_input += k * edge_val;
            }
        }

        // Walk through the distant in-edges of my neuron
        const NetworkGraph::DistantEdges& in_edges = network_graph.get_distant_in_edges(id);

        for (const auto& [key, edge_val] : in_edges) {
            const auto& rank = key.get_rank();
            const auto& initiator_neuron_id = key.get_neuron_id();

            const auto contains_id = fired_status_comm->contains(rank, initiator_neuron_id);
            if (contains_id) {
                total_input += k * edge_val;
            }
        }

        synaptic_input[neuron_id] = total_input;
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);
}

void NeuronModel::update_electrical_activity_calculate_background(const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);

    // There might be background activity
    if (background_activity_stddev > 0.0) {
        for (size_t neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
            if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
                continue;
            }

            const double rnd = RandomHolder::get_random_normal_double(RandomHolderKey::NeuronModel, background_activity_mean, background_activity_stddev);
            const double input = base_background_activity + rnd;
            background_activity[neuron_id] = input;
        }
    } else {
        std::ranges::fill(background_activity, base_background_activity);
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
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
    return {
        Parameter<double>{ "k", k, NeuronModel::min_k, NeuronModel::max_k },
        Parameter<unsigned int>{ "Number integration steps", h, NeuronModel::min_h, NeuronModel::max_h },
        Parameter<double>{ "Base background activity", base_background_activity, NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity },
        Parameter<double>{ "Background activity mean", background_activity_mean, NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean },
        Parameter<double>{ "Background activity standard deviation", background_activity_stddev, NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev },
    };
}
