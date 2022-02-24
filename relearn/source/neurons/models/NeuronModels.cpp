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

NeuronModel::NeuronModel(const double k, const double tau_C, const double beta, const unsigned int h, const double base_background_activity, const double background_activity_mean, const double background_activity_stddev)
    : k(k)
    , tau_C(tau_C)
    , beta(beta)
    , h(h)
    , base_background_activity(base_background_activity)
    , background_activity_mean(background_activity_mean)
    , background_activity_stddev(background_activity_stddev) {
}

void NeuronModel::update_electrical_activity(const NetworkGraph& network_graph, const std::vector<UpdateStatus>& disable_flags) {
    const auto& firing_neuron_ids_outgoing = update_electrical_activity_prepare_sending_spikes(network_graph, disable_flags);

    Timers::start(TimerRegion::EXCHANGE_NEURON_IDS);
    const auto& firing_neuron_ids_incoming = MPIWrapper::exchange_requests(firing_neuron_ids_outgoing);
    Timers::stop_and_add(TimerRegion::EXCHANGE_NEURON_IDS);

    /**
     * Now fired contains spikes only from my own neurons
     * (spikes from local neurons)
     *
     * The incoming spikes of neurons from other ranks are in firing_neuron_ids_incoming
     * (spikes from neurons from other ranks)
     */

    update_electrical_activity_serial_initialize(disable_flags);

    update_electrical_activity_calculate_background(disable_flags);
    update_electrical_activity_calculate_input(network_graph, firing_neuron_ids_incoming, disable_flags);
    update_electrical_activity_update_activity(disable_flags);
}

void NeuronModel::update_electrical_activity_update_activity(const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_ACTIVITY);

    // For my neurons
#pragma omp parallel for shared(disable_flags) default(none)
    for (auto neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::DISABLED) {
            continue;
        }

        update_activity(NeuronID{ neuron_id });
    }

    Timers::stop_and_add(TimerRegion::CALC_ACTIVITY);
}

void NeuronModel::update_electrical_activity_calculate_input(const NetworkGraph& network_graph, const CommunicationMap<NeuronID>& firing_neuron_ids_incoming, const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_INPUT);

#pragma omp parallel for shared(firing_neuron_ids_incoming, network_graph, disable_flags) default(none)
    for (auto neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::DISABLED) {
            continue;
        }

        NeuronID id{ neuron_id };
        /**
         * Determine synaptic input from neurons connected to me
         */

        // Walk through the local in-edges of my neuron
        const NetworkGraph::LocalEdges& local_in_edges = network_graph.get_local_in_edges(id);

        auto I = I_syn[neuron_id];
        for (const auto& [src_neuron_id, edge_val] : local_in_edges) {
            const auto spike = fired[src_neuron_id.get_local_id()];
            if (spike != 0) {
                I += k * edge_val;
            }
        }

        // Walk through the distant in-edges of my neuron
        const NetworkGraph::DistantEdges& in_edges = network_graph.get_distant_in_edges(id);

        for (const auto& [key, edge_val] : in_edges) {
            const auto& rank = key.get_rank();
            if (!firing_neuron_ids_incoming.contains(rank)) {
                continue;
            }

            const auto& initiator_neuron_id = key.get_neuron_id();

            const auto& firing_ids = firing_neuron_ids_incoming.get_requests(rank);
            const auto contains_id = std::binary_search(firing_ids.begin(), firing_ids.end(), initiator_neuron_id);

            if (contains_id) {
                I_syn[neuron_id] += k * edge_val;
            }
        }

        I_syn[neuron_id] = I;
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);
}

void NeuronModel::update_electrical_activity_calculate_background(const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);

    // There might be background activity
    if (background_activity_stddev > 0.0) {
        for (size_t neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
            if (disable_flags[neuron_id] == UpdateStatus::DISABLED) {
                continue;
            }

            const double rnd = RandomHolder::get_random_normal_double(RandomHolderKey::NeuronModel, background_activity_mean, background_activity_stddev);
            const double input = base_background_activity + rnd;
            I_syn[neuron_id] = input;
        }
    } else {
        std::fill(I_syn.begin(), I_syn.end(), base_background_activity);
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
}

CommunicationMap<NeuronID> NeuronModel::update_electrical_activity_prepare_sending_spikes(const NetworkGraph& network_graph, const std::vector<UpdateStatus>& disable_flags) {
    const auto mpi_ranks = MPIWrapper::get_num_ranks();

    CommunicationMap<NeuronID> spiking_ids(mpi_ranks);

    // If there is no other rank, then we can just skip
    if (mpi_ranks == 1) {
        return spiking_ids;
    }

    /**
     * Check which of my neurons fired and determine which ranks need to know about it.
     * That is, they contain the neurons connecting the axons of my firing neurons.
     */

    Timers::start(TimerRegion::PREPARE_SENDING_SPIKES);

    // For my neurons
    for (size_t neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::DISABLED) {
            continue;
        }

        if (fired[neuron_id] == 0) {
            continue;
        }

        const auto id = NeuronID{ neuron_id };

        // Don't send firing neuron id to myself as I already have this info
        const NetworkGraph::DistantEdges& distant_out_edges = network_graph.get_distant_out_edges(id);

        // Find all target neurons which should receive the signal fired.
        // That is, neurons which connect axons from neuron "neuron_id"
        for (const auto& [edge_key, _] : distant_out_edges) {
            const auto target_rank = edge_key.get_rank();

            // Function expects to insert neuron ids in sorted order
            // Append if it is not already in
            spiking_ids.append(target_rank, id);
        }
    } // For my neurons
    Timers::stop_and_add(TimerRegion::PREPARE_SENDING_SPIKES);

    return spiking_ids;
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
        Parameter<double>{ "tau_C", tau_C, NeuronModel::min_tau_C, NeuronModel::max_tau_C },
        Parameter<double>{ "beta", beta, NeuronModel::min_beta, NeuronModel::max_beta },
        Parameter<unsigned int>{ "Number integration steps", h, NeuronModel::min_h, NeuronModel::max_h },
        Parameter<double>{ "Base background activity", base_background_activity, NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity },
        Parameter<double>{ "Background activity mean", background_activity_mean, NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean },
        Parameter<double>{ "Background activity standard deviation", background_activity_stddev, NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev },
    };
}

void NeuronModel::init(size_t number_neurons) {
    number_local_neurons = number_neurons;
    x.resize(number_neurons, 0.0);
    fired.resize(number_neurons, 0);
    I_syn.resize(number_neurons, 0.0);
}

void NeuronModel::create_neurons(size_t creation_count) {
    const auto current_size = number_local_neurons;
    const auto new_size = current_size + creation_count;
    number_local_neurons = new_size;

    x.resize(new_size, 0.0);
    fired.resize(new_size, 0);
    I_syn.resize(new_size, 0.0);
}
