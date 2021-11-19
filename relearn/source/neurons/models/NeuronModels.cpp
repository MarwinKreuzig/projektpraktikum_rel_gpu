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

#include "../../Config.h"
#include "../../mpi/MPIWrapper.h"
#include "../../util/Random.h"
#include "../../util/Timers.h"
#include "../NetworkGraph.h"
#include "../Neurons.h"

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

    MapFiringNeuronIds firing_neuron_ids_outgoing = update_electrical_activity_prepare_sending_spikes(network_graph, disable_flags);
    std::vector<size_t> num_incoming_ids = update_electrical_activity_prepare_receiving_spikes(firing_neuron_ids_outgoing);

    MapFiringNeuronIds firing_neuron_ids_incoming = update_electrical_activity_exchange_neuron_ids(firing_neuron_ids_outgoing, num_incoming_ids);

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

void NeuronModel::update_electrical_activity_calculate_input(const NetworkGraph& network_graph, const MapFiringNeuronIds& firing_neuron_ids_incoming, const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_INPUT);
    // For my neurons

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
            const auto spike = fired[src_neuron_id.id];
            if (spike != 0) {
                I += k * edge_val;
            }
        }

        // Walk through the distant in-edges of my neuron
        const NetworkGraph::DistantEdges& in_edges = network_graph.get_distant_in_edges(id);

        for (const auto& [key, edge_val] : in_edges) {
            const auto& rank = key.get_rank();
            const auto& src_neuron_id = key.get_neuron_id();

            const auto it = firing_neuron_ids_incoming.find(rank);
            const auto found = (it != firing_neuron_ids_incoming.end()) && (it->second.find(src_neuron_id));

            if (found) {
                I += k * edge_val;
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

std::vector<size_t> NeuronModel::update_electrical_activity_prepare_receiving_spikes(const MapFiringNeuronIds& firing_neuron_ids_outgoing) {
    Timers::start(TimerRegion::PREPARE_NUM_NEURON_IDS);

    const auto num_ranks = MPIWrapper::get_num_ranks();
    std::vector<size_t> num_firing_neuron_ids_incoming(num_ranks, 0);

    /**
     * Send to every rank the number of firing neuron ids it should prepare for from me.
     * Likewise, receive the number of firing neuron ids that I should prepare for from every rank.
     */
    std::vector<size_t> num_firing_neuron_ids_for_ranks(num_ranks, 0);
    std::vector<size_t> num_firing_neuron_ids_from_ranks(num_ranks, Constants::uninitialized);

    // Fill vector with my number of firing neuron ids for every rank (excluding me)
    for (const auto& [rank, neuron_ids] : firing_neuron_ids_outgoing) {
        const auto num_neuron_ids = neuron_ids.size();
        num_firing_neuron_ids_for_ranks[rank] = num_neuron_ids;
    }
    Timers::stop_and_add(TimerRegion::PREPARE_NUM_NEURON_IDS);

    Timers::start(TimerRegion::ALL_TO_ALL);
    // Send and receive the number of firing neuron ids
    MPIWrapper::all_to_all(num_firing_neuron_ids_for_ranks, num_firing_neuron_ids_from_ranks);
    Timers::stop_and_add(TimerRegion::ALL_TO_ALL);

    Timers::start(TimerRegion::ALLOC_MEM_FOR_NEURON_IDS);
    // Now I know how many neuron ids I will get from every rank.
    // Allocate memory for all incoming neuron ids.
    for (auto rank = 0; rank < num_ranks; ++rank) {
        // Only create key-value pair in map for "rank" if necessary
        if (auto num_neuron_ids = num_firing_neuron_ids_from_ranks[rank]; 0 != num_neuron_ids) {
            num_firing_neuron_ids_incoming[rank] = num_neuron_ids;
        }
    }
    Timers::stop_and_add(TimerRegion::ALLOC_MEM_FOR_NEURON_IDS);

    return num_firing_neuron_ids_incoming;
}

NeuronModel::MapFiringNeuronIds NeuronModel::update_electrical_activity_exchange_neuron_ids(const MapFiringNeuronIds& firing_neuron_ids_outgoing, const std::vector<size_t>& num_incoming_ids) {
    Timers::start(TimerRegion::EXCHANGE_NEURON_IDS);

    /**
     * Send and receive actual neuron ids
     */

    MapFiringNeuronIds firing_neuron_ids_incoming;
    for (auto rank = 0; rank < MPIWrapper::get_num_ranks(); rank++) {
        const auto num_incoming_ids_from_frank = num_incoming_ids[rank];
        if (num_incoming_ids_from_frank > 0) {
            firing_neuron_ids_incoming[rank].resize(num_incoming_ids_from_frank);
        }
    }

    std::vector<MPIWrapper::AsyncToken>
        mpi_requests(firing_neuron_ids_outgoing.size() + firing_neuron_ids_incoming.size());

    auto mpi_requests_index = 0;

    // Receive actual neuron ids
    for (auto& it : firing_neuron_ids_incoming) {
        auto rank = it.first;
        auto* buffer = it.second.get_neuron_ids();
        const auto size_in_bytes = static_cast<int>(it.second.get_neuron_ids_size_in_bytes());

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

        ++mpi_requests_index;
    }

    // Send actual neuron ids
    for (const auto& it : firing_neuron_ids_outgoing) {
        auto rank = it.first;
        const auto* buffer = it.second.get_neuron_ids();
        const auto size_in_bytes = static_cast<int>(it.second.get_neuron_ids_size_in_bytes());

        MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

        ++mpi_requests_index;
    }

    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);

    Timers::stop_and_add(TimerRegion::EXCHANGE_NEURON_IDS);

    return firing_neuron_ids_incoming;
}

NeuronModel::MapFiringNeuronIds NeuronModel::update_electrical_activity_prepare_sending_spikes(const NetworkGraph& network_graph, const std::vector<UpdateStatus>& disable_flags) {
    // If there is no other rank, then we can just skip
    if (const auto number_mpi_ranks = MPIWrapper::get_num_ranks(); number_mpi_ranks == 1) {
        return {};
    }

    /**
     * Check which of my neurons fired and determine which ranks need to know about it.
     * That is, they contain the neurons connecting the axons of my firing neurons.
     */

    Timers::start(TimerRegion::PREPARE_SENDING_SPIKES);

    NeuronModel::MapFiringNeuronIds firing_neuron_ids_outgoing{};

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
            firing_neuron_ids_outgoing[target_rank].append(id);
        }
    } // For my neurons
    Timers::stop_and_add(TimerRegion::PREPARE_SENDING_SPIKES);

    return firing_neuron_ids_outgoing;
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
