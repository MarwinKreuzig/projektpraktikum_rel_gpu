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
#include "MPIWrapper.h"
#include "NetworkGraph.h"
#include "Random.h"
#include "Timers.h"

NeuronModels::NeuronModels(double k, double tau_C, double beta, unsigned int h, double background_activity, double background_activity_mean, double background_activity_stddev)
    : k(k)
    , tau_C(tau_C)
    , beta(beta)
    , h(h)
    , base_background_activity(background_activity)
    , background_activity_mean(background_activity_mean)
    , background_activity_stddev(background_activity_stddev) {
}

void NeuronModels::update_electrical_activity(const NetworkGraph& network_graph) {

    MapFiringNeuronIds firing_neuron_ids_outgoing = update_electrical_activity_prepare_sending_spikes(network_graph);
    std::vector<size_t> num_incoming_ids = update_electrical_activity_prepare_receiving_spikes(firing_neuron_ids_outgoing);

    MapFiringNeuronIds firing_neuron_ids_incoming = update_electrical_activity_exchange_neuron_ids(firing_neuron_ids_outgoing, num_incoming_ids);
    /**
	 * Now fired contains spikes only from my own neurons
	 * (spikes from local neurons)
	 *
	 * The incoming spikes of neurons from other ranks are in firing_neuron_ids_incoming
	 * (spikes from neurons from other ranks)
	 */

    update_electrical_activity_serial_initialize();

    update_electrical_activity_calculate_background();
    update_electrical_activity_calculate_input(network_graph, firing_neuron_ids_incoming);
    update_electrical_activity_update_activity();
}

void NeuronModels::update_electrical_activity_update_activity() {
    GlobalTimers::timers.start(TimerRegion::CALC_ACTIVITY);

    // For my neurons
#pragma omp parallel for default(none)
    for (auto i = 0; i < my_num_neurons; ++i) {
        update_activity(i);
    }

    GlobalTimers::timers.stop_and_add(TimerRegion::CALC_ACTIVITY);
}

void NeuronModels::update_electrical_activity_calculate_input(const NetworkGraph& network_graph, const MapFiringNeuronIds& firing_neuron_ids_incoming) {
    const auto my_rank = MPIWrapper::get_my_rank();

    GlobalTimers::timers.start(TimerRegion::CALC_SYNAPTIC_INPUT);
    // For my neurons

#pragma omp parallel for shared(my_rank, firing_neuron_ids_incoming) default(none)
    for (auto neuron_id = 0; neuron_id < my_num_neurons; ++neuron_id) {
        /**
		 * Determine synaptic input from neurons connected to me
		 */

        // Walk through in-edges of my neuron
        const NetworkGraph::Edges& in_edges = network_graph.get_in_edges(neuron_id);

        for (const auto& [key, edge_val] : in_edges) {
            const auto& [rank, src_neuron_id] = key;

            bool spike{ false };
            if (rank == my_rank) {
                spike = static_cast<bool>(fired[src_neuron_id]);
            } else {
                const auto it = firing_neuron_ids_incoming.find(rank);
                spike = (it != firing_neuron_ids_incoming.end()) && (it->second.find(src_neuron_id));
            }
            I_syn[neuron_id] += k * edge_val * static_cast<double>(spike);
        }
    }
    GlobalTimers::timers.stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);
}

void NeuronModels::update_electrical_activity_calculate_background() {
    GlobalTimers::timers.start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);

    // There might be background activity
    if (background_activity_stddev > 0.0) {
        for (size_t neuron_id = 0; neuron_id < my_num_neurons; ++neuron_id) {
            const double rnd = RandomHolder::get_random_normal_double(RandomHolderKey::NeuronModels, background_activity_mean, background_activity_stddev);
            const double input = base_background_activity + rnd;
            I_syn[neuron_id] = input;
        }
    } else {
        std::fill(I_syn.begin(), I_syn.end(), 0.0);
    }

    GlobalTimers::timers.stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
}

std::vector<size_t> NeuronModels::update_electrical_activity_prepare_receiving_spikes(const MapFiringNeuronIds& firing_neuron_ids_outgoing) {
    GlobalTimers::timers.start(TimerRegion::PREPARE_NUM_NEURON_IDS);

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
    GlobalTimers::timers.stop_and_add(TimerRegion::PREPARE_NUM_NEURON_IDS);

    GlobalTimers::timers.start(TimerRegion::ALL_TO_ALL);
    // Send and receive the number of firing neuron ids
    MPIWrapper::all_to_all(num_firing_neuron_ids_for_ranks, num_firing_neuron_ids_from_ranks, MPIWrapper::Scope::global);
    GlobalTimers::timers.stop_and_add(TimerRegion::ALL_TO_ALL);

    GlobalTimers::timers.start(TimerRegion::ALLOC_MEM_FOR_NEURON_IDS);
    // Now I know how many neuron ids I will get from every rank.
    // Allocate memory for all incoming neuron ids.
    for (auto rank = 0; rank < num_ranks; ++rank) {
        // Only create key-value pair in map for "rank" if necessary
        if (auto num_neuron_ids = num_firing_neuron_ids_from_ranks[rank]; 0 != num_neuron_ids) {
            num_firing_neuron_ids_incoming[rank] = num_neuron_ids;
        }
    }
    GlobalTimers::timers.stop_and_add(TimerRegion::ALLOC_MEM_FOR_NEURON_IDS);

    return num_firing_neuron_ids_incoming;
}

NeuronModels::MapFiringNeuronIds NeuronModels::update_electrical_activity_exchange_neuron_ids(const MapFiringNeuronIds& firing_neuron_ids_outgoing, const std::vector<size_t>& num_incoming_ids) {
    GlobalTimers::timers.start(TimerRegion::EXCHANGE_NEURON_IDS);

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

        MPIWrapper::async_receive(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

        ++mpi_requests_index;
    }

    // Send actual neuron ids
    for (const auto& it : firing_neuron_ids_outgoing) {
        auto rank = it.first;
        const auto* buffer = it.second.get_neuron_ids();
        const auto size_in_bytes = static_cast<int>(it.second.get_neuron_ids_size_in_bytes());

        MPIWrapper::async_send(buffer, size_in_bytes, rank, MPIWrapper::Scope::global, mpi_requests[mpi_requests_index]);

        ++mpi_requests_index;
    }

    // Wait for all sends and receives to complete
    MPIWrapper::wait_all_tokens(mpi_requests);

    GlobalTimers::timers.stop_and_add(TimerRegion::EXCHANGE_NEURON_IDS);

    return firing_neuron_ids_incoming;
}

NeuronModels::MapFiringNeuronIds NeuronModels::update_electrical_activity_prepare_sending_spikes(const NetworkGraph& network_graph) {
    const auto my_rank = MPIWrapper::get_my_rank();

    NeuronModels::MapFiringNeuronIds firing_neuron_ids_outgoing;

    /**
	* Check which of my neurons fired and determine which ranks need to know about it.
	* That is, they contain the neurons connecting the axons of my firing neurons.
	*/
    GlobalTimers::timers.start(TimerRegion::PREPARE_SENDING_SPIKES);
    // For my neurons
    for (size_t neuron_id = 0; neuron_id < my_num_neurons; ++neuron_id) {
        // My neuron fired
        if (static_cast<bool>(fired[neuron_id])) {
            const NetworkGraph::Edges& out_edges = network_graph.get_out_edges(neuron_id);

            // Find all target neurons which should receive the signal fired.
            // That is, neurons which connect axons from neuron "neuron_id"
            for (const auto& it_out_edge : out_edges) {
                //target_neuron_id = it_out_edge->first.second;
                const auto target_rank = it_out_edge.first.first;

                // Don't send firing neuron id to myself as I already have this info
                if (target_rank != my_rank) {
                    // Function expects to insert neuron ids in sorted order
                    // Append if it is not already in
                    firing_neuron_ids_outgoing[target_rank].append_if_not_found_sorted(neuron_id);
                }
            }
        } // My neuron fired
    } // For my neurons
    GlobalTimers::timers.stop_and_add(TimerRegion::PREPARE_SENDING_SPIKES);

    return firing_neuron_ids_outgoing;
}

std::vector<std::unique_ptr<NeuronModels>> NeuronModels::get_models() {
    std::vector<std::unique_ptr<NeuronModels>> res;
    res.push_back(NeuronModels::create<models::ModelA>());
    res.push_back(NeuronModels::create<models::IzhikevichModel>());
    res.push_back(NeuronModels::create<models::FitzHughNagumoModel>());
    res.push_back(NeuronModels::create<models::AEIFModel>());
    return res;
}

std::vector<ModelParameter> NeuronModels::get_parameter() {
    return {
        Parameter<double>{ "k", k, NeuronModels::min_k, NeuronModels::max_k },
        Parameter<double>{ "tau_C", tau_C, NeuronModels::min_tau_C, NeuronModels::max_tau_C },
        Parameter<double>{ "beta", beta, NeuronModels::min_beta, NeuronModels::max_beta },
        Parameter<unsigned int>{ "Number integration steps", h, NeuronModels::min_h, NeuronModels::max_h },
        Parameter<double>{ "Base background activity", base_background_activity, NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity },
        Parameter<double>{ "Background activity mean", background_activity_mean, NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean },
        Parameter<double>{ "Background activity standard deviation", background_activity_stddev, NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev },
    };
}

void NeuronModels::init(size_t num_neurons) {
    my_num_neurons = num_neurons;
    x.resize(num_neurons, 0.0);
    fired.resize(num_neurons, false);
    I_syn.resize(num_neurons, 0.0);
}
