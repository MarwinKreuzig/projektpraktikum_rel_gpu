/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Algorithm.h"

#include "../util/Timers.h"

std::tuple<LocalSynapses, DistantInSynapses, DistantOutSynapses> Algorithm::update_connectivity(size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
    const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

    MapSynapseCreationRequests synapse_creation_requests_outgoing = find_target_neurons(number_neurons, disable_flags, extra_infos);

    //std::stringstream ss{};
    //ss << "I'm rank " << MPIWrapper::get_my_rank() << " and I have the following requests:\n";
    //for (const auto& [rank, requests] : synapse_creation_requests_outgoing) {
    //    ss << rank << ": " << requests.size() << '\n';
    //}

    //std::cout << ss.str();
    //fflush(stdout);

    Timers::start(TimerRegion::CREATE_SYNAPSES);

    auto synapse_creation_requests_incoming = SynapseCreationRequests::exchange_requests(synapse_creation_requests_outgoing);
    auto in_synapses = create_synapses_process_requests(number_neurons, synapse_creation_requests_incoming);

    const auto num_synapses_created = in_synapses.size();

    SynapseCreationRequests::exchange_responses(synapse_creation_requests_incoming, synapse_creation_requests_outgoing);
    auto out_synapses = create_synapses_process_responses(synapse_creation_requests_outgoing);

    Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES);

    return {
        {}, std::move(in_synapses), std::move(out_synapses)
    };
}


DistantInSynapses Algorithm::create_synapses_process_requests(size_t number_neurons, MapSynapseCreationRequests& synapse_creation_requests_incoming) {
    if (synapse_creation_requests_incoming.empty()) {
        return {};
    }

    const auto my_rank = MPIWrapper::get_my_rank();
    DistantInSynapses synapses{};

    for (auto& [source_rank, requests] : synapse_creation_requests_incoming) {
        const auto num_requests = requests.size();

        // All requests of a rank
        for (auto request_index = 0; request_index < num_requests; request_index++) {
            const auto& [source_neuron_id, target_neuron_id, dendrite_type_needed] = requests.get_request(request_index);
            RelearnException::check(target_neuron_id < number_neurons, "Neurons::create_synapses_process_requests: Target_neuron_id exceeds my neurons");

            int weight = 0;
            unsigned int number_free_elements = 0;

            // DendriteType::INHIBITORY dendrite requested
            if (SignalType::INHIBITORY == dendrite_type_needed) {
                number_free_elements = inhibitory_dendrites->get_free_elements(target_neuron_id);
                weight = -1;
            }
            // DendriteType::EXCITATORY dendrite requested
            else {
                number_free_elements = excitatory_dendrites->get_free_elements(target_neuron_id);
                weight = +1;
            }

            if (number_free_elements > 0) {
                // Increment num of connected dendrites
                if (SignalType::INHIBITORY == dendrite_type_needed) {
                    inhibitory_dendrites->update_connected_elements(target_neuron_id, 1);
                } else {
                    excitatory_dendrites->update_connected_elements(target_neuron_id, 1);
                }

                synapses.emplace_back(target_neuron_id, RankNeuronId{ source_rank, source_neuron_id }, weight);

                // Set response to "connected" (success)
                requests.set_response(request_index, 1);
            } else {
                // Other axons were faster and came first
                // Set response to "not connected" (not success)
                requests.set_response(request_index, 0);
            }
        } // All requests of a rank
    } // Increasing order of ranks that sent requests

    return synapses;
}

DistantOutSynapses Algorithm::create_synapses_process_responses(const MapSynapseCreationRequests& synapse_creation_requests_outgoing) {
    const auto my_rank = MPIWrapper::get_my_rank();
    DistantOutSynapses synapses{};

    /**
	 * Register which axons could be connected
	 *
	 * NOTE: Do not create synapses in the network for my own responses as the corresponding synapses, if possible,
	 * would have been created before sending the response to myself (see above).
	 */
    for (const auto& [target_rank, requests] : synapse_creation_requests_outgoing) {
        const auto num_requests = requests.size();

        // All responses from a rank
        for (auto request_index = 0; request_index < num_requests; request_index++) {
            const auto connected = requests.get_response(request_index);

            if (connected == 0) {
                continue;
            }

            const auto& [source_neuron_id, target_neuron_id, dendrite_type_needed] = requests.get_request(request_index);

            // Increment num of connected axons
            axons->update_connected_elements(source_neuron_id, 1);

            // I have already created the synapse in the network
            // if the response comes from myself
            if (target_rank != my_rank) {
                // Update network
                const auto weight = (SignalType::INHIBITORY == dendrite_type_needed) ? -1 : +1;
                synapses.emplace_back(RankNeuronId{ target_rank, target_neuron_id }, source_neuron_id, weight);
            }
        } // All responses from a rank
    } // All outgoing requests

    return synapses;
}
