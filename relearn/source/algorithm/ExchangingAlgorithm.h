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

#include "Algorithm.h"
#include "Types.h"
#include "mpi/CommunicationMap.h"
#include "mpi/MPIWrapper.h"
#include "util/Timers.h"

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

class NeuronsExtraInfo;

/**
 * @brief This class manages the exchange of requests and responses, and their distribution on all MPI ranks
 * @tparam RequestType The type of creation requests
 * @tparam ResponseType The type of creation responses
 */
template <typename RequestType, typename ResponseType>
class ExchangingAlgorithm : public Algorithm {
public:
    /**
     * @brief Updates the connectivity with the algorithm. Already updates the synaptic elements, i.e., the axons and dendrites (both excitatory and inhibitory).
     *      Does not update the network graph. Performs communication with MPI
     * @param number_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so, the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @exception Can throw a RelearnException
     * @return A tuple with the created synapses that must be committed to the network graph
     */
    [[nodiscard]] std::tuple<LocalSynapses, DistantInSynapses, DistantOutSynapses> Algorithm::update_connectivity(size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
        const std::unique_ptr<NeuronsExtraInfo>& extra_infos) {

        Timers::start(TimerRegion::CREATE_SYNAPSES);

        Timers::start(TimerRegion::FIND_TARGET_NEURONS);
        const auto& synapse_creation_requests_outgoing = find_target_neurons(number_neurons, disable_flags, extra_infos);
        Timers::stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

        Timers::start(TimerRegion::CREATE_SYNAPSES_EXCHANGE_REQUESTS);
        const auto& synapse_creation_requests_incoming = MPIWrapper::exchange_requests(synapse_creation_requests_outgoing);
        Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES_EXCHANGE_REQUESTS);

        Timers::start(TimerRegion::CREATE_SYNAPSES_PROCESS_REQUESTS);
        auto [responses_outgoing, synapses] = create_synapses_process_requests(number_neurons, synapse_creation_requests_incoming);
        auto& [local_synapses, distant_in_synapses] = synapses;
        Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES_PROCESS_REQUESTS);

        Timers::start(TimerRegion::CREATE_SYNAPSES_EXCHANGE_RESPONSES);
        const auto& responses_incoming = MPIWrapper::exchange_requests(responses_outgoing);
        Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES_EXCHANGE_RESPONSES);

        Timers::start(TimerRegion::CREATE_SYNAPSES_PROCESS_RESPONSES);
        auto out_synapses = create_synapses_process_responses(synapse_creation_requests_outgoing, responses_incoming);
        Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES_PROCESS_RESPONSES);

        Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES);

        return {
            std::move(local_synapses), std::move(distant_in_synapses), std::move(out_synapses)
        };
    }

protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron
     * @param number_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so, the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    [[nodiscard]] virtual CommunicationMap<RequestType> find_target_neurons(size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
        const std::unique_ptr<NeuronsExtraInfo>& extra_infos) = 0;

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param number_neurons The number of local neurons
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] virtual std::pair<CommunicationMap<ResponseType>, std::pair<LocalSynapses, DistantInSynapses>>
        create_synapses_process_requests(size_t number_neurons, const CommunicationMap<RequestType>& creation_requests) = 0;

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] virtual DistantOutSynapses 
        create_synapses_process_responses(const CommunicationMap<RequestType>& creation_requests, const CommunicationMap<ResponseType>& creation_responses) = 0;
};
