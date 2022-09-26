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

#include "FiredStatusCommunicator.h"

#include "mpi/CommunicationMap.h"
#include "neurons/FiredStatus.h"
#include "neurons/UpdateStatus.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"

#include <algorithm>
#include <vector>

class NetworkGraph;

/**
 * This class communicates the fired status of the local neurons
 * via two separate CommunicationMap<FiredStatus>
 */
class FiredStatusCommunicationMap : public FiredStatusCommunicator {
public:
    /**
     * @brief Constructs a new object with the given number of ranks and local neurons (mainly used for preallocating memory)
     * @param number_ranks The number of MPI ranks
     * @param number_neurons The number of local neurons
     * @exception Throws a RelearnException if number_ranks <= 0
     */
    FiredStatusCommunicationMap(int number_ranks, size_t number_neurons)
        : FiredStatusCommunicator(number_ranks, number_neurons)
        , outgoing_ids(number_ranks, std::min(size_t(number_ranks), number_neurons))
        , incoming_ids(number_ranks, std::min(size_t(number_ranks), number_neurons)) {
        RelearnException::check(number_ranks > 0, "FiredStatusCommunicationMap::FiredStatusCommunicationMap: number_ranks is too small: {}", number_ranks);
    }

    /**
     * @brief Registers the fired status of the local neurons that are not disabled.
     *      Potentially uses the out-edges of the network graph
     * @param fired_status The current fired status of the neurons
     * @param disable_flags The current disable flags for the neurons
     * @param network_graph The network graph that is currently being used
     */
    void set_local_fired_status(const std::vector<FiredStatus>& fired_status, const std::vector<UpdateStatus>& disable_flags, const NetworkGraph& network_graph) override;

    /**
     * @brief Exchanges the fired status with all MPI ranks
     * @exception Can throw a RelearnException
     */
    void exchange_fired_status() override;

    /**
     * @brief Checks if the communicator contains the specified neuron of the rank,
     *      i.e., whether that neuron fired in the last update step.
     * @param rank The MPI rank that owns the neuron
     * @param neuron_id The neuron in question
     * @exception Throws a RelearnException if rank is not from [0, number_ranks) or the neuron_id is virtual
     */
    bool contains(int rank, NeuronID neuron_id) const override;

private:
    CommunicationMap<NeuronID> outgoing_ids;
    CommunicationMap<NeuronID> incoming_ids;
};
