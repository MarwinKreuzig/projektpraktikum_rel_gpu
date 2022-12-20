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

#include "Types.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"

#include <vector>

class NetworkGraph;

/**
 * This class provides a virtual interface for exchanging the NeuronID of those that fired in the simulation step.
 */
class FiredStatusCommunicator {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new object with the given number of ranks and local neurons (mainly used for preallocating memory)
     * @param number_ranks The number of MPI ranks
     * @param number_neurons The number of local neurons
     * @exception Throws a RelearnException if number_ranks <= 0
     */
    FiredStatusCommunicator(const size_t number_ranks, const number_neurons_type number_local_neurons)
        : number_ranks(number_ranks)
        , number_local_neurons(number_local_neurons) {
        RelearnException::check(number_ranks > 0, "FiredStatusCommunicator::FiredStatusCommunicator: number_ranks is too small: {}", number_ranks);
    }

    /**
     * @brief Registers the fired status of the local neurons that are not disabled.
     *      Potentially uses the out-edges of the network graph
     * @param fired_status The current fired status of the neurons
     * @param disable_flags The current disable flags for the neurons
     * @param network_graph The network graph that is currently being used
     * @exception Can throw a RelearnException
     */
    virtual void set_local_fired_status(std::span<const FiredStatus> fired_status, std::span<const UpdateStatus> disable_flags, const NetworkGraph& network_graph_static, const NetworkGraph& network_graph_plastic) = 0;

    /**
     * @brief Exchanges the fired status with all MPI ranks
     * @exception Can throw a RelearnException
     */
    virtual void exchange_fired_status() = 0;

    /**
     * @brief Checks if the communicator contains the specified neuron of the rank,
     *      i.e., whether that neuron fired in the last update step.
     * @param rank The MPI rank that owns the neuron
     * @param neuron_id The neuron in question
     * @exception Can throw a RelearnException
     */
    [[nodiscard]] virtual bool contains(MPIRank rank, NeuronID neuron_id) const = 0;

    /**
     * @brief Returns the number of MPI ranks
     * @return The number of MPI ranks
     */
    [[nodiscard]] size_t get_number_ranks() const noexcept {
        return number_ranks;
    }

    /**
     * @brief Returns the number of local neurons
     * @return The number of local neurons
     */
    [[nodiscard]] number_neurons_type get_number_local_neurons() const noexcept {
        return number_local_neurons;
    }

    virtual ~FiredStatusCommunicator() = default;

private:
    size_t number_ranks{ 0 };
    number_neurons_type number_local_neurons{ 0 };
};
