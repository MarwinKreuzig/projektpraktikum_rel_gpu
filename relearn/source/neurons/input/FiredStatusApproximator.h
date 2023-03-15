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

#include "Types.h"
#include "neurons/enums/FiredStatus.h"
#include "util/MPIRank.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"

#include <memory>
#include <unordered_map>
#include <span>
#include <vector>

class FiredStatusApproximator : public FiredStatusCommunicator {
public:
    /**
     * @brief Approximates the firing rate of distant neurons by a constant frequency
     * @param number_ranks The number of ranks
     * @param number_neurons The number of local neurons
     */
    FiredStatusApproximator(const size_t number_ranks)
        : FiredStatusCommunicator(number_ranks)
        , firing_rate_cache(number_ranks) {
        RelearnException::check(number_ranks > 0, "FiredStatusApproximator::FiredStatusApproximator: number_ranks is too small: {}", number_ranks);
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<FiredStatusCommunicator> clone() const override {
        return std::make_unique<FiredStatusApproximator>(get_number_ranks());
    }

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0
     */
    void init(const number_neurons_type number_neurons) override {
        FiredStatusCommunicator::init(number_neurons);

        accumulated_fired.resize(number_neurons, 0);
        latest_firing_rate.resize(number_neurons, 0.0);
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(const number_neurons_type creation_count) override {
        const auto old_size = get_number_local_neurons();

        FiredStatusCommunicator::create_neurons(creation_count);

        const auto new_size = old_size + creation_count;

        accumulated_fired.resize(new_size, 0);
        latest_firing_rate.resize(new_size, 0.0);
    }

    /**
     * @brief Registers the fired status of the local neurons that are not disabled.
     * @param step The current update step
     * @param fired_status The current fired status of the neurons
     * @exception Can throw a RelearnException
     */
    void set_local_fired_status(step_type step, std::span<const FiredStatus> fired_status) override;

    /**
     * @brief Exchanges the fired status with all MPI ranks
     * @param step The current update step
     * @exception Can throw a RelearnException
     */
    void exchange_fired_status(step_type step) override;

    /**
     * @brief Checks if the communicator contains the specified neuron of the rank,
     *      i.e., whether that neuron fired in the last update step.
     * @param rank The MPI rank that owns the neuron
     * @param neuron_id The neuron in question
     * @exception Throws a RelearnException if rank is not from [0, number_ranks) or the neuron_id is virtual
     */
    bool contains(MPIRank rank, NeuronID neuron_id) const override;

    /**
     * @brief Recalculate the cached firing rates for the distant neurons
     * @param step The current simulation step
     */
    void notify_of_plasticity_change(step_type step) override;

private:
    std::vector<size_t> accumulated_fired{};
    std::vector<double> latest_firing_rate{};
    std::vector<std::unordered_map<NeuronID, double>> firing_rate_cache{};

    step_type last_synced{ 0 };
};
