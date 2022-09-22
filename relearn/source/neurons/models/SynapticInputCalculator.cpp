/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticInputCalculator.h"

#include "mpi/MPIWrapper.h"
#include "neurons/models/FiredStatusCommunicationMap.h"

void SynapticInputCalculator::init(const size_t number_neurons) {
    number_local_neurons = number_neurons;

    synaptic_input.resize(number_neurons, 0.0);
    background_activity.resize(number_neurons, 0.0);

    fired_status_comm = std::make_unique<FiredStatusCommunicationMap>(MPIWrapper::get_num_ranks(), number_neurons);
}

void SynapticInputCalculator::create_neurons(const size_t creation_count) {
    const auto current_size = number_local_neurons;
    const auto new_size = current_size + creation_count;
    number_local_neurons = new_size;

    synaptic_input.resize(new_size, 0.0);
    background_activity.resize(new_size, 0.0);

    fired_status_comm = std::make_unique<FiredStatusCommunicationMap>(MPIWrapper::get_num_ranks(), new_size);
}
