/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include <cstddef>

class Parameters {
public:
    size_t total_num_neurons; // Number of neurons

    // Connectivity
    double accept_criterion; // Barnes-Hut acceptance criterion
    double sigma; // Probability parameter in MSP (dispersion). The higher sigma the more likely to form synapses with remote neurons
    bool naive_method; // Consider all neurons as target neurons for synapse creation, regardless of whether dendrites are available or not
    size_t max_num_pending_vacant_axons; // Maximum number of vacant axons which are considered at the same time for finding a target neuron

    // Overload << operator for proper output
    void print() const;

private:
    // Width of column containing parameter names
    static const int column_width = 28;
};
