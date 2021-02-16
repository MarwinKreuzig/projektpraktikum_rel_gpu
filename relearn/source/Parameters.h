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

#include "LogFiles.h"
#include "Random.h"

#include <cstdint>
#include <iomanip>
#include <sstream>

class Parameters {
public:
    size_t total_num_neurons; // Number of neurons

    // Connectivity
    double accept_criterion; // Barnes-Hut acceptance criterion
    double sigma; // Probability parameter in MSP (dispersion). The higher sigma the more likely to form synapses with remote neurons
    bool naive_method; // Consider all neurons as target neurons for synapse creation, regardless of whether dendrites are available or not
    size_t max_num_pending_vacant_axons; // Maximum number of vacant axons which are considered at the same time for finding a target neuron

    // Overload << operator for proper output
    void print() {
        std::stringstream ss;

        ss << "** PARAMETERS **\n\n";
        ss << std::left << std::setw(column_width) << "num_neurons"
           << " : " << total_num_neurons << "\n";
        ss << std::left << std::setw(column_width) << "accept_criterion (BH)"
           << " : " << accept_criterion << "\n";
        ss << std::left << std::setw(column_width) << "sigma"
           << " : " << sigma << "\n";
        ss << std::left << std::setw(column_width) << "naive_method (BH)"
           << " : " << naive_method << "\n";
        ss << std::left << std::setw(column_width) << "max_num_pending_vacant_axons"
           << " : " << max_num_pending_vacant_axons << "\n";
        ss << std::left << std::setw(column_width) << "seed_octree"
           << " : " << randomNumberSeeds::octree << "\n";
        ss << std::left << std::setw(column_width) << "seed_partition"
           << " : "
           << "Local MPI rank"
           << "\n";

        LogFiles::write_to_file(LogFiles::EventType::Cout, ss.str(), true);
    }

private:
    // Width of column containing parameter names
    static const int column_width = 28;
};
