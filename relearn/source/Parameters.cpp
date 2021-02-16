/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Parameters.h"

#include "LogFiles.h"
#include "Random.h"

#include <iomanip>
#include <sstream>

void Parameters::print() const {
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
