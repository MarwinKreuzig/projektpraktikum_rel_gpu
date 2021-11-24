/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Partition.h"

#include "../io/LogFiles.h"

#include <sstream>

Partition::Partition(const size_t num_ranks, const size_t my_rank)
    : number_local_neurons(0)
    , total_number_neurons(0) {
    RelearnException::check(num_ranks > 0, "Partition::Partition: Number of MPI ranks must be a positive number: {}", num_ranks);
    RelearnException::check(num_ranks > my_rank, "Partition::Partition: My rank must be smaller than number of ranks: {} vs {}", num_ranks, my_rank);

    /**
	 * Total number of local_subdomains is smallest power of 8 that is >= num_ranks.
	 * We choose power of 8 as every domain subdivision creates 8 local_subdomains (in 3d).
	 */
    const double smallest_exponent = ceil(log(num_ranks) / log(8.0));
    level_of_subdomain_trees = static_cast<size_t>(smallest_exponent);
    total_number_subdomains = 1ULL << (3 * level_of_subdomain_trees); // 8^level_of_subdomain_trees

    // Every rank should get at least one subdomain
    RelearnException::check(total_number_subdomains >= num_ranks, "Partition::Partition: Total num local_subdomains is smaller than number ranks: {} vs {}", total_number_subdomains, num_ranks);

    /**
	 * Calc my number of local_subdomains
	 *
	 * NOTE:
	 * Every rank gets the same number of local_subdomains first.
	 * The remaining m local_subdomains are then assigned to the first m ranks,
	 * one subdomain more per rank.
	 *
	 * For #procs = 2^n and 8^level_of_subdomain_trees local_subdomains, every proc's #local_subdomains is the same power of two of {1, 2, 4}.
	 */
    // NOLINTNEXTLINE
    number_local_subdomains = total_number_subdomains / num_ranks;
    const size_t rest = total_number_subdomains % num_ranks;
    number_local_subdomains += (my_rank < rest) ? 1 : 0;

    if (rest != 0) {
        LogFiles::print_message_rank(-1, "My rank is: {}; There are {} ranks in total; The rest is: {}", my_rank, num_ranks, rest);
        RelearnException::fail("Partition::Partition: Number of ranks must be of the form 2^n but was {}", num_ranks);
    }

    /**
	 * Set parameter of space filling curve before it can be used.
	 * total_number_subdomains = 8^level_of_subdomain_trees = (2^3)^level_of_subdomain_trees = 2^(3*level_of_subdomain_trees).
	 * Thus, number of local_subdomains per dimension (3d) is (2^(3*level_of_subdomain_trees))^(1/3) = 2^level_of_subdomain_trees.
	 */
    number_subdomains_per_dimension = 1ULL << level_of_subdomain_trees;
    space_curve.set_refinement_level(level_of_subdomain_trees);

    // Calc start and end index of subdomain
    local_subdomain_id_start = (total_number_subdomains / num_ranks) * my_rank;
    local_subdomain_id_end = local_subdomain_id_start + number_local_subdomains - 1;

    // Allocate vector with my number of local_subdomains
    local_subdomains = std::vector<Subdomain>(number_local_subdomains);

    for (size_t i = 0; i < number_local_subdomains; i++) {
        Subdomain& current_subdomain = local_subdomains[i];

        // Set space filling curve indices in 1d and 3d
        current_subdomain.index_1d = local_subdomain_id_start + i;
        current_subdomain.index_3d = space_curve.map_1d_to_3d(static_cast<uint64_t>(current_subdomain.index_1d));
    }

    LogFiles::print_message_rank(0, "Total number local_subdomains        : {}", total_number_subdomains);
    LogFiles::print_message_rank(0, "Number local_subdomains per dimension: {}", number_subdomains_per_dimension);
}

void Partition::print_my_subdomains_info_rank(const int rank) {
    std::stringstream sstream{};

    sstream << "My number of neurons   : " << number_local_neurons << "\n";
    sstream << "My number of local_subdomains: " << number_local_subdomains << "\n";
    sstream << "My subdomain ids       : [ " << local_subdomain_id_start
            << " , "
            << local_subdomain_id_end
            << " ]"
            << "\n";

    for (size_t i = 0; i < number_local_subdomains; i++) {
        sstream << "Subdomain: " << i << "\n";
        sstream << "    number_neurons: " << local_subdomains[i].number_neurons << "\n";
        sstream << "    index_1d   : " << local_subdomains[i].index_1d << "\n";

        sstream << "    index_3d   : "
                << "( " << local_subdomains[i].index_3d.get_x()
                << " , " << local_subdomains[i].index_3d.get_y()
                << " , " << local_subdomains[i].index_3d.get_z()
                << " )"
                << "\n";

        sstream << "    minimum_position    : "
                << "( " << local_subdomains[i].minimum_position.get_x()
                << " , " << local_subdomains[i].minimum_position.get_y()
                << " , " << local_subdomains[i].minimum_position.get_z()
                << " )"
                << "\n";

        sstream << "    maximum_position    : "
                << "( " << local_subdomains[i].maximum_position.get_x()
                << " , " << local_subdomains[i].maximum_position.get_y()
                << " , " << local_subdomains[i].maximum_position.get_z()
                << " )\n";
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, false, sstream.str());
}

void Partition::set_simulation_box_size(const Vec3d& min, const Vec3d& max) {
    simulation_box_length = max - min;
    const auto& subdomain_length = simulation_box_length / static_cast<double>(number_subdomains_per_dimension);

    LogFiles::print_message_rank(0, "Simulation box length (height, width, depth)\t: ({}, {}, {})",
        simulation_box_length.get_x(), simulation_box_length.get_y(), simulation_box_length.get_z());
    LogFiles::print_message_rank(0, "Subdomain length (height, width, depth)\t: ({}, {}, {})",
        subdomain_length.get_x(), subdomain_length.get_y(), subdomain_length.get_z());
}
