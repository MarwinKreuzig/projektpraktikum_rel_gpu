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

#include "LogFiles.h"
#include "NeuronToSubdomainAssignment.h"
#include "Neurons.h"
#include "RelearnException.h"
#include "Vec3.h"

#include <sstream>

Partition::Partition(size_t num_ranks, size_t my_rank)
    : my_num_neurons(0)
    , total_num_neurons(0)
    , neurons_loaded(false) {
    RelearnException::check(num_ranks > 0, "Number of MPI ranks must be a positive number");
    RelearnException::check(num_ranks > my_rank, "My rank must be smaller than number of ranks");

    /**
	* Total number of subdomains is smallest power of 8 that is >= num_ranks.
	* We choose power of 8 as every domain subdivision creates 8 subdomains (in 3d).
	*/
    const double smallest_exponent = ceil(log(num_ranks) / log(8.0));
    level_of_subdomain_trees = static_cast<size_t>(smallest_exponent);
    total_num_subdomains = 1ULL << (3 * level_of_subdomain_trees); // 8^level_of_subdomain_trees

    // Every rank should get at least one subdomain
    RelearnException::check(total_num_subdomains >= num_ranks, "In partition, total num subdomains is smaller than number ranks");

    /**
	* Calc my number of subdomains
	*
	* NOTE:
	* Every rank gets the same number of subdomains first.
	* The remaining m subdomains are then assigned to the first m ranks,
	* one subdomain more per rank.
	*
	* For #procs = 2^n and 8^level_of_subdomain_trees subdomains, every proc's #subdomains is the same power of two of {1, 2, 4}.
	*/
    // NOLINTNEXTLINE
    my_num_subdomains = total_num_subdomains / num_ranks;
    const size_t rest = total_num_subdomains % num_ranks;
    my_num_subdomains += (my_rank < rest) ? 1 : 0;

    if (rest != 0) {
        std::stringstream sstream;
        sstream << "My rank is: " << my_rank << "; There are " << num_ranks << " ranks in total; The rest is: " << rest << "\n";
        sstream << std::flush;
        LogFiles::print_message_rank(sstream.str().c_str(), -1);
        RelearnException::fail("Number of ranks must be of the form 2^n");
    }

    /**
	* Set parameter of space filling curve before it can be used.
	* total_num_subdomains = 8^level_of_subdomain_trees = (2^3)^level_of_subdomain_trees = 2^(3*level_of_subdomain_trees).
	* Thus, number of subdomains per dimension (3d) is (2^(3*level_of_subdomain_trees))^(1/3) = 2^level_of_subdomain_trees.
	*/
    num_subdomains_per_dimension = 1ULL << level_of_subdomain_trees;
    space_curve.set_refinement_level(level_of_subdomain_trees);

    // Calc start and end index of subdomain
    my_subdomain_id_start = (total_num_subdomains / num_ranks) * my_rank;
    my_subdomain_id_end = my_subdomain_id_start + my_num_subdomains - 1;

    // Allocate vector with my number of subdomains
    subdomains = std::vector<Subdomain>(my_num_subdomains);

    for (size_t i = 0; i < my_num_subdomains; i++) {
        Subdomain& current_subdomain = subdomains[i];

        // Set space filling curve indices in 1d and 3d
        current_subdomain.index_1d = my_subdomain_id_start + i;
        BoxCoordinates box_coords;
        box_coords = space_curve.map_1d_to_3d(static_cast<uint64_t>(current_subdomain.index_1d));
        current_subdomain.index_3d = box_coords;
    }

    std::stringstream sstream;
    sstream << "Total number subdomains        : " << total_num_subdomains << "\n";
    sstream << "Number subdomains per dimension: " << num_subdomains_per_dimension;
    LogFiles::print_message_rank(sstream.str().c_str(), 0);
}

void Partition::print_my_subdomains_info_rank(int rank) {
    std::stringstream sstream;

    sstream << "My number of neurons   : " << my_num_neurons << "\n";
    sstream << "My number of subdomains: " << my_num_subdomains << "\n";
    sstream << "My subdomain ids       : [ " << my_subdomain_id_start
            << " , "
            << my_subdomain_id_end
            << " ]"
            << "\n";

    for (size_t i = 0; i < my_num_subdomains; i++) {
        sstream << "Subdomain: " << i << "\n";
        sstream << "    num_neurons: " << subdomains[i].num_neurons << "\n";
        sstream << "    index_1d   : " << subdomains[i].index_1d << "\n";

        sstream << "    index_3d   : "
                << "( " << subdomains[i].index_3d[0]
                << " , " << subdomains[i].index_3d[1]
                << " , " << subdomains[i].index_3d[2]
                << " )"
                << "\n";

        sstream << "    xyz_min    : "
                << "( " << subdomains[i].xyz_min[0]
                << " , " << subdomains[i].xyz_min[1]
                << " , " << subdomains[i].xyz_min[2]
                << " )"
                << "\n";

        sstream << "    xyz_max    : "
                << "( " << subdomains[i].xyz_max[0]
                << " , " << subdomains[i].xyz_max[1]
                << " , " << subdomains[i].xyz_max[2]
                << " )\n";
    }

    LogFiles::print_message_rank(sstream.str().c_str(), rank);
}

bool Partition::is_neuron_local(size_t neuron_id) const {
    RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
    for (const Subdomain& subdomain : subdomains) {
        const bool found = std::binary_search(subdomain.global_neuron_ids.begin(), subdomain.global_neuron_ids.end(), neuron_id);
        if (found) {
            return true;
        }
    }

    return false;
}

size_t Partition::get_subdomain_id_from_pos(const Vec3d& pos) const {
    RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
    const Vec3d subdomain_length = simulation_box_length / static_cast<double>(num_subdomains_per_dimension);

    const Vec3d subdomain_3d{ pos.get_x() / subdomain_length.get_x(), pos.get_y() / subdomain_length.get_y(), pos.get_z() / subdomain_length.get_z() };
    const Vec3s id_3d = subdomain_3d.floor_componentwise();
    const size_t id_1d = space_curve.map_3d_to_1d(id_3d);

    const size_t rank = id_1d / my_num_subdomains;

    return rank;
}

size_t Partition::get_global_id(size_t local_id) const {
    RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
    size_t counter = 0;
    for (const auto& subdomain : subdomains) {
        const size_t old_counter = counter;

        counter += subdomain.global_neuron_ids.size();
        if (local_id < counter) {
            const size_t local_local_id = local_id - old_counter;
            return subdomain.global_neuron_ids[local_local_id];
        }
    }

    return local_id;
}

size_t Partition::get_local_id(size_t global_id) const {
    RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
    size_t id = 0;

    for (const Subdomain& current_subdomain : subdomains) {
        const std::vector<size_t>& ids = current_subdomain.global_neuron_ids;
        const auto pos = std::lower_bound(ids.begin(), ids.end(), global_id);

        if (pos != ids.end()) {
            id += pos - ids.begin();
            return id;
        }

        id += ids.size();
    }

    RelearnException::fail("Didn't find global id in Partition.h");
    return 0;
}

size_t Partition::get_total_num_neurons() const noexcept {
    return total_num_neurons;
}

void Partition::set_total_num_neurons(size_t total_num) noexcept {
    total_num_neurons = total_num;
}

void Partition::load_data_from_subdomain_assignment(const std::shared_ptr<Neurons>& neurons, std::unique_ptr<NeuronToSubdomainAssignment> neurons_in_subdomain) {
    RelearnException::check(!neurons_loaded, "Neurons are already loaded, cannot load anymore");

    simulation_box_length = neurons_in_subdomain->get_simulation_box_length();

    // Set subdomain length
    const Vec3d subdomain_length = simulation_box_length / static_cast<double>(num_subdomains_per_dimension);

    /**
	* Output all parameters calculated so far
	*/
    std::stringstream sstream;
    sstream << "Simulation box length          : " << simulation_box_length.get_x() << " (height)"
            << "\n";
    sstream << "Simulation box length          : " << simulation_box_length.get_y() << " (width)"
            << "\n";
    sstream << "Simulation box length          : " << simulation_box_length.get_z() << " (depth)"
            << "\n";
    sstream << "Subdomain length          : " << subdomain_length.get_x() << " (height)"
            << "\n";
    sstream << "Subdomain length          : " << subdomain_length.get_y() << " (width)"
            << "\n";
    sstream << "Subdomain length          : " << subdomain_length.get_z() << " (depth)";
    LogFiles::print_message_rank(sstream.str().c_str(), 0);

    my_num_neurons = 0;
    for (size_t i = 0; i < my_num_subdomains; i++) {
        Subdomain& current_subdomain = subdomains[i];

        // Set position of subdomain
        std::tie(current_subdomain.xyz_min, current_subdomain.xyz_max) = neurons_in_subdomain->get_subdomain_boundaries(current_subdomain.index_3d,
            num_subdomains_per_dimension);

        // Set number of neurons in this subdomain
        const auto& xyz_min = current_subdomain.xyz_min;
        const auto& xyz_max = current_subdomain.xyz_max;

        neurons_in_subdomain->fill_subdomain(current_subdomain.index_1d,
            total_num_subdomains, xyz_min, xyz_max);

        current_subdomain.num_neurons = neurons_in_subdomain->num_neurons(current_subdomain.index_1d,
            total_num_subdomains, xyz_min, xyz_max);

        // Add subdomain's number of neurons to rank's number of neurons
        my_num_neurons += current_subdomain.num_neurons;

        // Set start and end of local neuron ids
        // 0-th subdomain starts with neuron id 0
        current_subdomain.neuron_local_id_start = (i == 0) ? 0 : (subdomains[i - 1].neuron_local_id_end + 1);
        current_subdomain.neuron_local_id_end = current_subdomain.neuron_local_id_start + current_subdomain.num_neurons - 1;

        current_subdomain.global_neuron_ids = neurons_in_subdomain->neuron_global_ids(current_subdomain.index_1d,
            total_num_subdomains,
            current_subdomain.neuron_local_id_start,
            current_subdomain.neuron_local_id_end);

        std::sort(current_subdomain.global_neuron_ids.begin(), current_subdomain.global_neuron_ids.end());

        /**
		* Set octree parameters.
		* Only those that are necessary for
		* inserting neurons into the tree
		*/
        // Init domain size
        current_subdomain.octree.set_size(current_subdomain.xyz_min, current_subdomain.xyz_max);
        // Set tree's root level
        // It determines later at which level this
        // local tree will be inserted into the global tree
        current_subdomain.octree.set_root_level(level_of_subdomain_trees);

        // Tree's destructor should not free the tree nodes
        // The freeing is done by the global tree destructor later
        // as the nodes in this tree will be attached to the global tree
        current_subdomain.octree.set_no_free_in_destructor();
    }

    neurons->init(my_num_neurons);

    std::vector<double> x_dims(my_num_neurons);
    std::vector<double> y_dims(my_num_neurons);
    std::vector<double> z_dims(my_num_neurons);

    std::vector<std::string> area_names(my_num_neurons);
    std::vector<SignalType> signal_types(my_num_neurons);

    for (size_t i = 0; i < my_num_subdomains; i++) {
        const auto& subdomain_pos_min = subdomains[i].xyz_min;
        const auto& subdomain_pos_max = subdomains[i].xyz_max;

        const auto subdomain_idx = i + my_subdomain_id_start;

        // Get neuron positions in subdomain i
        std::vector<NeuronToSubdomainAssignment::Position> vec_pos = neurons_in_subdomain->neuron_positions(subdomain_idx, total_num_subdomains,
            subdomain_pos_min, subdomain_pos_max);

        // Get neuron area names in subdomain i
        std::vector<std::string> vec_area = neurons_in_subdomain->neuron_area_names(subdomain_idx, total_num_subdomains,
            subdomain_pos_min, subdomain_pos_max);

        // Get neuron types in subdomain i
        std::vector<SignalType> vec_type = neurons_in_subdomain->neuron_types(subdomain_idx, total_num_subdomains,
            subdomain_pos_min, subdomain_pos_max);

        size_t neuron_id = subdomains[i].neuron_local_id_start;
        for (size_t j = 0; j < subdomains[i].num_neurons; j++) {
            x_dims[neuron_id] = vec_pos[j].get_x();
            y_dims[neuron_id] = vec_pos[j].get_y();
            z_dims[neuron_id] = vec_pos[j].get_z();

            area_names[neuron_id] = vec_area[j];

            // Mark neuron as DendriteType::EXCITATORY or DendriteType::INHIBITORY
            signal_types[neuron_id] = vec_type[j];

            // Insert neuron into tree
            auto* node = subdomains[i].octree.insert(vec_pos[j], neuron_id, MPIWrapper::get_my_rank());
            RelearnException::check(node != nullptr, "node is nullptr");

            neuron_id++;
        }
    }

    neurons->set_area_names(std::move(area_names));
    neurons->set_x_dims(std::move(x_dims));
    neurons->set_y_dims(std::move(y_dims));
    neurons->set_z_dims(std::move(z_dims));
    neurons->set_signal_types(std::move(signal_types));

    neurons_loaded = true;
}
