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

#include "RelearnException.h"

Partition::Partition(int num_ranks, int my_rank) : my_num_neurons(0), total_num_neurons(0) {
	/**
	* Total number of subdomains is smallest power of 8 that is >= num_ranks.
	* We choose power of 8 as every domain subdivision creates 8 subdomains (in 3d).
	*/
	const double smallest_exponent = ceil(log(num_ranks) / log(8.0));
	level_of_subdomain_trees = static_cast<size_t>(smallest_exponent);
	total_num_subdomains = 1ull << (3 * level_of_subdomain_trees); // 8^level_of_subdomain_trees

																   // Every rank should get at least one subdomain
	RelearnException::check(total_num_subdomains >= num_ranks);

	/**
	* Set parameter of space filling curve before it can be used.
	* total_num_subdomains = 8^level_of_subdomain_trees = (2^3)^level_of_subdomain_trees = 2^(3*level_of_subdomain_trees).
	* Thus, number of subdomains per dimension (3d) is (2^(3*level_of_subdomain_trees))^(1/3) = 2^level_of_subdomain_trees.
	*/
	num_subdomains_per_dimension = 1ull << level_of_subdomain_trees;
	space_curve.set_refinement_level(level_of_subdomain_trees);

	std::stringstream sstream;
	sstream << "Total number subdomains        : " << total_num_subdomains;
	LogMessages::print_message_rank(sstream.str().c_str(), 0);
	sstream.str("");

	sstream << "Number subdomains per dimension: " << num_subdomains_per_dimension;
	LogMessages::print_message_rank(sstream.str().c_str(), 0);
	sstream.str("");

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
	my_num_subdomains = total_num_subdomains / num_ranks;
	const size_t rest = total_num_subdomains % num_ranks;
	my_num_subdomains += (my_rank < rest) ? 1 : 0;

	if (rest != 0) {
		std::stringstream sstream;
		sstream << "My rank is: " << my_rank << "; There are " << num_ranks << " ranks in total; The rest is: " << rest << "\n";
		std::cout << sstream.str().c_str() << std::flush;
		sstream.str("");
		RelearnException::fail("rest != 0");
	}

	// Calc start and end index of subdomain
	my_subdomain_id_start = (total_num_subdomains / num_ranks) * my_rank;
	my_subdomain_id_start += std::min(rest, static_cast<size_t>(my_rank));
	my_subdomain_id_end = my_subdomain_id_start + my_num_subdomains - 1;

	// Allocate vector with my number of subdomains
	subdomains = std::vector<Subdomain>(my_num_subdomains);

	for (size_t i = 0; i < my_num_subdomains; i++) {
		Subdomain& current_subdomain = subdomains[i];

		// Set space filling curve indices in 1d and 3d
		BoxCoordinates box_coords;
		current_subdomain.index_1d = my_subdomain_id_start + i;
		space_curve.map_1d_to_3d(static_cast<uint64_t>(current_subdomain.index_1d), box_coords);
		current_subdomain.index_3d = box_coords;
	}
}

void Partition::print_my_subdomains_info_rank(int rank) {
	std::stringstream sstream;

	sstream << "My number of neurons   : " << my_num_neurons;
	LogMessages::print_message_rank(sstream.str().c_str(), rank);
	sstream.str("");

	sstream << "My number of subdomains: " << my_num_subdomains;
	LogMessages::print_message_rank(sstream.str().c_str(), rank);
	sstream.str("");

	sstream << "My subdomain ids       : [ " << my_subdomain_id_start
		<< " , "
		<< my_subdomain_id_end
		<< " ]";
	LogMessages::print_message_rank(sstream.str().c_str(), rank);
	sstream.str("");


	for (size_t i = 0; i < my_num_subdomains; i++) {
		sstream << "Subdomain: " << i;
		LogMessages::print_message_rank(sstream.str().c_str(), rank);
		sstream.str("");

		sstream << "    num_neurons: " << subdomains[i].num_neurons;
		LogMessages::print_message_rank(sstream.str().c_str(), rank);
		sstream.str("");

		sstream << "    index_1d   : " << subdomains[i].index_1d;
		LogMessages::print_message_rank(sstream.str().c_str(), rank);
		sstream.str("");

		sstream << "    index_3d   : " << "( " << subdomains[i].index_3d[0]
			<< " , " << subdomains[i].index_3d[1]
			<< " , " << subdomains[i].index_3d[2]
			<< " )";
		LogMessages::print_message_rank(sstream.str().c_str(), rank);
		sstream.str("");

		sstream << "    xyz_min    : " << "( " << subdomains[i].xyz_min[0]
			<< " , " << subdomains[i].xyz_min[1]
			<< " , " << subdomains[i].xyz_min[2]
			<< " )";
		LogMessages::print_message_rank(sstream.str().c_str(), rank);
		sstream.str("");

		sstream << "    xyz_max    : " << "( " << subdomains[i].xyz_max[0]
			<< " , " << subdomains[i].xyz_max[1]
			<< " , " << subdomains[i].xyz_max[2]
			<< " )\n";
		LogMessages::print_message_rank(sstream.str().c_str(), rank);
		sstream.str("");
	}
}

Neurons Partition::load_neurons(const Parameters& params, NeuronToSubdomainAssignment& neurons_in_subdomain) {

	simulation_box_length = neurons_in_subdomain.get_simulation_box_length();

	// Set subdomain length
	Vec3d subdomain_length = simulation_box_length / static_cast<double>(num_subdomains_per_dimension);

	/**
	* Output all parameters calculated so far
	*/
	std::stringstream sstream;
	sstream << "Simulation box length          : " << simulation_box_length.x << " (height)" << "\n";
	sstream << "Simulation box length          : " << simulation_box_length.y << " (width)" << "\n";
	sstream << "Simulation box length          : " << simulation_box_length.z << " (depth)" << "\n";
	LogMessages::print_message_rank(sstream.str().c_str(), 0);
	sstream.str("");

	sstream << "Subdomain length          : " << subdomain_length.x << " (height)" << "\n";
	sstream << "Subdomain length          : " << subdomain_length.y << " (width)" << "\n";
	sstream << "Subdomain length          : " << subdomain_length.z << " (depth)" << "\n";
	LogMessages::print_message_rank(sstream.str().c_str(), 0);
	sstream.str("");

	my_num_neurons = 0;
	for (size_t i = 0; i < my_num_subdomains; i++) {
		Subdomain& current_subdomain = subdomains[i];

		// Set position of subdomain
		neurons_in_subdomain.get_subdomain_boundaries(current_subdomain.index_3d,
			num_subdomains_per_dimension, current_subdomain.xyz_min, current_subdomain.xyz_max);

		// Set number of neurons in this subdomain
		const auto& xyz_min = current_subdomain.xyz_min;
		const auto& xyz_max = current_subdomain.xyz_max;

		neurons_in_subdomain.fill_subdomain(current_subdomain.index_1d,
			total_num_subdomains, xyz_min, xyz_max);

		current_subdomain.num_neurons =
			neurons_in_subdomain.num_neurons(current_subdomain.index_1d,
				total_num_subdomains, xyz_min, xyz_max);

		// Add subdomain's number of neurons to rank's number of neurons
		my_num_neurons += current_subdomain.num_neurons;

		// Set start and end of local neuron ids
		// 0-th subdomain starts with neuron id 0
		current_subdomain.neuron_local_id_start = (i == 0) ? 0 : (subdomains[i - 1].neuron_local_id_end + 1);
		current_subdomain.neuron_local_id_end = current_subdomain.neuron_local_id_start + current_subdomain.num_neurons - 1;

		neurons_in_subdomain.neuron_global_ids(current_subdomain.index_1d,
			total_num_subdomains,
			current_subdomain.neuron_local_id_start,
			current_subdomain.neuron_local_id_end,
			current_subdomain.global_neuron_ids);

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

	Neurons neurons(my_num_neurons, params, *this);

	Positions& neuron_positions = neurons.get_positions();
	SynapticElements& axons = neurons.get_axons();
	std::vector<std::string>& area_names = neurons.get_area_names();

	for (size_t i = 0; i < my_num_subdomains; i++) {
		const auto& subdomain_pos_min = subdomains[i].xyz_min;
		const auto& subdomain_pos_max = subdomains[i].xyz_max;

		const auto subdomain_idx = i + my_subdomain_id_start;

		const auto subdomain_num_neurons =
			neurons_in_subdomain.num_neurons(subdomain_idx, my_num_subdomains, subdomain_pos_min, subdomain_pos_max);

		// Get neuron positions in subdomain i
		std::vector<NeuronToSubdomainAssignment::Position> vec_pos;
		vec_pos.reserve(subdomain_num_neurons);
		neurons_in_subdomain.neuron_positions(subdomain_idx, total_num_subdomains,
			subdomain_pos_min, subdomain_pos_max, vec_pos);

		// Get neuron area names in subdomain i
		std::vector<std::string> vec_area;
		vec_area.reserve(subdomain_num_neurons);
		neurons_in_subdomain.neuron_area_names(subdomain_idx, total_num_subdomains,
			subdomain_pos_min, subdomain_pos_max, vec_area);

		// Get neuron types in subdomain i
		std::vector<SynapticElements::SignalType> vec_type;
		vec_type.reserve(subdomain_num_neurons);
		neurons_in_subdomain.neuron_types(subdomain_idx, total_num_subdomains,
			subdomain_pos_min, subdomain_pos_max, vec_type);

		size_t neuron_id = subdomains[i].neuron_local_id_start;
		for (size_t j = 0; j < subdomains[i].num_neurons; j++) {
			neuron_positions.set_x(neuron_id, vec_pos[j].x);
			neuron_positions.set_y(neuron_id, vec_pos[j].y);
			neuron_positions.set_z(neuron_id, vec_pos[j].z);

			area_names[neuron_id] = vec_area[j];

			// Mark neuron as DendriteType::EXCITATORY or DendriteType::INHIBITORY
			axons.set_signal_type(neuron_id, vec_type[j]);

			// Insert neuron into tree
			subdomains[i].octree.insert(vec_pos[j], neuron_id, MPIWrapper::my_rank);

			neuron_id++;
		}
	}

	return neurons;
}
