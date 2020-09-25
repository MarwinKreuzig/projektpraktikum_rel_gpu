/*
 * File:   Partition.h
 * Author: rinke
 *
 * Created on Feb 27, 2017
 */

#ifndef PARTITION_H
#define PARTITION_H

#include <cmath>
#include <cassert>
#include <algorithm>
#include <sstream>
#include <random>

#include "SpaceFillingCurve.h"
#include "LogMessages.h"
#include "Octree.h"
#include "MPI_RMA_MemAllocator.h"
#include "Positions.h"
#include "SynapticElements.h"
#include "randomNumberSeeds.h"
#include "NeuronToSubdomainAssignment.h"
#include "Random.h"
#include "Vec3.h"

class Partition {
public:
	struct Subdomain {
		Vec3d xyz_min;
		Vec3d xyz_max;

		size_t num_neurons;

		// Local start and end neuron id
		size_t neuron_id_start;
		size_t neuron_id_end;

		size_t index_1d;
		//size_t index_3d[3];  // [0] = x, [1] = y, [2] = z

		Vec3<size_t> index_3d;

		// The octree contains all neurons in
		// this subdomain. It is only used as a container
		// for the neurons
		Octree octree;
	};

	Partition(size_t num_neurons,
		int num_ranks,
		int my_rank,
		MPI_RMA_MemAllocator<OctreeNode>& mpi_rma_mem_allocator,
		NeuronsInSubdomain& neurons_in_subdomain) :
		total_num_neurons(num_neurons),
		num_ranks(num_ranks),
		my_rank(my_rank),
		mpi_rma_mem_allocator(mpi_rma_mem_allocator),
		random_number_generator(RandomHolder<Partition>::get_random_generator()) {

		// Init seed of random number generator
		random_number_generator.seed(randomNumberSeeds::partition);

		simulation_box_length = neurons_in_subdomain.simulation_box_length();

		/**
		 * Total number of subdomains is smallest power of 8 that is >= num_ranks.
		 * We choose power of 8 as every domain subdivision creates 8 subdomains (in 3d).
		 */
		const size_t k = static_cast<size_t>(ceil(log(num_ranks) / log(8.)));
		total_num_subdomains = 1ull << (3 * k); // 8^k
		level_of_subdomain_trees = k;

		// Every rank should get at least one subdomain
		assert(total_num_subdomains >= num_ranks);

		/**
		 * Set parameter of space filling curve before it can be used.
		 * total_num_subdomains = 8^k = (2^3)^k = 2^(3k).
		 * Thus, number of subdomains per dimension (3d) is (2^(3k))^(1/3) = 2^k.
		 */
		num_subdomains_per_dimension = 1ull << k;
		space_curve.set_refinement_level(k);

		// Set subdomain length
		subdomain_length = simulation_box_length / num_subdomains_per_dimension;


		/**
		 * Output all parameters calculated so far
		 */
		std::stringstream sstream;
		sstream << "Simulation box length          : " << simulation_box_length << " (height = width = depth)";
		LogMessages::print_message_rank(sstream.str().c_str(), 0);
		sstream.str("");

		sstream << "Subdomain length               : " << subdomain_length << " (height = width = depth)";
		LogMessages::print_message_rank(sstream.str().c_str(), 0);
		sstream.str("");

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
		 * For #procs = 2^n and 8^k subdomains, every proc's #subdomains is the same power of two of {1, 2, 4}.
		 */
		my_num_subdomains = total_num_subdomains / num_ranks;
		const size_t rest = total_num_subdomains % num_ranks;
		my_num_subdomains += (my_rank < rest) ? 1 : 0;

		if (rest != 0) {
			sstream << "My rank is: " << my_rank << "; There are " << num_ranks << " ranks in total; The rest is: " << rest << "\n";
			std::cout << sstream.str().c_str() << std::flush;
			sstream.str("");
		}

		// Calc start and end index of subdomain
		my_subdomain_id_start = (total_num_subdomains / num_ranks) * my_rank;
		my_subdomain_id_start += std::min(rest, (size_t)my_rank);
		my_subdomain_id_end = my_subdomain_id_start + my_num_subdomains - 1;

		// Allocate array with my number of subdomains
		subdomains = new Subdomain[my_num_subdomains];

		/**
		 * Every subdomain in the simulation box gets the value of how many neurons it contains.
		 */
		BoxCoordinates box_coords;
		my_num_neurons = 0;
		for (size_t i = 0; i < my_num_subdomains; i++) {
			auto& current_subdomain = subdomains[i];

			// Set space filling curve indices in 1d and 3d
			current_subdomain.index_1d = my_subdomain_id_start + i;
			space_curve.map_1d_to_3d(static_cast<uint64_t>(current_subdomain.index_1d), box_coords);
			current_subdomain.index_3d = box_coords;

			// Set position of subdomain
			neurons_in_subdomain.get_subdomain_boundaries(current_subdomain.index_3d,
				num_subdomains_per_dimension, current_subdomain.xyz_min, current_subdomain.xyz_max);

			// Set number of neurons in this subdomain
			const auto& xyz_min = current_subdomain.xyz_min;
			const auto& xyz_max = current_subdomain.xyz_max;

			neurons_in_subdomain.lazily_fill(current_subdomain.index_1d,
				total_num_subdomains, xyz_min, xyz_max);

			current_subdomain.num_neurons =
				neurons_in_subdomain.num_neurons(current_subdomain.index_1d,
					total_num_subdomains, xyz_min, xyz_max);

			// Add subdomain's number of neurons to rank's number of neurons
			my_num_neurons += current_subdomain.num_neurons;

			// Set start and end of local neuron ids
			// 0-th subdomain starts with neuron id 0
			current_subdomain.neuron_id_start = (i == 0) ? 0 : (subdomains[i - 1].neuron_id_end + 1);
			current_subdomain.neuron_id_end = current_subdomain.neuron_id_start + current_subdomain.num_neurons - 1;

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

			// Provide MPI RMA memory allocator
			current_subdomain.octree.set_mpi_rma_mem_allocator(&mpi_rma_mem_allocator);
		}
	}

	~Partition() {
		delete[] subdomains;
	}

	Partition(const Partition& other) = delete;
	Partition(Partition&& other) = delete;

	Partition& operator=(const Partition& other) = delete;
	Partition& operator=(Partition&& other) = delete;

	void print_my_subdomains_info_rank(int rank) {
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

	// We need the "axons" parameter to set for every neuron the type of axons it grows (exc./inh.)
	void insert_neurons_into_my_subdomains(const NeuronsInSubdomain& neurons_in_subdomain,
		Positions& neuron_positions, SynapticElements& axons,
		std::vector<std::string>& area_names) {
		for (size_t i = 0; i < my_num_subdomains; i++) {
			const auto subdomain_pos_min = subdomains[i].xyz_min;
			const auto subdomain_pos_max = subdomains[i].xyz_max;

			// Get neuron positions in subdomain i
			std::vector<NeuronsInSubdomain::Position> vec_pos;
			neurons_in_subdomain.neuron_positions(i, total_num_subdomains,
				subdomain_pos_min, subdomain_pos_max, vec_pos);

			// Get neuron area names in subdomain i
			std::vector<std::string> vec_area;
			neurons_in_subdomain.neuron_area_names(i, total_num_subdomains,
				subdomain_pos_min, subdomain_pos_max, vec_area);

			// Get neuron types in subdomain i
			std::vector<SynapticElements::SignalType> vec_type;
			neurons_in_subdomain.neuron_types(i, total_num_subdomains,
				subdomain_pos_min, subdomain_pos_max, vec_type);

			size_t neuron_id = subdomains[i].neuron_id_start;
			for (size_t j = 0; j < subdomains[i].num_neurons; j++) {
				neuron_positions.set_x(neuron_id, vec_pos[j].x);
				neuron_positions.set_y(neuron_id, vec_pos[j].y);
				neuron_positions.set_z(neuron_id, vec_pos[j].z);

				area_names[neuron_id] = vec_area[j];

				// Mark neuron as EXCITATORY or INHIBITORY
				axons.set_signal_type(neuron_id, vec_type[j]);

				// Insert neuron into tree
				subdomains[i].octree.insert({ vec_pos[j].x, vec_pos[j].y, vec_pos[j].z }, neuron_id, my_rank);

				neuron_id++;
			}
		}
	}

	inline size_t get_my_num_neurons() const noexcept {
		return my_num_neurons;
	}

	inline size_t get_my_num_subdomains() const noexcept {
		return my_num_subdomains;
	}

	inline void get_simulation_box_size(Vec3d& min, Vec3d& max) const noexcept {
		min = Vec3d(0);
		max = Vec3d(simulation_box_length);
	}

	Octree& get_subdomain_tree(size_t subdomain_id) noexcept {
		assert(subdomain_id < my_num_subdomains);

		return subdomains[subdomain_id].octree;
	}

	inline size_t get_my_subdomain_id_start() const noexcept {
		return my_subdomain_id_start;
	}

	inline size_t get_my_subdomain_id_end() const noexcept {
		return my_subdomain_id_end;
	}

	inline size_t get_level_of_subdomain_trees() const noexcept {
		return level_of_subdomain_trees;
	}

	inline size_t get_total_num_subdomains() const noexcept {
		return total_num_subdomains;
	}

private:
	size_t total_num_neurons;
	size_t my_num_neurons;
	double simulation_box_length;
	double subdomain_length;

	size_t total_num_subdomains;
	size_t num_subdomains_per_dimension;
	size_t level_of_subdomain_trees;
	size_t my_num_subdomains;
	size_t my_subdomain_id_start;
	size_t my_subdomain_id_end;

	Subdomain* subdomains;
	SpaceFillingCurve<Morton> space_curve;

	int num_ranks;
	int my_rank;

	// The memory allocator for the octree of each of my subdomains
	MPI_RMA_MemAllocator<OctreeNode>& mpi_rma_mem_allocator;

	// Randpm number generator for this class (C++11)
	std::mt19937& random_number_generator;
};

#endif /* PARTITION_H */
