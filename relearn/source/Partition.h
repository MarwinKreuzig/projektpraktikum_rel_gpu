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
#include <vector>


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

#include "Neurons.h"

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

		Vec3<size_t> index_3d;

		// The octree contains all neurons in
		// this subdomain. It is only used as a container
		// for the neurons
		Octree octree;
	};

	Partition(int num_ranks, int my_rank);

	~Partition() = default;

	Partition(const Partition& other) = delete;
	Partition(Partition&& other) = delete;

	Partition& operator=(const Partition& other) = delete;
	Partition& operator=(Partition&& other) = delete;

	void print_my_subdomains_info_rank(int rank);

	void set_mpi_rma_mem_allocator(MPI_RMA_MemAllocator<OctreeNode>& mpi_rma_mem_allocator) {
		for (Subdomain& current_subdomain : subdomains) {
			// Provide MPI RMA memory allocator
			current_subdomain.octree.set_mpi_rma_mem_allocator(&mpi_rma_mem_allocator);
		}
	}

	Neurons get_local_neurons(const Parameters& params, NeuronToSubdomainAssignment& neurons_in_subdomain) {
		Neurons neurons = load_neurons(params, neurons_in_subdomain);
		return neurons;
	}

	size_t get_my_num_neurons() const noexcept {
		return my_num_neurons;
	}

	size_t get_my_num_subdomains() const noexcept {
		return my_num_subdomains;
	}

	void get_simulation_box_size(Vec3d& min, Vec3d& max) const noexcept {
		min = Vec3d(0);
		max = Vec3d(simulation_box_length);
	}

	Octree& get_subdomain_tree(size_t subdomain_id) noexcept {
		assert(subdomain_id < my_num_subdomains);

		return subdomains[subdomain_id].octree;
	}

	size_t get_my_subdomain_id_start() const noexcept {
		return my_subdomain_id_start;
	}

	size_t get_my_subdomain_id_end() const noexcept {
		return my_subdomain_id_end;
	}

	size_t get_level_of_subdomain_trees() const noexcept {
		return level_of_subdomain_trees;
	}

	size_t get_total_num_subdomains() const noexcept {
		return total_num_subdomains;
	}

	size_t get_num_subdomains_per_dimension() const noexcept {
		return num_subdomains_per_dimension;
	}

private:
	// We need the "axons" parameter to set for every neuron the type of axons it grows (exc./inh.)
	Neurons load_neurons(const Parameters& params, NeuronToSubdomainAssignment& neurons_in_subdomain);

	size_t total_num_neurons;
	size_t my_num_neurons;

	size_t total_num_subdomains;
	size_t num_subdomains_per_dimension;
	size_t level_of_subdomain_trees;

	size_t my_num_subdomains;
	size_t my_subdomain_id_start;
	size_t my_subdomain_id_end;

	Vec3d simulation_box_length;

	std::vector<Subdomain> subdomains;
	SpaceFillingCurve<Morton> space_curve;
};

#endif /* PARTITION_H */
