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
		size_t neuron_local_id_start;
		size_t neuron_local_id_end;

		std::vector<size_t> global_neuron_ids;

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
		assert(false && "Don't use this function any more! set_mpi_rma_mem_allocator in partition.h");
	}

	bool is_neuron_local(size_t neuron_id) const noexcept {
		for (const Subdomain& subdomain : subdomains) {
			const bool found = std::binary_search(subdomain.global_neuron_ids.begin(), subdomain.global_neuron_ids.end(), neuron_id);
			if (found) {
				return true;
			}
		}

		return false;
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

	//size_t get_num_subdomains_per_axis() const noexcept {
	//	const double subdomains_d = static_cast<double>(total_num_subdomains);
	//	const double third_root = std::pow(subdomains_d, (1.0 / 3.0));
	//	const size_t subdomains_per_axis = static_cast<size_t>(std::round(third_root));
	//	return subdomains_per_axis;
	//}

	void get_simulation_box_size(Vec3d& min, Vec3d& max) const noexcept {
		min = Vec3d(0);
		max = simulation_box_length;
	}

	Vec3d get_simulation_box_size() const noexcept {
		return simulation_box_length;
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

	size_t get_subdomain_id_from_pos(const Vec3d& pos) const noexcept {
		Vec3d new_pos = pos / num_subdomains_per_dimension;
		Vec3<size_t> id_3d = new_pos.floor_componentwise();
		size_t id_1d = space_curve.map_3d_to_1d(id_3d);

		size_t rank = id_1d / my_num_subdomains;

		return rank;
	}

	size_t get_global_id(size_t local_id) const noexcept {
		size_t counter = 0;
		for (size_t i = 0; i < subdomains.size(); i++) {
			size_t old_counter = counter;

			counter += subdomains[i].global_neuron_ids.size();
			if (local_id < counter) {
				size_t local_local_id = local_id - old_counter;
				return subdomains[i].global_neuron_ids[local_local_id];
			}
		}

		assert(false && "Didn't find local id in Partition.h");
	}

	size_t get_local_id(size_t global_id) const noexcept {
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

		assert(false && "Didn't find global id in Partition.h");

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
