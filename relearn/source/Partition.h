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

#include "LogMessages.h"
#include "MPI_RMA_MemAllocator.h"
#include "Neurons.h"
#include "NeuronToSubdomainAssignment.h"
#include "Octree.h"
#include "Positions.h"
#include "Random.h"
#include "RelearnException.h"
#include "SpaceFillingCurve.h"
#include "SynapticElements.h"
#include "Vec3.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

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

	Partition(size_t num_ranks, size_t my_rank);

	~Partition() = default;

	Partition(const Partition& other) = delete;
	Partition(Partition&& other) = delete;

	Partition& operator=(const Partition& other) = delete;
	Partition& operator=(Partition&& other) = delete;

	void print_my_subdomains_info_rank(int rank);

	bool is_neuron_local(size_t neuron_id) const {
		RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
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

	size_t get_my_num_neurons() const {
		RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
		return my_num_neurons;
	}

	size_t get_my_num_subdomains() const noexcept {
		return my_num_subdomains;
	}

	void get_simulation_box_size(Vec3d& min, Vec3d& max) const {
		RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
		min = Vec3d(0);
		max = simulation_box_length;
	}

	Vec3d get_simulation_box_size() const {
		RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
		return simulation_box_length;
	}

	Octree& get_subdomain_tree(size_t subdomain_id) {
		RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
		RelearnException::check(subdomain_id < my_num_subdomains);

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

	size_t get_subdomain_id_from_pos(const Vec3d& pos) const {
		RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
		const Vec3d new_pos = pos / static_cast<double>(num_subdomains_per_dimension);
		const Vec3<size_t> id_3d = new_pos.floor_componentwise();
		const size_t id_1d = space_curve.map_3d_to_1d(id_3d);

		const size_t rank = id_1d / my_num_subdomains;

		return rank;
	}

	size_t get_global_id(size_t local_id) const {
		RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
		size_t counter = 0;
		for (size_t i = 0; i < subdomains.size(); i++) {
			const size_t old_counter = counter;

			counter += subdomains[i].global_neuron_ids.size();
			if (local_id < counter) {
				const size_t local_local_id = local_id - old_counter;
				return subdomains[i].global_neuron_ids[local_local_id];
			}
		}

		return local_id;
	}

	size_t get_local_id(size_t global_id) const {
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

protected:
	// We need the "axons" parameter to set for every neuron the type of axons it grows (exc./inh.)
	Neurons load_neurons(const Parameters& params, NeuronToSubdomainAssignment& neurons_in_subdomain);

	bool neurons_loaded;

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
