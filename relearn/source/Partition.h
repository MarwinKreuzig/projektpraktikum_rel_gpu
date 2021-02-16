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

#include "Octree.h"
#include "RelearnException.h"
#include "SpaceFillingCurve.h"
#include "Vec3.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <tuple>
#include <vector>

class Neurons;
class NeuronModels;
class NeuronToSubdomainAssignment;
class SynapticElements;

class Partition {
public:
    struct Subdomain {
        Vec3d xyz_min;
        Vec3d xyz_max;

        size_t num_neurons{ Constants::uninitialized };

        // Local start and end neuron id
        size_t neuron_local_id_start{ Constants::uninitialized };
        size_t neuron_local_id_end{ Constants::uninitialized };

        std::vector<size_t> global_neuron_ids;

        size_t index_1d{ Constants::uninitialized };

        Vec3s index_3d;

        // The octree contains all neurons in
        // this subdomain. It is only used as a container
        // for the neurons
        Octree octree;
    };

    Partition(size_t num_ranks, size_t my_rank);

    ~Partition() = default;

    Partition(const Partition& other) = delete;
    Partition(Partition&& other) = default;

    Partition& operator=(const Partition& other) = delete;
    Partition& operator=(Partition&& other) = default;

    void print_my_subdomains_info_rank(int rank);

    [[nodiscard]] bool is_neuron_local(size_t neuron_id) const;

    [[nodiscard]] std::shared_ptr<Neurons> load_neurons(
        std::unique_ptr<NeuronToSubdomainAssignment> neurons_in_subdomain,
        std::unique_ptr<NeuronModels> neuron_models, 
        std::unique_ptr<SynapticElements> axons_ptr = nullptr, 
        std::unique_ptr<SynapticElements> dend_ex_ptr = nullptr,
        std::unique_ptr<SynapticElements> dend_in_ptr = nullptr);

    [[nodiscard]] size_t get_my_num_neurons() const {
        RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
        return my_num_neurons;
    }

    [[nodiscard]] size_t get_my_num_subdomains() const noexcept {
        return my_num_subdomains;
    }

    [[nodiscard]] std::tuple<Vec3d, Vec3d> get_simulation_box_size() const {
        RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
        Vec3d min{ 0 };
        Vec3d max{ simulation_box_length };

        return std::make_tuple(min, max);
    }

    [[nodiscard]] Octree& get_subdomain_tree(size_t subdomain_id) {
        RelearnException::check(neurons_loaded, "Neurons are not loaded yet");
        RelearnException::check(subdomain_id < my_num_subdomains, "Subdomain ID was too large");

        return subdomains[subdomain_id].octree;
    }

    [[nodiscard]] size_t get_my_subdomain_id_start() const noexcept {
        return my_subdomain_id_start;
    }

    [[nodiscard]] size_t get_my_subdomain_id_end() const noexcept {
        return my_subdomain_id_end;
    }

    [[nodiscard]] size_t get_level_of_subdomain_trees() const noexcept {
        return level_of_subdomain_trees;
    }

    [[nodiscard]] size_t get_total_num_subdomains() const noexcept {
        return total_num_subdomains;
    }

    [[nodiscard]] size_t get_num_subdomains_per_dimension() const noexcept {
        return num_subdomains_per_dimension;
    }

    [[nodiscard]] size_t get_subdomain_id_from_pos(const Vec3d& pos) const;

    [[nodiscard]] size_t get_global_id(size_t local_id) const;

    [[nodiscard]] size_t get_local_id(size_t global_id) const;

    [[nodiscard]] size_t get_total_num_neurons() const noexcept;

    void set_total_num_neurons(size_t total_num) noexcept;

private:
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
