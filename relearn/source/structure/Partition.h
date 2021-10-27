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

#include "../Config.h"
#include "../algorithm/BarnesHutCell.h"
#include "../mpi/MPIWrapper.h"
#include "../neurons/models/NeuronModels.h"
#include "../structure/SpaceFillingCurve.h"
#include "../util/Vec3.h"

#include <memory>
#include <tuple>
#include <vector>

class Neurons;
class NeuronToSubdomainAssignment;
template <typename T>
class OctreeNode;

/**
 * This class provides all kinds of functionality that deals with the the local portion of neurons on the current MPI rank.
 * It constructs OctreeNode* via the MPIWrapper and deletes those again.
 * The local neurons are divided into Subdomains, from which each MPI rank has 1, 2, or 4
 */
class Partition {
public:
    using position_type = RelearnTypes::position_type;
    using box_size_type = RelearnTypes::box_size_type;

    /**
     * Subdomain is a type that represents one part of the octree at the level of the branching nodes.
     * It's composed of the min and max positions of the subdomain, the number of neurons in this subdomain,
     * the start and end local neuron ids, all global neuron ids for the local neurons, 
     * it's 1d and 3d index for all Subdomains and the positions of the neurons in that Subdomain.
     */
    struct Subdomain {
        box_size_type xyz_min{ Constants::uninitialized };
        box_size_type xyz_max{ Constants::uninitialized };

        size_t num_neurons{ Constants::uninitialized };

        size_t neuron_local_id_start{ Constants::uninitialized };
        size_t neuron_local_id_end{ Constants::uninitialized };

        std::vector<size_t> global_neuron_ids{};

        std::vector<position_type> local_positions{};

        size_t index_1d{ Constants::uninitialized };

        Vec3s index_3d{ Constants::uninitialized };
    };

    /**
     * @brief Constructs a new object and uses the number of MPI ranks and the current MPI rank as foundation for the calculations
     * @param num_ranks The number of MPI ranks
     * @param my_rank The current MPI rank
     * @exception Throws a RelearnException if 0 <= my_rank < num_ranks is violated, if the number of MPI ranks is not of the form 2^k
     */
    Partition(size_t num_ranks, size_t my_rank);

    ~Partition() = default;

    Partition(const Partition& other) = delete;
    Partition(Partition&& other) = default;

    Partition& operator=(const Partition& other) = delete;
    Partition& operator=(Partition&& other) = default;

    /**
     * @brief Prints the current subdomains as messages on the rank
     * @param rank The rank that should print the subdomains
     */
    void print_my_subdomains_info_rank(int rank);

    /**
     * @brief Checks if the neuron id is local to the current MPI rank
     * @param neuron_id The neuron id for which it should be determined if it's local on the current MPI rank
     * @return True iff the neuron id is local
     */
    [[nodiscard]] bool is_neuron_local(size_t neuron_id) const;

    /**
     * @brief Loads the local neurons from neurons_in_subdomain into neurons
     * @param neurons The neurons that will be filled
     * @param neurons_in_subdomain The class that provides the neuron placements
     * @exception Throws a RelearnException of the neurons have already been loaded
     */
    void load_data_from_subdomain_assignment(const std::shared_ptr<Neurons>& neurons, std::unique_ptr<NeuronToSubdomainAssignment>&& neurons_in_subdomain);

    /**
     * @brief Returns the number of local neurons
     * @exception Throws a RelearnException if the load_data_from_subdomain_assignment has not been called
     * @return The number of local neurons
     */
    [[nodiscard]] size_t get_my_num_neurons() const {
        RelearnException::check(neurons_loaded, "Partition::get_my_num_neurons: Neurons are not loaded yet");
        return my_num_neurons;
    }

    /**
     * @brief Returns the number of local subdomains
     * @exception Throws a RelearnException if the load_data_from_subdomain_assignment has not been called
     * @return The number of local subdomains (1, 2, or 4)
     */
    [[nodiscard]] size_t get_my_num_subdomains() const noexcept {
        return my_num_subdomains;
    }

    /**
     * @brief Returns the size of the simulation box
     * @exception Throws a RelearnException if the load_data_from_subdomain_assignment has not been called
     * @return The size of the simulation box as tuple (min, max)
     */
    [[nodiscard]] std::tuple<box_size_type, box_size_type> get_simulation_box_size() const {
        RelearnException::check(neurons_loaded, "Partition::get_simulation_box_size: Neurons are not loaded yet");
        box_size_type min{ 0 };
        box_size_type max{ simulation_box_length };

        return std::make_tuple(min, max);
    }

    /**
     * @brief Returns the first id of the local subdomains in the global setting (don't use that in combination with get_subdomain_tree)
     * @return The first id of the local subdomains in the global setting
     */
    [[nodiscard]] size_t get_my_subdomain_id_start() const noexcept {
        return my_subdomain_id_start;
    }

    /**
     * @brief Returns the last id of the local subdomains in the global setting (don't use that in combination with get_subdomain_tree)
     * @return The last id of the local subdomains in the global setting
     */
    [[nodiscard]] size_t get_my_subdomain_id_end() const noexcept {
        return my_subdomain_id_end;
    }

    /**
     * @brief Returns the level in the octree on which the subdomains start
     * @return The level in the octree on which the subdomains start
     */
    [[nodiscard]] size_t get_level_of_subdomain_trees() const noexcept {
        return level_of_subdomain_trees;
    }

    /**
     * @brief Returns the total number of subdomains
     * @return The total number of subdomains
     */
    [[nodiscard]] size_t get_total_num_subdomains() const noexcept {
        return total_num_subdomains;
    }

    /**
     * @brief Returns the number of subdomains per dimension (total number of subdomains)^(1/3)
     * @return The number of subdomains per dimension
     */
    [[nodiscard]] size_t get_num_subdomains_per_dimension() const noexcept {
        return num_subdomains_per_dimension;
    }

    /**
     * @brief Returns the mpi rank that is responsible for the position
     * @param pos The position which shall be resolved
     * @exception Throws a RelearnException if the load_data_from_subdomain_assignment has not been called
     * @return Returns the MPI rank that is responsible for the position
     */
    [[nodiscard]] size_t get_mpi_rank_from_pos(const position_type& pos) const;

    /**
     * @brief Translates a local neuron id to a global neuron id by prefix-summing the local neuron ids over all MPI ranks
     * @param local_id The local neuron id that should be translated
     * @exception Throws a RelearnException if the load_data_from_subdomain_assignment has not been called
     * @return Returns the global neuron id
     */
    [[nodiscard]] size_t get_global_id(size_t local_id) const;

    /**
     * @brief Translates a global neuron id to a local neuron id
     * @param local_id The global neuron id that should be translated
     * @exception Throws a RelearnException if the load_data_from_subdomain_assignment has not been called
     * @return Returns the local neuron id
     */
    [[nodiscard]] size_t get_local_id(size_t global_id) const;

    /**
     * @brief Returns the total number of neurons
     * @exception Throws a RelearnException if the load_data_from_subdomain_assignment has not been called
     * @return The total number of neurons
     */
    [[nodiscard]] size_t get_total_num_neurons() const noexcept {
        return total_num_neurons;
    }

    /**
     * @brief Translates a local subdomain id to the global subdomain id 
     * @param subdomain_id The local subdomain id that should be translated
     * @exception Throws a RelearnException if subdomain_id is >= number of local subdomains
     * @return Returns the global subdomain id
     */
    [[nodiscard]] size_t get_1d_index_for_local_subdomain(const size_t subdomain_id) const {
        RelearnException::check(subdomain_id < my_num_subdomains, "Partition::get_1d_index_for_local_subdomain: Subdomain ID was too large: {} vs {}", subdomain_id, subdomains.size());
        return subdomains[subdomain_id].index_1d;
    }

    /**
     * @brief Sets the total number of neurons
     * @param total_num The total number of neurons
     */
    void set_total_num_neurons(const size_t total_num) noexcept {
        total_num_neurons = total_num;
    }

    /**
     * @brief Inserts all neurons for a given subdomain id into the octree node. 
     * @tparam AdditionalCellAttributes The additional cell attributes for the octree node
     * @param local_root The octree node where all neurons shall be inserted
     * @param subdomain_id The local subdomain id which's neurons shall be inserted
     * @exception Throws a RelearnException if local_root is nullptr, subdomain_id is larger than the stored subdomains,
     *      or there was a memory error when creating the octree nodes
     */
    template <typename AdditionalCellAttributes>
    void insert_nodes_into(OctreeNode<AdditionalCellAttributes>* local_root, const size_t subdomain_id) const {
        RelearnException::check(local_root != nullptr, "Partition::insert_nodes_into: local_root was nullptr");
        RelearnException::check(subdomain_id < subdomains.size(), "Partition::insert_nodes_into: subdomain_id was too large: {} vs {}", subdomain_id, subdomains.size());

        const auto& current_subdomain = subdomains[subdomain_id];
        auto neuron_id = current_subdomain.neuron_local_id_start;

        const auto my_rank = MPIWrapper::get_my_rank();
        for (size_t j = 0; j < current_subdomain.num_neurons; j++) {
            // Insert neuron into tree
            auto* const node = local_root->insert(current_subdomain.local_positions[j], neuron_id, my_rank);
            RelearnException::check(node != nullptr, "node is nullptr");

            neuron_id++;
        }
    }

private:
    bool neurons_loaded{ false };

    size_t total_num_neurons{ Constants::uninitialized };
    size_t my_num_neurons{ Constants::uninitialized };

    size_t total_num_subdomains{ Constants::uninitialized };
    size_t num_subdomains_per_dimension{ Constants::uninitialized };
    size_t level_of_subdomain_trees{ Constants::uninitialized };

    size_t my_num_subdomains{ Constants::uninitialized };
    size_t my_subdomain_id_start{ Constants::uninitialized };
    size_t my_subdomain_id_end{ Constants::uninitialized };

    box_size_type simulation_box_length{ Constants::uninitialized };

    std::vector<Subdomain> subdomains{};
    SpaceFillingCurve<Morton> space_curve{};
};
