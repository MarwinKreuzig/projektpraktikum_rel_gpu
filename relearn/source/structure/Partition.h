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
#include "../structure/SpaceFillingCurve.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"

#include <functional>
#include <tuple>
#include <vector>

/**
 * This class provides all kinds of functionality that deals with the the local portion of neurons on the current MPI rank.
 * The local neurons are divided into Subdomains, from which each MPI rank has 1, 2, or 4
 */
class Partition {
public:
    using position_type = RelearnTypes::position_type;
    using box_size_type = RelearnTypes::box_size_type;

    /**
     * Subdomain is a type that represents one part of the octree at the level of the branching nodes.
     * It's composed of the min and max positions of the subdomain, the number of neurons in this subdomain,
     * the start and end local neuron ids, and its 1d and 3d index for all Subdomains.
     */
    struct Subdomain {
        box_size_type minimum_position{ Constants::uninitialized };
        box_size_type maximum_position{ Constants::uninitialized };

        size_t number_neurons{ Constants::uninitialized };

        size_t neuron_local_id_start{ Constants::uninitialized };
        size_t neuron_local_id_end{ Constants::uninitialized };

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
     * @brief Prints the current local_subdomains as messages on the rank
     * @param rank The rank that should print the local_subdomains
     */
    void print_my_subdomains_info_rank(int rank);

    /**
     * @brief Sets the total number of neurons
     * @param total_num The total number of neurons
     */
    void set_total_number_neurons(const size_t total_num) noexcept {
        total_number_neurons = total_num;
    }

    /**
     * @brief Returns the total number of neurons
     * @exception Throws a RelearnException if the number has not been set previously
     * @return The total number of neurons
     */
    [[nodiscard]] size_t get_total_number_neurons() const {
        RelearnException::check(total_number_neurons < Constants::uninitialized, "Partition::get_total_number_neurons: total_number_neurons was not set");
        return total_number_neurons;
    }

    /**
     * @brief Returns the number of local neurons
     * @exception Throws a RelearnException if the calculate_local_ids has not been called
     * @return The number of local neurons
     */
    [[nodiscard]] size_t get_number_local_neurons() const {
        RelearnException::check(number_local_neurons < Constants::uninitialized, "Partition::get_number_local_neurons: Neurons are not loaded yet");
        return number_local_neurons;
    }

    /**
     * @brief Returns the total number of local_subdomains
     * @return The total number of local_subdomains
     */
    [[nodiscard]] size_t get_total_number_subdomains() const noexcept {
        return total_number_subdomains;
    }

    /**
     * @brief Returns the number of local_subdomains per dimension (total number of local_subdomains)^(1/3)
     * @return The number of local_subdomains per dimension
     */
    [[nodiscard]] size_t get_number_subdomains_per_dimension() const noexcept {
        return number_subdomains_per_dimension;
    }

    /**
     * @brief Returns the level in the octree on which the local_subdomains start
     * @return The level in the octree on which the local_subdomains start
     */
    [[nodiscard]] size_t get_level_of_subdomain_trees() const noexcept {
        return level_of_subdomain_trees;
    }

    /**
     * @brief Returns the number of local subdomains
     * @return The number of local subdomains (1, 2, or 4)
     */
    [[nodiscard]] size_t get_number_local_subdomains() const noexcept {
        return number_local_subdomains;
    }

    /**
     * @brief Returns the first id of the local subdomains in the global setting
     * @return The first id of the local local_subdomains in the global setting
     */
    [[nodiscard]] size_t get_local_subdomain_id_start() const noexcept {
        return local_subdomain_id_start;
    }

    /**
     * @brief Returns the last id of the local subdomains in the global setting
     * @return The last id of the local local_subdomains in the global setting
     */
    [[nodiscard]] size_t get_local_subdomain_id_end() const noexcept {
        return local_subdomain_id_end;
    }

    /**
     * @brief Returns the mpi rank that is responsible for the position
     * @param pos The position which shall be resolved
     * @exception Throws a RelearnException if the calculate_local_ids has not been called
     * @return Returns the MPI rank that is responsible for the position
     */
    [[nodiscard]] size_t get_mpi_rank_from_position(const position_type& position) const {
        RelearnException::check(simulation_box_length.get_x() < Constants::uninitialized / 2, "Partition::get_mpi_rank_from_position: Neurons are not loaded yet");
        const box_size_type subdomain_length = simulation_box_length / static_cast<double>(number_subdomains_per_dimension);

        const box_size_type subdomain_3d{ position.get_x() / subdomain_length.get_x(), position.get_y() / subdomain_length.get_y(), position.get_z() / subdomain_length.get_z() };
        const Vec3s id_3d = subdomain_3d.floor_componentwise();
        const size_t id_1d = space_curve.map_3d_to_1d(id_3d);

        const size_t rank = id_1d / number_local_subdomains;

        return rank;
    }

    /**
     * @brief Returns the flattened index of the subdomain in the global setting
     * @param local_subdomain_index The local subdomain index
     * @exception Throws a RelearnException if local_subdomain_index is larger or equal to the number of local subdomains
     * @return The flattened index of the subdomain in the local index
     */
    [[nodiscard]] size_t get_1d_index_of_subdomain(const size_t local_subdomain_index) const {
        RelearnException::check(local_subdomain_index < local_subdomains.size(),
            "Partition::get_1d_index_of_subdomain: index ({}) was too large for the number of local subdomains ({})", local_subdomain_index, local_subdomains.size());
        return local_subdomains[local_subdomain_index].index_1d;
    }

    /**
     * @brief Returns the 3-dimensional index of the subdomain in the global setting
     * @param local_subdomain_index The local subdomain index
     * @exception Throws a RelearnException if local_subdomain_index is larger or equal to the number of local subdomains
     * @return The 3-dimensional of the subdomain in the global setting
     */
    [[nodiscard]] Vec3s get_3d_index_of_subdomain(const size_t local_subdomain_index) const {
        RelearnException::check(local_subdomain_index < local_subdomains.size(),
            "Partition::get_3d_index_of_subdomain: index ({}) was too large for the number of local subdomains ({})", local_subdomain_index, local_subdomains.size());
        return local_subdomains[local_subdomain_index].index_3d;
    }

    /**
     * @brief Returns the first local neuron id of the subdomain
     * @param local_subdomain_index The local subdomain index
     * @exception Throws a RelearnException if local_subdomain_index is larger or equal to the number of local subdomains
     * @return The first local neuron id of the subdomain
     */
    [[nodiscard]] size_t get_local_subdomain_local_neuron_id_start(const size_t local_subdomain_index) const {
        RelearnException::check(local_subdomain_index < local_subdomains.size(),
            "Partition::get_local_subdomain_local_neuron_id_start: index ({}) was too large for the number of local subdomains ({})", local_subdomain_index, local_subdomains.size());
        return local_subdomains[local_subdomain_index].neuron_local_id_start;
    }

    /**
     * @brief Sets the number of neurons in all subdomains, and updates the dependent values
     * @param number_local_neurons_in_subdomains The number of neurons in each local subdomain
     * @exception Throws a RelearnException if number_local_neurons_in_subdomains has a size unequal to the number of local subdomains
     */
    void set_subdomain_number_neurons(const std::vector<size_t>& number_local_neurons_in_subdomains) {
        RelearnException::check(number_local_neurons_in_subdomains.size() == local_subdomains.size(),
            "Partition::set_subdomain_number_neurons: number_local_neurons_in_subdomains had a different size ({}) then the number of local subdomains ({})", number_local_neurons_in_subdomains.size(), local_subdomains.size());

        number_local_neurons = 0;
        for (auto subdomain_index = 0; subdomain_index < number_local_neurons_in_subdomains.size(); subdomain_index++) {
            Subdomain& current_subdomain = local_subdomains[subdomain_index];

            current_subdomain.number_neurons = number_local_neurons_in_subdomains[subdomain_index];

            // Add subdomain's number of neurons to rank's number of neurons
            number_local_neurons += current_subdomain.number_neurons;

            // Set start and end of local neuron ids
            // 0-th subdomain starts with neuron id 0
            current_subdomain.neuron_local_id_start = (subdomain_index == 0) ? 0 : (local_subdomains[subdomain_index - 1].neuron_local_id_end + 1);
            current_subdomain.neuron_local_id_end = current_subdomain.neuron_local_id_start + current_subdomain.number_neurons - 1;
        }
    }

    /**
     * @brief Sets the boundaries of the subdomain
     * @param local_subdomain_index The local subdomain index
     * @param min The smallest position in the subdomain
     * @param max The largest position in the subdomain
     * @exception Throws a RelearnException if local_subdomain_index is larger or equal to the number of local subdomains
     */
    void set_subdomain_boundaries(const size_t local_subdomain_index, const Vec3d& min, const Vec3d& max) {
        RelearnException::check(local_subdomain_index < local_subdomains.size(),
            "Partition::set_subdomain_boundaries: index ({}) was too large for the number of local subdomains ({})", local_subdomain_index, local_subdomains.size());
        local_subdomains[local_subdomain_index].minimum_position = min;
        local_subdomains[local_subdomain_index].maximum_position = max;
    }

    /**
     * @brief Sets the boundaries of the simulation box
     * @param min The smallest position in the simulation box
     * @param max The largest position in the simulation box
     */
    void set_simulation_box_size(const Vec3d& min, const Vec3d& max);

    /**
     * @brief Returns the boundaries of the subdomain
     * @param subdomain_index_1d The flattened index of the subdomain
     * @return (minimum, maximum) of the subdomain
     */
    std::pair<Vec3d, Vec3d> get_subdomain_boundaries(const size_t subdomain_index_1d) {
        return get_subdomain_boundaries(space_curve.map_1d_to_3d(subdomain_index_1d));
    }

    /**
     * @brief Returns the boundaries of the subdomain
     * @param subdomain_index_3d The 3-dimensional index of the subdomain
     * @return (minimum, maximum) of the subdomain
     */
    std::pair<Vec3d, Vec3d> get_subdomain_boundaries(const Vec3s& subdomain_index_3d) {
        // auto f = std::bind(&Foo::print_sum, &foo, 95, _1);
        const auto requested_subdomain_x = subdomain_index_3d.get_x();
        const auto requested_subdomain_y = subdomain_index_3d.get_y();
        const auto requested_subdomain_z = subdomain_index_3d.get_z();

        const auto& [sim_box_min, sim_box_max] = get_simulation_box_size();
        const auto& simulation_box_length = (sim_box_max - sim_box_min);

        const auto& subdomain_length = simulation_box_length / number_subdomains_per_dimension;

        const auto subdomain_x_length = subdomain_length.get_x();
        const auto subdomain_y_length = subdomain_length.get_y();
        const auto subdomain_z_length = subdomain_length.get_z();

        box_size_type min{
            requested_subdomain_x * subdomain_x_length,
            requested_subdomain_y * subdomain_y_length,
            requested_subdomain_z * subdomain_z_length
        };

        const auto next_x = static_cast<box_size_type::value_type>(requested_subdomain_x + 1) * subdomain_x_length;
        const auto next_y = static_cast<box_size_type::value_type>(requested_subdomain_y + 1) * subdomain_y_length;
        const auto next_z = static_cast<box_size_type::value_type>(requested_subdomain_z + 1) * subdomain_z_length;

        box_size_type max{ next_x, next_y, next_z };

        auto currected_min = boundary_corrector(min);
        auto currected_max = boundary_corrector(max);

        return std::make_pair(currected_min, currected_max);
    }

    /**
     * @brief Returns the size of the simulation box
     * @exception Throws a RelearnException if set_simulation_box_size was not called before
     * @return The size of the simulation box as tuple (min, max)
     */
    [[nodiscard]] std::tuple<box_size_type, box_size_type> get_simulation_box_size() const {
        RelearnException::check(simulation_box_length.get_x() < Constants::uninitialized / 2, "Partition::get_simulation_box_size: set_simulation_box_size was not called before");
        box_size_type min{ 0 };
        box_size_type max{ simulation_box_length };

        return std::make_tuple(min, max);
    }

    /**
     * @brief Sets the correction function for the boundaries of the subdomains
     * @param corrector The correction function
     * @exception Throws a RelearnException if corrector is not valid
     */
    void set_boundary_correction_function(std::function<box_size_type(box_size_type)> corrector) {
        RelearnException::check(corrector.operator bool(), "Partition::set_boundary_correction_function: corrector was empty");
        boundary_corrector = std::move(corrector);
    }

private:
    size_t total_number_neurons{ Constants::uninitialized };
    size_t number_local_neurons{ Constants::uninitialized };

    size_t total_number_subdomains{ Constants::uninitialized };
    size_t number_subdomains_per_dimension{ Constants::uninitialized };
    size_t level_of_subdomain_trees{ Constants::uninitialized };

    size_t number_local_subdomains{ Constants::uninitialized };
    size_t local_subdomain_id_start{ Constants::uninitialized };
    size_t local_subdomain_id_end{ Constants::uninitialized };

    box_size_type simulation_box_length{ Constants::uninitialized };

    std::vector<Subdomain> local_subdomains{};
    SpaceFillingCurve<Morton> space_curve{};

    std::function<box_size_type(box_size_type)> boundary_corrector{
        [](box_size_type bst) { return bst; }
    };
};
