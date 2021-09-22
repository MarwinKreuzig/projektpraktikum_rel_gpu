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

#include "../util/RelearnException.h"
#include "../util/Vec3.h"
#include "helper/RankNeuronId.h"

#include <optional>
#include <string>
#include <vector>

/**
  * An object of type NeuronsExtraInfo additional informations of neurons.
  * For a single neuron, these additional informations are: its x-, y-, and z- position and the name of the area the neuron is in.
  * It furthermore stores a map from the MPI rank to the (global) starting neuron id. 
  * This is useful whenever one wants to print all neurons across multiple MPI ranks, while ommiting the MPI rank itself.
  */
class NeuronsExtraInfo {
    size_t size{ 0 };

    std::vector<std::string> area_names{};
    std::vector<double> x_dims{};
    std::vector<double> y_dims{};
    std::vector<double> z_dims{};

    std::vector<size_t> mpi_rank_to_local_start_id{};

public:
    /**
     * @brief Initializes a NeuronsExtraInfo that holds at most the given number of neurons. Performs communication across all MPI ranks. 
     *      Must only be called once. Does not initialize dimensions or area names.
     * @param num_neurons The number of neurons, greater than 0
     * @exception Throws an RelearnAxception if number_neurons is 0 or if called multiple times.
     */
    void init(const size_t number_neurons);

    /**
     * @brief Inserts additional neurons with UNKNOWN area name and x-, y-, z- positions randomly picked from already existing ones. Requires only one MPI rank.
     * @param creation_count The number of new neuorns, greater than 0
     * @exception Throws an RelearnAxception if creation_count is 0, if x_dims, y_dims, or z_dims are empty, or if more than one MPI rank computes
     */
    void create_neurons(const size_t creation_count);

    /**
     * @brief Overwrites the current area names with the supplied ones
     * @param names The new area names, must have the same size as neurons are stored
     * @exception Throws an RelearnAxception if names.empty() or if the number of supplied elements does not match the number of stored neurons 
     */
    void set_area_names(std::vector<std::string> names) {
        RelearnException::check(!names.empty(), "NeuronsExtraInformation::set_area_names: New area names are empty");
        RelearnException::check(size == names.size(), "NeuronsExtraInformation::set_area_names: Size does not match area names count");
        area_names = std::move(names);
    }

    /**
     * @brief Overwrites the current x- positions with the supplied ones
     * @param dims The x- positions, must have the same size as neurons are stored
     * @exception Throws an RelearnAxception if dims.empty() or if the number of supplied elements does not match the number of stored neurons 
     */
    void set_x_dims(std::vector<double> dims) {
        RelearnException::check(!dims.empty(), "NeuronsExtraInformation::set_x_dims: New x dimensions are empty");
        RelearnException::check(size == dims.size(), "NeuronsExtraInformation::set_x_dims: Size does not match area names count");
        x_dims = std::move(dims);
    }

    /**
     * @brief Overwrites the current y- positions with the supplied ones
     * @param dims The y- positions, must have the same size as neurons are stored
     * @exception Throws an RelearnAxception if dims.empty() or if the number of supplied elements does not match the number of stored neurons 
     */
    void set_y_dims(std::vector<double> dims) {
        RelearnException::check(!dims.empty(), "NeuronsExtraInformation::set_y_dims: New y dimensions are empty");
        RelearnException::check(size == dims.size(), "NeuronsExtraInformation::set_y_dims: Size does not match area names count");
        y_dims = std::move(dims);
    }

    /**
     * @brief Overwrites the current z- positions with the supplied ones
     * @param dims The z- positions, must have the same size as neurons are stored
     * @exception Throws an RelearnAxception if dims.empty() or if the number of supplied elements does not match the number of stored neurons 
     */
    void set_z_dims(std::vector<double> dims) {
        RelearnException::check(!dims.empty(), "NeuronsExtraInformation::set_z_dims: New z dimensions are empty");
        RelearnException::check(size == dims.size(), "NeuronsExtraInformation::set_z_dims: Size does not match area names count");
        z_dims = std::move(dims);
    }

    /**
     * @brief Returns the currently stored x- positions as a vector. The reference is invalidated whenever init or create_neurons is called.
     */
    [[nodiscard]] const std::vector<double>& get_x_dims() const noexcept {
        return x_dims;
    }

    /**
     * @brief Returns the currently stored y- positions as a vector. The reference is invalidated whenever init or create_neurons is called.
     */
    [[nodiscard]] const std::vector<double>& get_y_dims() const noexcept {
        return y_dims;
    }

    /**
     * @brief Returns the currently stored z- positions as a vector. The reference is invalidated whenever init or create_neurons is called.
     */
    [[nodiscard]] const std::vector<double>& get_z_dims() const noexcept {
        return z_dims;
    }

    /**
     * @brief Returns the currently stored area names as a vector. The reference is invalidated whenever init or create_neurons is called.
     */
    [[nodiscard]] const std::vector<std::string>& get_area_names() const noexcept {
        return area_names;
    }

    /**
     * @brief Returns a Vec3d with the x-, y-, and z- positions for a specified neuron.
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnAxception if the specified id exceeds the number of stored neurons
     */
    [[nodiscard]] Vec3d get_position(const size_t neuron_id) const {
        RelearnException::check(neuron_id < size, "NeuronsExtraInfo::get_position: neuron_id must be smaller than size but was {}", neuron_id);
        RelearnException::check(neuron_id < x_dims.size(), "NeuronsExtraInfo::get_position: neuron_id must be smaller than x_dims.size() but was {}", neuron_id);
        RelearnException::check(neuron_id < y_dims.size(), "NeuronsExtraInfo::get_position: neuron_id must be smaller than y_dims.size() but was {}", neuron_id);
        RelearnException::check(neuron_id < z_dims.size(), "NeuronsExtraInfo::get_position: neuron_id must be smaller than z_dims.size() but was {}", neuron_id);
        return Vec3d{ x_dims[neuron_id], y_dims[neuron_id], z_dims[neuron_id] };
    }

    /**
     * @brief Returns the x- position for a specified neuron.
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnAxception if the specified id exceeds the number of stored neurons
     */
    [[nodiscard]] double get_x(const size_t neuron_id) const {
        RelearnException::check(neuron_id < x_dims.size(), "NeuronsExtraInfo::get_x: neuron_id must be smaller than size but was {}", neuron_id);
        return x_dims[neuron_id];
    }

    /**
     * @brief Returns the y- position for a specified neuron.
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnAxception if the specified id exceeds the number of stored neurons
     */
    [[nodiscard]] double get_y(const size_t neuron_id) const {
        RelearnException::check(neuron_id < y_dims.size(), "NeuronsExtraInfo::get_y: neuron_id must be smaller than size but was {}", neuron_id);
        return y_dims[neuron_id];
    }

    /**
     * @brief Returns the z- positions for a specified neuron.
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnAxception if the specified id exceeds the number of stored neurons
     */
    [[nodiscard]] double get_z(const size_t neuron_id) const {
        RelearnException::check(neuron_id < z_dims.size(), "NeuronsExtraInfo::get_z: neuron_id must be smaller than size but was {}", neuron_id);
        return z_dims[neuron_id];
    }

    /**
     * @brief Returns the area name for a specified neuron.
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnAxception if the specified id exceeds the number of stored neurons
     */
    [[nodiscard]] const std::string& get_area_name(const size_t neuron_id) const {
        RelearnException::check(neuron_id < area_names.size(), "NeuronsExtraInfo::get_area_name: neuron_id must be smaller than size but was {}", neuron_id);
        return area_names[neuron_id];
    }

    /**
     * @brief Returns the global neuron id for a specified pair of MPI rank and local neuron id
     * @param rank_neuron_id The specified MPI rank and local neuron id
     * @exception Throws an RelearnAxception the MPI rank is smaller than 0 or exceeds the number of stored MPI ranks, 
     *      if the local neuron id is unitialized or if the translated neuron id exceeds the number of local neurons for the specified rank 
     */
    [[nodiscard]] size_t rank_neuron_id2glob_id(const RankNeuronId& rank_neuron_id) {
        const auto requested_rank = rank_neuron_id.get_rank();
        const auto requested_local_neuron_id = rank_neuron_id.get_neuron_id();

        RelearnException::check(requested_rank >= 0, "NeuronsExtraInfo::rank_neuron_id2glob_id: There was a negative MPI rank");
        RelearnException::check(requested_rank < mpi_rank_to_local_start_id.size(), "NeuronsExtraInfo::rank_neuron_id2glob_id: The requested MPI rank is not stored");
        RelearnException::check(requested_local_neuron_id < Constants::uninitialized, "NeuronsExtraInfo::rank_neuron_id2glob_id: The requested neuron id is unitialized");

        const auto glob_id = mpi_rank_to_local_start_id[requested_rank] + requested_local_neuron_id;

        if (rank_neuron_id.get_rank() < mpi_rank_to_local_start_id.size() - 1) {
            RelearnException::check(glob_id < mpi_rank_to_local_start_id[requested_rank + 1], "NeuronsExtraInfo::rank_neuron_id2glob_id: The translated id exceeded the starting id of the next rank");
        }

        return glob_id;
    }
};
