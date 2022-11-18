#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"

#include <string>
#include <vector>
#include <set>

/**
 * An object of type NeuronsExtraInfo additional informations of neurons.
 * For a single neuron, these additional informations are: its x-, y-, and z- position and the name of the area the neuron is in.
 * It furthermore stores a map from the MPI rank to the (global) starting neuron id.
 * This is useful whenever one wants to print all neurons across multiple MPI ranks, while ommiting the MPI rank itself.
 */
class NeuronsExtraInfo {
public:
    using position_type = RelearnTypes::position_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Initializes a NeuronsExtraInfo that holds at most the given number of neurons.
     *      Must only be called once. Does not initialize dimensions or area names.
     * @param number_neurons The number of neurons, greater than 0
     * @exception Throws an RelearnAxception if number_neurons is 0 or if called multiple times.
     */
    void init(number_neurons_type number_neurons) {
        RelearnException::check(size == 0, "NeuronsExtraInfo::init: NeuronsExtraInfo initialized two times");
        size = number_neurons;
    }

    /**
     * @brief Inserts additional neurons with UNKNOWN area name and x-, y-, z- positions randomly picked from already existing ones. Requires only one MPI rank.
     * @param creation_count The number of new neuorns, greater than 0
     * @exception Throws an RelearnAxception if creation_count is 0, if x_dims, y_dims, or z_dims are empty, or if more than one MPI rank computes
     */
    void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Overwrites the current assignment of area ids to area names
     * @param names The new area names. names[i] is assigned to the area id i
     * @exception Throws an RelearnException if names.empty() or if the number of supplied elements does not match the number of stored neurons
     */
    void set_area_id_vs_area_name(std::vector<RelearnTypes::area_name> names) {
        RelearnException::check(!names.empty(), "NeuronsExtraInformation::set_area_id_vs_area_name: New area names are empty");
        area_id_vs_area_name = std::move(names);
    }
    /**
     * @brief Overwrites the current area ids with the supplied ones
     * @param ids The new area ids, must have the same size as neurons are stored. Neuron i is assigned to the area id ids[i]
     * @exception Throws an RelearnException if ids.empty() or if the number of supplied elements does not match the number of stored neurons
     */
    void set_neuron_id_vs_area_id(std::vector<RelearnTypes::area_id> ids) {
        RelearnException::check(!ids.empty(), "NeuronsExtraInformation::set_neuron_id_vs_area_id: New area ids are empty");
        RelearnException::check(size == ids.size(), "NeuronsExtraInformation::set_neuron_id_vs_area_id: Size does not match area ids count");
        RelearnException::check(!area_id_vs_area_name.empty(), "NeuronsExtraInfo::set_neuron_id_vs_area_id: Area id <-> names assignment must be called beforehand");

        neuron_id_vs_area_id = std::move(ids);
        for (const auto& neuron_id : NeuronID::range(size)) {
            const auto& area_id = neuron_id_vs_area_id[neuron_id.get_neuron_id()];
            RelearnException::check(area_id >= 0 && area_id < area_id_vs_area_name.size(), "NeuronsExtraInfo::set_neuron_id_vs_area_id: Invalid area id {}. Must be between 0 and {}", area_id, area_id_vs_area_name.size());
        }
    }

    /**
     * @brief Returns a vector that assigns each local neuron id an area id. Neuron i has area id get_neuron_id_vs_area_id()[i]
     * @return The currently stored area ids as a vector
     */
    [[nodiscard]] const std::vector<RelearnTypes::area_id>& get_neuron_id_vs_area_id() const noexcept {
        return neuron_id_vs_area_id;
    }

    /**
     * @brief Returns the mapping of area ids to area names. Area id i has area name get_area_id_vs_area_name()[i]
     * @return The currently stored area names as a vector
     */
    [[nodiscard]] const std::vector<RelearnTypes::area_name>& get_area_id_vs_area_name() const noexcept {
        return area_id_vs_area_name;
    }

    [[nodiscard]] const RelearnTypes::area_name get_area_name_for_neuron_id(const RelearnTypes::neuron_id neuron_id) const noexcept {
        return area_id_vs_area_name[neuron_id_vs_area_id[neuron_id]];
    }

    /**
     * @brief Number of neurons placed with a certain area name
     * @return Number of neurons currently stored under the given area name
     */
    [[nodiscard]] RelearnTypes::number_neurons_type get_nr_neurons_in_area(const RelearnTypes::area_id& area_id) const {
        const auto counted = std::count(neuron_id_vs_area_id.begin(), neuron_id_vs_area_id.end(), area_id);
        return static_cast<RelearnTypes::number_neurons_type>(counted);
    }

    /**
     * @brief Overwrites the current positions with the supplied ones
     * @param names The new positions, must have the same size as neurons are stored
     * @exception Throws an RelearnAxception if pos.empty() or if the number of supplied elements does not match the number of stored neurons
     */
    void set_positions(std::vector<position_type> pos) {
        RelearnException::check(!pos.empty(), "NeuronsExtraInformation::set_positions: New positions are empty");
        RelearnException::check(size == pos.size(), "NeuronsExtraInformation::set_positions: Size does not match area names count");
        positions = std::move(pos);
    }

    /**
     * @brief Returns the currently stored positions as a vector. The reference is invalidated whenever init or create_neurons is called.
     * @return The currently stored positions as a vector
     */
    [[nodiscard]] const std::vector<position_type>& get_positions() const noexcept {
        return positions;
    }

    /**
     * @brief Returns a position_type with the x-, y-, and z- positions for a specified neuron.
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     */
    [[nodiscard]] position_type get_position(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::get_position: neuron_id must be smaller than size but was {}", neuron_id);
        RelearnException::check(local_neuron_id < positions.size(), "NeuronsExtraInfo::get_position: neuron_id must be smaller than positions.size() but was {}", neuron_id);
        return positions[local_neuron_id];
    }

    /**
     * @brief Returns the area id for a specified neuron.
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     */
    [[nodiscard]] const RelearnTypes::area_id& get_area_id_for_neuron_id(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < neuron_id_vs_area_id.size(), "NeuronsExtraInfo::get_area_name: neuron_id must be smaller than size but was {}", neuron_id);
        return neuron_id_vs_area_id[local_neuron_id];
    }

private:
    number_neurons_type size{ 0 };

    std::vector<RelearnTypes::area_id> neuron_id_vs_area_id{};
    std::vector<RelearnTypes::area_name> area_id_vs_area_name{};
    std::vector<position_type> positions{};
};
