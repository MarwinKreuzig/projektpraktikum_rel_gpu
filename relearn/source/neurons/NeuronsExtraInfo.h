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

#include <span>
#include <string>
#include <vector>

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
    void init(const number_neurons_type number_neurons) {
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
    [[nodiscard]] std::span<const position_type> get_positions() const noexcept {
        return positions;
    }

    /**
     * @brief Returns a position_type with the x-, y-, and z- positions for a specified neuron.
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     */
    [[nodiscard]] position_type get_position(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::get_position: neuron_id must be smaller than size but was {}", neuron_id);
        RelearnException::check(local_neuron_id < positions.size(), "NeuronsExtraInfo::get_position: neuron_id must be smaller than positions.size() but was {}", neuron_id);
        return positions[local_neuron_id];
    }

private:
    number_neurons_type size{ 0 };

    std::vector<position_type> positions{};
};
