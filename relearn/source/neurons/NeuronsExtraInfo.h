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
#include "neurons/enums/UpdateStatus.h"
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
     *      Must only be called once. Sets up all neurons so that they update, but does not initialize the positions.
     * @param number_neurons The number of neurons, greater than 0
     * @exception Throws an RelearnException if number_neurons is 0 or if called multiple times.
     */
    void init(const number_neurons_type number_neurons) {
        RelearnException::check(number_neurons > 0, "NeuronsExtraInfo::init: number_neurons must be larger than 0.");
        RelearnException::check(size == 0, "NeuronsExtraInfo::init: NeuronsExtraInfo initialized two times, its size is already {}", size);
        
        size = number_neurons;
        update_status.resize(number_neurons, UpdateStatus::Enabled);
    }

    /**
     * @brief Inserts additional neurons with x-, y-, z- positions randomly picked from already existing ones.
     *      Sets all neurons to update. Only works with one MPI rank.
     * @param creation_count The number of new neurons, greater than 0
     * @exception Throws an RelearnException if creation_count is 0, if the positions are empty, or if more than one MPI rank is active
     */
    void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Marks the specified neurons as enabled
     * @param enabled_neurons The neuron ids from the now enabled neurons
     * @exception Throws a RelearnException if one of the specified ids exceeds the number of stored neurons
     */
    void set_enabled_neurons(const std::span<const NeuronID> enabled_neurons) {
        for (const auto& neuron_id : enabled_neurons) {
            const auto local_neuron_id = neuron_id.get_neuron_id();
            RelearnException::check(local_neuron_id < size, "NeuronsExtraInformation::set_enabled_neurons: NeuronID {} is too large: {}", neuron_id);

            update_status[local_neuron_id] = UpdateStatus::Enabled;
        }
    }

    /**
     * @brief Marks the specified neurons as disabled
     * @param disabled_neurons The neuron ids from the now disabled neurons
     * @exception Throws a RelearnException if one of the specified ids exceeds the number of stored neurons
     */
    void set_disabled_neurons(const std::span<const NeuronID> disabled_neurons) {
        for (const auto& neuron_id : disabled_neurons) {
            const auto local_neuron_id = neuron_id.get_neuron_id();
            RelearnException::check(local_neuron_id < size, "NeuronsExtraInformation::set_disabled_neurons: NeuronID {} is too large: {}", neuron_id);

            RelearnException::check(update_status[local_neuron_id] != UpdateStatus::Static, "NeuronsExtraInformation::set_disabled_neurons: Cannot disable a static neuron");
            update_status[local_neuron_id] = UpdateStatus::Disabled;
        }
    }

    /**
     * @brief Marks the specified neurons as static
     * @param static_neurons The neuron ids from the now static neurons
     * @exception Throws a RelearnException if one of the specified ids exceeds the number of stored neurons
     */
    void set_static_neurons(const std::span<const NeuronID> static_neurons) {
        for (const auto& neuron_id : static_neurons) {
            const auto local_neuron_id = neuron_id.get_neuron_id();
            RelearnException::check(local_neuron_id < size, "NeuronsExtraInformation::set_static_neurons: NeuronID {} is too large: {}", neuron_id);

            update_status[local_neuron_id] = UpdateStatus::Static;
        }
    }

    /**
     * @brief Overwrites the current positions with the supplied ones
     * @param names The new positions, must have the same size as neurons are stored
     * @exception Throws an RelearnException if pos.empty() or if the number of supplied elements does not match the number of stored neurons
     */
    void set_positions(std::vector<position_type> pos) {
        RelearnException::check(!pos.empty(), "NeuronsExtraInformation::set_positions: New positions are empty");
        RelearnException::check(size == pos.size(), "NeuronsExtraInformation::set_positions: Size does not match area names count");
        positions = std::move(pos);
    }

    /**
     * @brief Returns the currently stored positions as a vector
     * @return The currently stored positions
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

    /**
     * @brief Checks for a neuron if it updates its electrical activity
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     * @return True iff the neuron updates its electrical activity
     */
    [[nodiscard]] bool does_update_electrical_actvity(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::does_update_electrical_actvity: neuron_id must be smaller than size but was {}", neuron_id);
        RelearnException::check(local_neuron_id < update_status.size(), "NeuronsExtraInfo::does_update_electrical_actvity: neuron_id must be smaller than update_status.size() but was {}", neuron_id);

        return update_status[local_neuron_id] != UpdateStatus::Disabled;
    }

    /**
     * @brief Checks for a neuron if it updates its plasticity
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     * @return True iff the neuron updates its plasticity
     */
    [[nodiscard]] bool does_update_plasticity(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::does_update_plasticity: neuron_id must be smaller than size but was {}", neuron_id);
        RelearnException::check(local_neuron_id < update_status.size(), "NeuronsExtraInfo::does_update_plasticity: neuron_id must be smaller than update_status.size() but was {}", neuron_id);

        return update_status[local_neuron_id] == UpdateStatus::Enabled;
    }

    /**
     * @brief Returns the disable flags for the neurons
     * @return The disable flags
     */
    [[nodiscard]] const std::span<const UpdateStatus> get_disable_flags() const noexcept {
        return update_status;
    }

    [[nodiscard]] number_neurons_type get_size() const noexcept {
        return size;
    }

private:
    number_neurons_type size{ 0 };

    std::vector<position_type> positions{};
    std::vector<UpdateStatus> update_status{};
};
