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

#include "VirtualPlasticityElement.h"
#include "neurons/SignalType.h"

#include <optional>

/**
 * This class has all the informations necessary for the Inverted Barnes Hut algorithm
 * that need to be stored in a Cell. It does not perform any checks and should
 * not be used on its own, only as template argument for Cell.
 */
class BarnesHutInvertedCell {
public:
    using position_type = VirtualPlasticityElement::position_type;
    using counter_type = VirtualPlasticityElement::counter_type;

    /**
     * @brief Sets the number of free excitatory axons in this cell
     * @param num_axons The number of free excitatory axons
     */
    void set_number_excitatory_axons(const counter_type num_axons) noexcept {
        excitatory_axons.set_number_free_elements(num_axons);
    }

    /**
     * @brief Returns the number of free excitatory axons in this cell
     * @return The number of free excitatory axons
     */
    [[nodiscard]] counter_type get_number_excitatory_axons() const noexcept {
        return excitatory_axons.get_number_free_elements();
    }

    /**
     * @brief Sets the number of free inhibitory axons in this cell
     * @param num_axons The number of free inhibitory axons
     */
    void set_number_inhibitory_axons(const counter_type num_axons) noexcept {
        inhibitory_axons.set_number_free_elements(num_axons);
    }

    /**
     * @brief Returns the number of free inhibitory axons in this cell
     * @return The number of free inhibitory axons
     */
    [[nodiscard]] counter_type get_number_inhibitory_axons() const noexcept {
        return inhibitory_axons.get_number_free_elements();
    }

    /**
     * @brief Returns the number of free axons in this cell for the requested signal type
     * @param axon_type The requested signal type
     * @return The number of free axons
     */
    [[nodiscard]] counter_type get_number_axons_for(const SignalType axon_type) const noexcept {
        if (axon_type == SignalType::EXCITATORY) {
            return excitatory_axons.get_number_free_elements();
        }

        return inhibitory_axons.get_number_free_elements();
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param opt_position The new position of the inhibitory axon
     */
    void set_inhibitory_axons_position(const std::optional<position_type>& opt_position) noexcept {
        inhibitory_axons.set_position(opt_position);
    }

    /**
     * @brief Returns the position of the inhibitory axon, which can be empty
     * @return The position of the inhibitory axon
     */
    [[nodiscard]] std::optional<position_type> get_inhibitory_axons_position() const noexcept {
        return inhibitory_axons.get_position();
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param opt_position The new position of the excitatory axon
     */
    void set_excitatory_axons_position(const std::optional<position_type>& opt_position) noexcept {
        excitatory_axons.set_position(opt_position);
    }

    /**
     * @brief Returns the position of the excitatory axon, which can be empty
     * @return The position of the excitatory axon
     */
    [[nodiscard]] std::optional<position_type> get_excitatory_axons_position() const noexcept {
        return excitatory_axons.get_position();
    }

    /**
     * @brief Returns the position of the axon for the requested signal type, which can be empty
     * @param axon_type The requested signal type
     * @return The position of the axon
     */
    [[nodiscard]] std::optional<position_type> get_axons_position_for(const SignalType axon_type) const noexcept {
        if (axon_type == SignalType::EXCITATORY) {
            return excitatory_axons.get_position();
        }

        return inhibitory_axons.get_position();
    }

    /**
     * @brief Sets the position of the neuron for every necessary part of the cell
     * @param opt_position The position, can be empty
     */
    void set_neuron_position(const std::optional<position_type>& opt_position) noexcept {
        set_excitatory_axons_position(opt_position);
        set_inhibitory_axons_position(opt_position);
    }

private:
    VirtualPlasticityElement excitatory_axons{};
    VirtualPlasticityElement inhibitory_axons{};
};
