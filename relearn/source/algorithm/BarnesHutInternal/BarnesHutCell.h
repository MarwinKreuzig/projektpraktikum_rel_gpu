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

#include "algorithm/VirtualPlasticityElement.h"
#include "neurons/SignalType.h"

#include <optional>

/**
 * This class has all the informations necessary for the Barnes Hut algorithm
 * that need to be stored in a Cell. It does not perform any checks and should
 * not be used on its own, only as template argument for Cell.
 */
class BarnesHutCell {
public:
    using position_type = VirtualPlasticityElement::position_type;
    using counter_type = VirtualPlasticityElement::counter_type;

    /**
     * @brief Sets the number of free excitatory dendrites in this cell
     * @param num_dendrites The number of free excitatory dendrites
     */
    void set_number_excitatory_dendrites(const counter_type num_dendrites) noexcept {
        excitatory_dendrites.set_number_free_elements(num_dendrites);
    }

    /**
     * @brief Returns the number of free excitatory dendrites in this cell
     * @return The number of free excitatory dendrites
     */
    [[nodiscard]] counter_type get_number_excitatory_dendrites() const noexcept {
        return excitatory_dendrites.get_number_free_elements();
    }

    /**
     * @brief Sets the number of free inhibitory dendrites in this cell
     * @param num_dendrites The number of free inhibitory dendrites
     */
    void set_number_inhibitory_dendrites(const counter_type num_dendrites) noexcept {
        inhibitory_dendrites.set_number_free_elements(num_dendrites);
    }

    /**
     * @brief Returns the number of free inhibitory dendrites in this cell
     * @return The number of free inhibitory dendrites
     */
    [[nodiscard]] counter_type get_number_inhibitory_dendrites() const noexcept {
        return inhibitory_dendrites.get_number_free_elements();
    }

    /**
     * @brief Returns the number of free dendrites in this cell for the requested signal type
     * @param dendrite_type The requested signal type
     * @return The number of free dendrites
     */
    [[nodiscard]] counter_type get_number_dendrites_for(const SignalType dendrite_type) const noexcept {
        if (dendrite_type == SignalType::EXCITATORY) {
            return excitatory_dendrites.get_number_free_elements();
        }

        return inhibitory_dendrites.get_number_free_elements();
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param opt_position The new position of the inhibitory dendrite
     */
    void set_inhibitory_dendrites_position(const std::optional<position_type>& opt_position) noexcept {
        inhibitory_dendrites.set_position(opt_position);
    }

    /**
     * @brief Returns the position of the inhibitory dendrite, which can be empty
     * @return The position of the inhibitory dendrite
     */
    [[nodiscard]] std::optional<position_type> get_inhibitory_dendrites_position() const noexcept {
        return inhibitory_dendrites.get_position();
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param opt_position The new position of the excitatory dendrite
     */
    void set_excitatory_dendrites_position(const std::optional<position_type>& opt_position) noexcept {
        excitatory_dendrites.set_position(opt_position);
    }

    /**
     * @brief Returns the position of the excitatory dendrite, which can be empty
     * @return The position of the excitatory dendrite
     */
    [[nodiscard]] std::optional<position_type> get_excitatory_dendrites_position() const noexcept {
        return excitatory_dendrites.get_position();
    }

    /**
     * @brief Returns the position of the dendrite for the requested signal type, which can be empty
     * @param dendrite_type The requested signal type
     * @return The position of the dendrite
     */
    [[nodiscard]] std::optional<position_type> get_dendrites_position_for(const SignalType dendrite_type) const noexcept {
        if (dendrite_type == SignalType::EXCITATORY) {
            return excitatory_dendrites.get_position();
        }

        return inhibitory_dendrites.get_position();
    }

    /**
     * @brief Sets the position of the neuron for every necessary part of the cell
     * @param opt_position The position, can be empty
     */
    void set_neuron_position(const std::optional<position_type>& opt_position) noexcept {
        set_excitatory_dendrites_position(opt_position);
        set_inhibitory_dendrites_position(opt_position);
    }

    /**
     * @brief Returns the position of the cell, which can be empty
     * @exception Throws a RelearnException if one position is valid and the other is not, or if they are at different points
     * @return The position of the excitatory dendrite
     */
    [[nodiscard]] std::optional<position_type> get_neuron_position() const {
        const auto& excitatory_position_opt = get_excitatory_dendrites_position();
        const auto& inhibitory_position_opt = get_inhibitory_dendrites_position();

        const bool ex_valid = excitatory_position_opt.has_value();
        const bool in_valid = inhibitory_position_opt.has_value();

        if (!ex_valid && !in_valid) {
            return {};
        }

        if (ex_valid && in_valid) {
            const auto& pos_ex = excitatory_position_opt.value();
            const auto& pos_in = inhibitory_position_opt.value();

            const auto diff = pos_ex - pos_in;
            const bool exc_position_equals_inh_position = diff.get_x() == 0.0 && diff.get_y() == 0.0 && diff.get_z() == 0.0;
            RelearnException::check(exc_position_equals_inh_position, "BarnesHutCell::get_neuron_position: positions are unequal");

            return pos_ex;
        }

        RelearnException::fail("BarnesHutCell::get_neuron_position: one pos was valid and one was not");

        return {};
    }

private:
    VirtualPlasticityElement excitatory_dendrites{};
    VirtualPlasticityElement inhibitory_dendrites{};
};
