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
#include "neurons/ElementType.h"
#include "neurons/SignalType.h"

#include <optional>

/**
 * This class has all the informations necessary for the Inverted Barnes Hut algorithm
 * that need to be stored in a Cell. It does not perform any checks and should
 * not be used on its own, only as template argument for Cell.
 */
class BarnesHutInvertedCell {
public:
    using position_type = typename RelearnTypes::position_type;
    using counter_type = typename RelearnTypes::counter_type;

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

    /**
     * @brief Returns the position of the cell, which can be empty
     * @exception Throws a RelearnException if one position is valid and the other is not, or if they are at different points
     * @return The position of the excitatory dendrite
     */
    [[nodiscard]] std::optional<position_type> get_neuron_position() const {
        const auto& excitatory_position_opt = get_excitatory_axons_position();
        const auto& inhibitory_position_opt = get_inhibitory_axons_position();

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
            RelearnException::check(exc_position_equals_inh_position, "BarnesHutInvertedCell::get_neuron_position: positions are unequal");

            return pos_ex;
        }

        RelearnException::fail("BarnesHutInvertedCell::get_neuron_position: one pos was valid and one was not");

        return {};
    }

    /**
     * @brief Returns the number of free elements for the associated type in this cell
     * @param axon_type The requested axons type
     * @return The number of free axons for the associated type
     */
    [[nodiscard]] counter_type get_number_elements_for(const ElementType element_type, const SignalType signal_type) const {
        if (element_type == ElementType::DENDRITE) {
            RelearnException::fail("BarnesHutInvertedCell::get_number_elements_for: Does not support dendrites");
        }

        return get_number_axons_for(signal_type);
    }

    /**
     * @brief Returns the position of the specified element with the given signal type
     * @param axon_type The requested element type
     * @param signal_type The requested signal type
     * @exception Might throw a RelearnException if this operation is not supported
     * @return The position of the associated element, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_position_for(const ElementType element_type, const SignalType signal_type) const {
        if (element_type == ElementType::DENDRITE) {
            RelearnException::fail("BarnesHutInvertedCell::get_position_for: Does not support axons");
        }

        return get_axons_position_for(signal_type);
    }

private:
    VirtualPlasticityElement excitatory_axons{};
    VirtualPlasticityElement inhibitory_axons{};
};
