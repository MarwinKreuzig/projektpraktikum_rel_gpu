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
 * This class has all the informations necessary for the Fast Multipole Methods algorithm
 * that need to be stored in a Cell. It does not perform any checks and should
 * not be used on its own, only as template argument for Cell.
 */
class FastMultipoleMethodsCell {
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
        if (dendrite_type == SignalType::Excitatory) {
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
     * @param axon_type The requested signal type
     * @return The position of the dendrite
     */
    [[nodiscard]] std::optional<position_type> get_dendrites_position_for(const SignalType dendrite_type) const noexcept {
        if (dendrite_type == SignalType::Excitatory) {
            return excitatory_dendrites.get_position();
        }

        return inhibitory_dendrites.get_position();
    }

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
        if (axon_type == SignalType::Excitatory) {
            return excitatory_axons.get_number_free_elements();
        }

        return inhibitory_axons.get_number_free_elements();
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param opt_position The new position of the inhibitory axons
     */
    void set_inhibitory_axons_position(const std::optional<position_type>& opt_position) noexcept {
        inhibitory_axons.set_position(opt_position);
    }

    /**
     * @brief Returns the position of the inhibitory axons, which can be empty
     * @return The position of the inhibitory axons
     */
    [[nodiscard]] std::optional<position_type> get_inhibitory_axons_position() const noexcept {
        return inhibitory_axons.get_position();
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param opt_position The new position of the excitatory axons
     */
    void set_excitatory_axons_position(const std::optional<position_type>& opt_position) noexcept {
        excitatory_axons.set_position(opt_position);
    }

    /**
     * @brief Returns the position of the excitatory axons, which can be empty
     * @return The position of the excitatory axons
     */
    [[nodiscard]] std::optional<position_type> get_excitatory_axons_position() const noexcept {
        return excitatory_axons.get_position();
    }

    /**
     * @brief Returns the position of the axons for the requested signal type, which can be empty
     * @param axon_type The requested signal type
     * @return The position of the dendrite
     */
    [[nodiscard]] std::optional<position_type> get_axons_position_for(const SignalType axon_type) const noexcept {
        if (axon_type == SignalType::Excitatory) {
            return excitatory_axons.get_position();
        }

        return inhibitory_axons.get_position();
    }

    /**
     * @brief Sets the position of the neuron for every necessary part of the cell
     * @param opt_position The position, can be empty
     */
    void set_neuron_position(const std::optional<position_type>& opt_position) noexcept {
        set_excitatory_dendrites_position(opt_position);
        set_inhibitory_dendrites_position(opt_position);
        set_excitatory_axons_position(opt_position);
        set_inhibitory_axons_position(opt_position);
    }

    /**
     * @brief Returns the position of the cell, which can be empty
     * @exception Throws a RelearnException if one position is valid and the others are not, or if they are at different points
     * @return The position of the excitatory dendrite
     */
    [[nodiscard]] std::optional<position_type> get_neuron_position() const {
        const auto& excitatory_axon_position_opt = get_excitatory_axons_position();
        const auto& inhibitory_axon_position_opt = get_inhibitory_axons_position();

        const auto& excitatory_dendrites_position_opt = get_excitatory_dendrites_position();
        const auto& inhibitory_dendrites_position_opt = get_inhibitory_dendrites_position();

        const bool ex_axon_valid = excitatory_axon_position_opt.has_value();
        const bool in_axon_valid = inhibitory_axon_position_opt.has_value();

        const bool ex_dendrite_valid = excitatory_dendrites_position_opt.has_value();
        const bool in_dendrite_valid = inhibitory_dendrites_position_opt.has_value();

        if (!ex_axon_valid && !in_axon_valid && !ex_dendrite_valid && !in_dendrite_valid) {
            return {};
        }

        if (ex_axon_valid && in_axon_valid && ex_dendrite_valid && in_dendrite_valid) {
            const auto& pos_ex_axon = excitatory_axon_position_opt.value();
            const auto& pos_in_axon = inhibitory_axon_position_opt.value();

            const auto& pos_ex_dendrite = excitatory_dendrites_position_opt.value();
            const auto& pos_in_dendrite = inhibitory_dendrites_position_opt.value();

            const auto diff1 = pos_ex_axon - pos_in_axon;
            const auto diff2 = pos_ex_axon - pos_ex_dendrite;
            const auto diff3 = pos_ex_axon - pos_in_dendrite;

            constexpr position_type null_position{ 0 };

            const auto all_equal = (diff1 == null_position) && (diff2 == null_position) && (diff3 == null_position);
            RelearnException::check(all_equal, "FastMultipoleMethodCell::get_neuron_position: positions are unequal");

            return pos_ex_axon;
        }

        RelearnException::fail("FastMultipoleMethodCell::get_neuron_position: one pos was valid and one was not");

        return {};
    }

    /**
     * @brief Returns the position of the specified element with the given signal type
     * @param axon_type The requested element type
     * @param signal_type The requested signal type
     * @return The position of the associated element, can be empty
     */
    [[nodiscard]] std::optional<position_type> get_position_for(const ElementType element_type, const SignalType signal_type) const noexcept {
        if (element_type == ElementType::Dendrite) {
            return get_dendrites_position_for(signal_type);
        }

        return get_axons_position_for(signal_type);
    }

    /**
     * @brief Returns the number of free elements for the associated type in this cell
     * @param axon_type The requested axons type
     * @return The number of free axons for the associated type
     */
    [[nodiscard]] counter_type get_number_elements_for(const ElementType element_type, const SignalType signal_type) const noexcept {
        if (element_type == ElementType::Dendrite) {
            return get_number_dendrites_for(signal_type);
        }

        return get_number_axons_for(signal_type);
    }

private:
    VirtualPlasticityElement excitatory_dendrites{};
    VirtualPlasticityElement inhibitory_dendrites{};

    VirtualPlasticityElement excitatory_axons{};
    VirtualPlasticityElement inhibitory_axons{};
};
