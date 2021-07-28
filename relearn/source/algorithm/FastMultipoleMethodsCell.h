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

#include "Types.h"
#include "../util/Vec3.h"

#include <optional>

/**
 * This class has all the informations necessary for the Fast Multipole Methods algorithm
 * that need to be stored in a Cell. It does not perform any checks and should
 * not be used on its own, only as template argument for Cell.
 */
class FastMultipoleMethodsCell {
public:
    /**
     * @brief Sets the number of free excitatory dendrites in this cell
     * @param num_dendrites The number of free excitatory dendrites
     */
    void set_number_excitatory_dendrites(unsigned int num_dendrites) noexcept {
        excitatory_dendrites.num_free_elements = num_dendrites;
    }

    /**
     * @brief Returns the number of free excitatory dendrites in this cell
     * @return The number of free excitatory dendrites
     */
    [[nodiscard]] unsigned int get_number_excitatory_dendrites() const noexcept {
        return excitatory_dendrites.num_free_elements;
    }

    /**
     * @brief Sets the number of free inhibitory dendrites in this cell
     * @param num_dendrites The number of free inhibitory dendrites
     */
    void set_number_inhibitory_dendrites(unsigned int num_dendrites) noexcept {
        inhibitory_dendrites.num_free_elements = num_dendrites;
    }

    /**
     * @brief Returns the number of free inhibitory dendrites in this cell
     * @return The number of free inhibitory dendrites
     */
    [[nodiscard]] unsigned int get_number_inhibitory_dendrites() const noexcept {
        return inhibitory_dendrites.num_free_elements;
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param opt_position The new position of the inhibitory dendrite
     */
    void set_inhibitory_dendrites_position(const std::optional<Vec3d>& opt_position) noexcept {
        inhibitory_dendrites.position = opt_position;
    }

    /**
     * @brief Returns the position of the inhibitory dendrite
     * @return The position of the inhibitory dendrite
     */
    [[nodiscard]] std::optional<Vec3d> get_inhibitory_dendrite_position() const noexcept {
        return inhibitory_dendrites.position;
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param opt_position The new position of the excitatory dendrite
     */
    void set_excitatory_dendrite_position(const std::optional<Vec3d>& opt_position) noexcept {
        excitatory_dendrites.position = opt_position;
    }

    /**
     * @brief Returns the position of the excitatory dendrite
     * @return The position of the excitatory dendrite
     */
    [[nodiscard]] std::optional<Vec3d> get_excitatory_dendrites_position() const noexcept {
        return excitatory_dendrites.position;
    }

    /**
     * @brief Sets the number of free excitatory axons in this cell
     * @param num_axons The number of free excitatory axons
     */
    void set_number_excitatory_axons(unsigned int num_axons) noexcept {
        excitatory_axons.num_free_elements = num_axons;
    }

    /**
     * @brief Returns the number of free excitatory axons in this cell
     * @return The number of free excitatory axons
     */
    [[nodiscard]] unsigned int get_number_excitatory_axons() const noexcept {
        return excitatory_axons.num_free_elements;
    }

    /**
     * @brief Sets the number of free inhibitory axons in this cell
     * @param num_axons The number of free inhibitory axons
     */
    void set_number_inhibitory_axons(unsigned int num_axons) noexcept {
        inhibitory_axons.num_free_elements = num_axons;
    }

    /**
     * @brief Returns the number of free inhibitory axons in this cell
     * @return The number of free inhibitory axons
     */
    [[nodiscard]] unsigned int get_number_inhibitory_axons() const noexcept {
        return inhibitory_axons.num_free_elements;
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param opt_position The new position of the inhibitory axons
     */
    void set_inhibitory_axons_position(const std::optional<Vec3d>& opt_position) noexcept {
        inhibitory_axons.position = opt_position;
    }

    /**
     * @brief Returns the position of the inhibitory axons
     * @return The position of the inhibitory axons
     */
    [[nodiscard]] std::optional<Vec3d> get_inhibitory_axons_position() const noexcept {
        return inhibitory_axons.position;
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param opt_position The new position of the excitatory axons
     */
    void set_excitatory_axons_position(const std::optional<Vec3d>& opt_position) noexcept {
        excitatory_axons.position = opt_position;
    }

    /**
     * @brief Returns the position of the excitatory axons
     * @return The position of the excitatory axons
     */
    [[nodiscard]] std::optional<Vec3d> get_excitatory_axons_position() const noexcept {
        return excitatory_axons.position;
    }

private:
    Dendrites excitatory_dendrites{};
    Dendrites inhibitory_dendrites{};

    Axons excitatory_axons{};
    Axons inhibitory_axons{};
};
