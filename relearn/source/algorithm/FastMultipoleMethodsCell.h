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
#include "../Config.h"
#include "../neurons/SignalType.h"
#include "../util/Vec3.h"

#include <array>
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
    [[nodiscard]] std::optional<Vec3d> get_inhibitory_dendrites_position() const noexcept {
        return inhibitory_dendrites.position;
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param opt_position The new position of the excitatory dendrite
     */
    void set_excitatory_dendrites_position(const std::optional<Vec3d>& opt_position) noexcept {
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

    [[nodiscard]] unsigned int get_number_axons_for(SignalType axon_type) const noexcept {
        if (axon_type == SignalType::EXCITATORY) {
            return excitatory_axons.num_free_elements;
        }

        return inhibitory_axons.num_free_elements;
    }

    [[nodiscard]] std::optional<Vec3d> get_axons_position_for(SignalType dendrite_type) const {
        if (dendrite_type == SignalType::EXCITATORY) {
            return excitatory_axons.position;
        }

        return inhibitory_axons.position;
    }

    [[nodiscard]] unsigned int get_number_dendrites_for(SignalType axon_type) const noexcept {
        if (axon_type == SignalType::EXCITATORY) {
            return excitatory_dendrites.num_free_elements;
        }

        return inhibitory_dendrites.num_free_elements;
    }

    [[nodiscard]] std::optional<Vec3d> get_dendrites_position_for(SignalType dendrite_type) const {
        if (dendrite_type == SignalType::EXCITATORY) {
            return excitatory_dendrites.position;
        }

        return inhibitory_dendrites.position;
    }

        void set_hermite_coef_ex(unsigned int x, double d) {
        hermite_coefficients_ex[x] = d;
    }

    void set_hermite_coef_in(unsigned int x, double d) {
        hermite_coefficients_in[x] = d;
    }

    void set_hermite_coef_for(unsigned int x, double d, SignalType needed) {
        if (needed == SignalType::EXCITATORY) {
            set_hermite_coef_ex(x, d);
        } else {
            set_hermite_coef_in(x, d);
        }
    }

    double get_hermite_coef_ex(unsigned int x) const {
        return hermite_coefficients_ex[x];
    }

    double get_hermite_coef_in(unsigned int x) const {
        return hermite_coefficients_in[x];
    }

    double get_hermite_coef_for(unsigned int x, SignalType needed) const {
        if (needed == SignalType::EXCITATORY) {
            return get_hermite_coef_ex(x);
        } else {
            return get_hermite_coef_in(x);
        }
    }

private:
    VirtualPlasticityElement excitatory_dendrites{};
    VirtualPlasticityElement inhibitory_dendrites{};

    VirtualPlasticityElement excitatory_axons{};
    VirtualPlasticityElement inhibitory_axons{};
        
    std::array<double, Constants::p3> hermite_coefficients_ex{ -1.0 };
    std::array<double, Constants::p3> hermite_coefficients_in{ -1.0 };
};
