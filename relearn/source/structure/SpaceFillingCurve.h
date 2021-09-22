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

#include "../Config.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"

#include <cstdint>
#include <type_traits>

/**
 * This class represents a MortonCurve in 3D.
 * This class does not perform argument checking. SpaceFillingCurve<Morton> should be used for that.
 */
class Morton {
public:
    using BoxCoordinates = Vec3s;

    /**
     * @brief Maps a one dimensional index into the three dimensional domain.
     * @param idx The one dimensional index
     * @return The three dimensional index
     */
    [[nodiscard]] static BoxCoordinates map_1d_to_3d(const uint64_t idx) noexcept;
   
    /**
     * @brief Maps a three dimensional index into the one dimensional domain.
     * @param idx The three dimensional index
     * @return The one dimensional index
     */
    [[nodiscard]] uint64_t map_3d_to_1d(const BoxCoordinates& coords) const noexcept;

    /**
     * @brief Returns the current refinement level
     * @return The current refinement level
     */
    [[nodiscard]] size_t get_refinement_level() const noexcept {
        return this->refinement_level;
    }

    /**
     * @brief Sets the new refinement level
     * @param refinement_level The new refinement level
     */
    void set_refinement_level(const size_t refinement_level) noexcept {
        this->refinement_level = refinement_level;
    }

private:
    [[nodiscard]] static uint64_t set_bit(const uint64_t variable, const uint8_t bit) noexcept {
        const auto val = variable | (static_cast<uint64_t>(1) << bit);
        return val;
    }

    [[nodiscard]] static uint64_t unset_bit(const uint64_t variable, const uint8_t bit) noexcept {
        const auto val = variable & ~(static_cast<uint64_t>(1) << bit);
        return val;
    }

    [[nodiscard]] static uint64_t copy_bit(const uint64_t source, const uint8_t source_bit, const uint64_t destination, const uint8_t destination_bit);

    [[nodiscard]] static uint64_t select_bit(const uint64_t number, const uint8_t bit) noexcept {
        return ((number & (static_cast<uint64_t>(1) << bit)) >> bit);
    }

    size_t refinement_level{ 0 };
};

/**
 * This class represents a space filling curve in 3D.
 * It is parameterized by an actual implementation T, which must be nothrow {constructible, copy constructible, move constructible}.
 */
template <class T>
class SpaceFillingCurve {
    static_assert(std::is_nothrow_constructible_v<T>);
    static_assert(std::is_nothrow_copy_constructible_v<T>);
    static_assert(std::is_nothrow_move_constructible_v<T>);

public:
    using BoxCoordinates = Vec3s;

    /**
     * @brief Constructs a new instance of a space filling curve with the desired refinement level
     * @param refinement_level The desired refinement level
     * @exception Throws a RelearnException if refinement_level > Constants::max_lvl_subdomains
     */
    explicit SpaceFillingCurve(const uint8_t refinement_level = 0) {
        set_refinement_level(refinement_level);
    }

    /**
     * @brief Returns the current refinement level
     * @return The current refinement level
     */
    [[nodiscard]] size_t get_refinement_level() const noexcept {
        return curve.get_refinement_level();
    }

    /**
     * @brief Sets the new refinement level
     * @param refinement_level The new refinement level
     * @exception Throws a RelearnException if refinement_level > Constants::max_lvl_subdomains
     */
    void set_refinement_level(const size_t refinement_level) {
        // With 64-bit keys we can only support 20 subdivisions per
        // dimension (i.e, 2^20 boxes per dimension)
        RelearnException::check(refinement_level <= Constants::max_lvl_subdomains, 
            "SpaceFillingCurve::set_refinement_level:Number of subdivisions is too large: {} vs {}", refinement_level, Constants::max_lvl_subdomains);

        curve.set_refinement_level(refinement_level);
    }

    /**
     * @brief Maps a one dimensional index into the three dimensional domain.
     * @param idx The one dimensional index
     * @return The three dimensional index
     */
    [[nodiscard]] BoxCoordinates map_1d_to_3d(const uint64_t idx) const noexcept {
        return curve.map_1d_to_3d(idx);
    }

    /**
     * @brief Maps a three dimensional index into the one dimensional domain.
     * @param idx The three dimensional index
     * @return The one dimensional index
     */
    [[nodiscard]] uint64_t map_3d_to_1d(const BoxCoordinates& coords) const noexcept {
        return curve.map_3d_to_1d(coords);
    }

private:
    T curve{};
};
