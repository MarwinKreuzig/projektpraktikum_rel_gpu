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

#include "RelearnException.h"
#include "Vec3.h"

#include <cmath>
#include <cstdint>

using BoxCoordinates = Vec3<uint64_t>;

class Morton {
public:
	Morton() = default;
	~Morton() = default;

	Morton(const Morton& other) = delete;
	Morton(Morton&& other) = default;

	Morton& operator=(const Morton& other) = delete;
	Morton& operator=(Morton&& other) = default;

	void map_1d_to_3d(uint64_t idx, BoxCoordinates& coords) const;

	uint64_t map_3d_to_1d(const BoxCoordinates& coords) const noexcept;

	size_t get_refinement_level() const noexcept {
		return this->refinement_level;
	}

	void set_refinement_level(size_t refinement_level) noexcept {
		this->refinement_level = refinement_level;
	}

private:
	void set_bit(uint64_t& variable, uint8_t bit) const noexcept {
		variable |= (static_cast<uint64_t>(1) << bit);
	}

	void unset_bit(uint64_t& variable, uint8_t bit) const noexcept {
		variable &= ~(static_cast<uint64_t>(1) << bit);
	}

	void copy_bit(const uint64_t& source, uint8_t source_bit, uint64_t& destination, uint8_t destination_bit) const;

	uint64_t select_bit(uint64_t number, uint8_t bit) const noexcept {
		return ((number & (static_cast<uint64_t>(1) << bit)) >> bit);
	}

	size_t refinement_level{ 0 };
};

template<class T>
class SpaceFillingCurve {
public:
	explicit SpaceFillingCurve(uint8_t refinement_level = 0) {
		set_refinement_level(refinement_level);
	}

	SpaceFillingCurve(const SpaceFillingCurve& other) = delete;
	SpaceFillingCurve(SpaceFillingCurve&& other) = default;

	SpaceFillingCurve& operator = (const SpaceFillingCurve& other) = delete;
	SpaceFillingCurve& operator = (SpaceFillingCurve&& other) = default;

	~SpaceFillingCurve() = default;

	size_t get_refinement_level() const noexcept {
		return curve.get_refinement_level();
	}

	void set_refinement_level(size_t num_subdivisions) {
		// With 64-bit keys we can only support 20 subdivisions per
		// dimension (i.e, 2^20 boxes per dimension)
		RelearnException::check(num_subdivisions <= 20, "Number of subdivisions is too large");

		curve.set_refinement_level(num_subdivisions);
	}

	void map_1d_to_3d(uint64_t idx, BoxCoordinates& coords) const noexcept {
		curve.map_1d_to_3d(idx, coords);
	}

	uint64_t map_3d_to_1d(const BoxCoordinates& coords) const noexcept {
		return curve.map_3d_to_1d(coords);
	}

private:
	T curve;
};
