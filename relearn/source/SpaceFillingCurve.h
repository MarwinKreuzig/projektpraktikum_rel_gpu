/*
 * File:   SpaceFillingCurve.h
 *
 * Created on Feb 28, 2017
 */

#ifndef SPACEFILLINGCURVE_H
#define SPACEFILLINGCURVE_H

#include <cassert>
#include <cmath>

#include <stdint.h>

#include "Vec3.h"

using BoxCoordinates = Vec3<uint64_t>;

class Morton {
public:
	Morton() noexcept
		: refinement_level(0) {
	}

	~Morton() noexcept {
	}

	Morton(const Morton& other) = delete;
	Morton(Morton&& other) = delete;

	Morton& operator=(const Morton& other) = delete;
	Morton& operator=(Morton&& other) = delete;

	void map_1d_to_3d(uint64_t idx, BoxCoordinates& coords) const noexcept {
		// clear coordinates
		coords.x = coords.y = coords.z = 0;

		// run over each bit and copy it to respective coordinate
		uint8_t coords_bit = 0;
		for (uint8_t idx_bit = 0; idx_bit < 60; idx_bit += 3) {
			copy_bit(idx, idx_bit, coords.x, coords_bit);
			++coords_bit;
		}
		coords_bit = 0;
		for (uint8_t idx_bit = 1; idx_bit < 60; idx_bit += 3) {
			copy_bit(idx, idx_bit, coords.y, coords_bit);
			++coords_bit;
		}
		coords_bit = 0;
		for (uint8_t idx_bit = 2; idx_bit < 60; idx_bit += 3) {
			copy_bit(idx, idx_bit, coords.z, coords_bit);
			++coords_bit;
		}
	}

	uint64_t map_3d_to_1d(const BoxCoordinates& coords) const noexcept {
		uint64_t result = 0;
		for (size_t i = 0; i < refinement_level; ++i) {
			uint64_t block = 0;
			const uint8_t short_i = static_cast<uint8_t>(i);
			block = ((select_bit(coords.z, short_i) << 2)
				+ (select_bit(coords.y, short_i) << 1)
				+ (select_bit(coords.x, short_i)));

			result |= block << (3 * i);
		}

		return result;
	}

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

	 void copy_bit(const uint64_t& source, uint8_t source_bit, uint64_t& destination, uint8_t destination_bit) const noexcept {
		// Not sure if this is correct and how it works.
		// That's why I provide a simpler solution below.
		// Please feel free to use your version if it's correct.
		//destination ^= (-select_bit(source, source_bit) ^ destination) & (1 << destination_bit);

		const uint64_t bit_in_source = select_bit(source, source_bit);
		if (1 == bit_in_source) {
			set_bit(destination, destination_bit);
		}
		else {
			unset_bit(destination, destination_bit);
		}
	}

	 uint64_t select_bit(uint64_t number, uint8_t bit) const noexcept {
		return ((number & (static_cast<uint64_t>(1) << bit)) >> bit);
	}

	size_t refinement_level;
};

template<class T>
class SpaceFillingCurve {
public:

	SpaceFillingCurve() noexcept {
		set_refinement_level(0);
	}

	SpaceFillingCurve(uint8_t refinement_level) {
		set_refinement_level(refinement_level);
	}

	SpaceFillingCurve(const SpaceFillingCurve& other) = delete;
	SpaceFillingCurve(SpaceFillingCurve&& other) = delete;

	SpaceFillingCurve& operator = (const SpaceFillingCurve & other) = delete;
	SpaceFillingCurve& operator = (SpaceFillingCurve && other) = delete;

	~SpaceFillingCurve() noexcept {
	}

	size_t get_refinement_level() const {
		return curve.get_refinement_level();
	}

	void set_refinement_level(size_t num_subdivisions) noexcept{
		// With 64-bit keys we can only support 20 subdivisions per
		// dimension (i.e, 2^20 boxes per dimension)
		assert(num_subdivisions <= 20);

		curve.set_refinement_level(num_subdivisions);
	}

	void map_1d_to_3d(uint64_t idx, BoxCoordinates& coords) noexcept {
		curve.map_1d_to_3d(idx, coords);
	}

	uint64_t map_3d_to_1d(const BoxCoordinates& coords) {
		return curve.map_3d_to_1d(coords);
	}

private:
	T curve;
};

#endif /* SPACEFILLINGCURVE_H */
