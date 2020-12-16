#include "SpaceFillingCurve.h"

BoxCoordinates Morton::map_1d_to_3d(uint64_t idx) const {
	// clear coordinates
	BoxCoordinates coords{ 0 };

	// run over each bit and copy it to respective coordinate
	uint8_t coords_bit = 0;
	for (uint8_t idx_bit = 0; idx_bit < 60; idx_bit += 3) {
		coords.x = copy_bit(idx, idx_bit, coords.x, coords_bit);
		++coords_bit;
	}
	coords_bit = 0;
	for (uint8_t idx_bit = 1; idx_bit < 60; idx_bit += 3) {
		coords.y = copy_bit(idx, idx_bit, coords.y, coords_bit);
		++coords_bit;
	}
	coords_bit = 0;
	for (uint8_t idx_bit = 2; idx_bit < 60; idx_bit += 3) {
		coords.z = copy_bit(idx, idx_bit, coords.z, coords_bit);
		++coords_bit;
	}

	return coords;
}

uint64_t Morton::map_3d_to_1d(const BoxCoordinates& coords) const noexcept {
	uint64_t result = 0;
	for (size_t i = 0; i < refinement_level; ++i) {
		uint64_t block = 0;
		const auto short_i = static_cast<uint8_t>(i);
		block = ((select_bit(coords.z, short_i) << 2u)
			+ (select_bit(coords.y, short_i) << 1u)
			+ (select_bit(coords.x, short_i)));

		result |= block << (3 * i);
	}

	return result;
}

uint64_t Morton::copy_bit(uint64_t source, uint8_t source_bit, uint64_t destination, uint8_t destination_bit) /*noexcept*/ {
	// A simpler solution might be:
	// destination ^= (-select_bit(source, source_bit) ^ destination) & (1 << destination_bit);

	const uint64_t bit_in_source = select_bit(source, source_bit);
	if (1 == bit_in_source) {
		destination = set_bit(destination, destination_bit);
		return destination;
	}

	RelearnException::check(0 == bit_in_source, "In Morton, copy_bit, bit_in_source is neither 0 nor 1");
	destination = unset_bit(destination, destination_bit);
	return destination;
}
