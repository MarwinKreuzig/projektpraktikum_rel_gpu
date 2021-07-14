/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SpaceFillingCurve.h"

Morton::BoxCoordinates Morton::map_1d_to_3d(uint64_t idx) {
    // clear coordinates
    BoxCoordinates coords{ 0 };

    constexpr uint8_t loop_bound = 60;

    // run over each bit and copy it to respective coordinate
    uint8_t coords_bit = 0;
    for (uint8_t idx_bit = 0; idx_bit < loop_bound; idx_bit += 3) {
        const auto& old = coords.get_x();
        const auto& new_val = copy_bit(idx, idx_bit, old, coords_bit);
        coords.set_x(new_val);
        ++coords_bit;
    }
    coords_bit = 0;
    for (uint8_t idx_bit = 1; idx_bit < loop_bound; idx_bit += 3) {
        const auto& old = coords.get_y();
        const auto& new_val = copy_bit(idx, idx_bit, old, coords_bit);
        coords.set_y(new_val);
        ++coords_bit;
    }
    coords_bit = 0;
    for (uint8_t idx_bit = 2; idx_bit < loop_bound; idx_bit += 3) {
        const auto& old = coords.get_z();
        const auto& new_val = copy_bit(idx, idx_bit, old, coords_bit);
        coords.set_z(new_val);
        ++coords_bit;
    }

    return coords;
}

uint64_t Morton::map_3d_to_1d(const Morton::BoxCoordinates& coords) const noexcept {
    uint64_t result = 0;
    for (size_t i = 0; i < refinement_level; ++i) {
        uint64_t block = 0;
        const auto short_i = static_cast<uint8_t>(i);
        block = ((select_bit(coords.get_z(), short_i) << 2U)
            + (select_bit(coords.get_y(), short_i) << 1U)
            + (select_bit(coords.get_x(), short_i)));

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
