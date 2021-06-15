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

#include <vector>
#include <tuple>

#include "RelearnException.h"

namespace Util {

template <typename T>
std::tuple<T, T, T, size_t> min_max_acc(const std::vector<T>& values, const std::vector<char>& disable_flags) {
    RelearnException::check(values.size() > 0, "In min_max_acc, values had size 0");
    RelearnException::check(values.size() == disable_flags.size(), "In min_max_acc, values and disable_flags hat different sizes");
    
    size_t first_index = 0;

    while (disable_flags[first_index] == 0) {
        first_index++;
    }

    T min = values[first_index];
    T max = values[first_index];
    T acc = values[first_index];

    size_t num_values = 1;

    for (auto i = first_index + 1; i < values.size(); i++) {
        if (disable_flags[i] == 0) {
            continue;
        }

        const T& current_value = values[i];

        if (current_value < min) {
            min = current_value;
        } else if (current_value > max) {
            max = current_value;
        }

        acc += current_value;
        num_values++;
    }

    return std::make_tuple(min, max, acc, num_values);
}

template <typename T>
constexpr unsigned int num_digits(T val) noexcept {
    unsigned int num_digits = 0;

    do {
        ++num_digits;
        // NOLINTNEXTLINE
        val /= 10;
    } while (val != 0);

    return num_digits;
}
} // namespace Util
