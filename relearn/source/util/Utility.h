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

#include "neurons/UpdateStatus.h"
#include "util/RelearnException.h"

#include <tuple>
#include <type_traits>
#include <vector>

namespace Util {

/**
 * @brief Calculates the minimum, maximum, and sum over all values in the vector, for which the disable flags are not enabled
 * @tparam T Must be a arithmetic (floating point or integral)
 * @param values The values that should be reduced
 * @param disable_flags The flags that indicate which values to skip
 * @exception Throws a RelearnException if (a) values.empty(), (b) values.size() != disable_flags.size(), (c) all values are disabled
 * @return Returns a tuple with (1) minimum and (2) maximum value from values, (3) the sum of all enabled values and (4) the number of enabled values
 */
template <typename T>
std::tuple<T, T, T, size_t> min_max_acc(const std::vector<T>& values, const std::vector<UpdateStatus>& disable_flags) {
    static_assert(std::is_arithmetic<T>::value);

    RelearnException::check(!values.empty(), "Util::min_max_acc: values are empty");
    RelearnException::check(values.size() == disable_flags.size(), "Util::min_max_acc: values and disable_flags had different sizes");

    size_t first_index = 0;

    while (first_index < values.size() && disable_flags[first_index] != UpdateStatus::Enabled) {
        first_index++;
    }

    RelearnException::check(first_index != values.size(), "Util::min_max_acc: all were disabled");

    T min = values[first_index];
    T max = values[first_index];
    T acc = values[first_index];

    size_t num_values = 1;

    for (auto i = first_index + 1; i < values.size(); i++) {
        if (disable_flags[i] != UpdateStatus::Enabled) {
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

/**
 * @brief Counts the number of digits necessary to print the value
 * @tparam T Must be integral
 * @param val The value to print
 * @return The number of digits of val
 */
template <typename T>
constexpr unsigned int num_digits(T val) noexcept {
    static_assert(std::is_integral<T>::value);

    constexpr const auto number_system_base = 10;
    unsigned int num_digits = 1;

    while (val >= T(number_system_base)) {
        ++num_digits;
        // NOLINTNEXTLINE
        val /= number_system_base;
    }

    return num_digits;
}

/**
 * @brief Calculates the faculty.
 * @param value
 * @tparam T Type of which a faculty should be calculated. Must fullfill std::is_unsigned_v<T>
 * @return Returns the faculty of the paramter value.
 */
template <typename T>
static constexpr T factorial(T value) noexcept {
    static_assert(std::is_unsigned_v<T>, "bad T");
    if (value < 2) {
        return 1;
    }

    T result = 1;
    while (value > 1) {
        result *= value;
        value--;
    }

    return result;
}
} // namespace Util
