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

#include "RelearnException.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

/**
 * This class provides a static interface to load interrupts from files, i.e., when during the simulation the neurons should be altered.
 */
class StringUtil {
public:
    /**
     * @brief Split a string based on a delimiter character in a list of substrings.
     *      Empty strings within two delimiters are or at the beginning kept while one at the end is discarded
     * @param string The string to split
     * @param delim Single char used as delimiter
     * @return Vector of substrings
     */
    static std::vector<std::string> split_string(const std::string& string, char delim) {
        std::vector<std::string> result{};
        result.reserve(string.size());

        std::stringstream ss(string);
        std::string item{};

        while (getline(ss, item, delim)) {
            result.emplace_back(std::move(item));
        }

        return result;
    }

    /**
     * @brief Checks if the string contains only digits
     * @param s a string
     * @return true if string is a number
     */
    static bool is_number(const std::string_view s) {
        return std::all_of(s.begin(), s.end(), [](const char c) { return std::isdigit(c); });
    }

    /**
     * Converts an integer to a string with leading zeros, having at least the number of specified digits
     * @param number The number will be converted to a string
     * @param nr_of_digits Number of digits including the leading zeros
     * @exception Throws a RelearnException if nr_of_digits == 0
     * @return string with the number and leading zeros if necessary
     */
    static std::string format_int_with_leading_zeros(const int number, const unsigned int nr_of_digits) {
        RelearnException::check(nr_of_digits >= 1, "StringUtil::format_with_leading_zeros. Number must has at least 1 digit");
        std::stringstream ss{};
        ss << std::setw(nr_of_digits) << std::setfill('0') << number;
        return ss.str();
    }
};
