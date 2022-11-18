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

#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/**
 * This class provides a static interface to load interrupts from files, i.e., when during the simulation the neurons should be altered.
 */
class Helper {
public:
    static std::vector<std::string> split_string(const std::string& string, char delim);

    /**
     * @brief Checks if the string is a int number
     * @param s a string
     * @return true if string is a number
     */
    static bool is_number(const std::string& s) {
        for (char const& ch : s) {
            if (std::isdigit(ch) == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Converts an integer to a string with leading zeros
     * @param number The number will be converted to a string
     * @param nr_of_digits Number of digits including the leading zeros
     * @return string with the number and leading zeros if necessary
     */
    static std::string format_int_with_leading_zeros(int number, int nr_of_digits) {
        std::stringstream ss;
        ss << std::setw(nr_of_digits) << std::setfill('0') << number;
        return ss.str();
    }

    template <typename T>
    static void stack_vectors(std::vector<std::vector<T>>& first, const std::vector<std::vector<T>>& second) {
        RelearnException::check(first.size() == second.size(), "Helper::stack_vectors: Cannot stack vectors with different size {} != {} ", first.size(), second.size());

        for (int i = 0; i < first.size(); i++) {
            first[i].insert(first[i].end(), second[i].begin(), second[i].end());
        }
    }
};
