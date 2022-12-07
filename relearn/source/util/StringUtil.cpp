/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "StringUtil.h"

#include <string>
#include <vector>

/**
 * @brief Split a string basedd on a delimiter character in a list of substrings
 * @param string The string to split
 * @param delim Single char used as delimiter
 * @return Vector of substrings
 */
std::vector<std::string> StringUtil::split_string(const std::string& string, char delim) {
    std::vector<std::string> result{};
    result.reserve(string.size());

    std::stringstream ss(string);
    std::string item{};

    while (getline(ss, item, delim)) {
        result.emplace_back(std::move(item));
    }

    return result;
}
