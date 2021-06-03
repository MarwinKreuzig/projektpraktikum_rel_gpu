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

#include "../util/RelearnException.h"

#include <string>
#include <utility>
#include <vector>

class InteractiveNeuronIO {
public:
    static std::vector<std::pair<size_t, std::vector<size_t>>> load_enable_interrups(const std::string& path_to_file);

    static std::vector<std::pair<size_t, std::vector<size_t>>> load_disable_interrups(const std::string& path_to_file);

    static std::vector<std::pair<size_t, size_t>> load_creation_interrups(const std::string& path_to_file);
};
