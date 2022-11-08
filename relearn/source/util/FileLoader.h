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

#include <string>
#include <utility>
#include <vector>
#include <map>
#include <functional>

/**
 * This class provides a static interface to load interrupts from files, i.e., when during the simulation the neurons should be altered.
 */
class FileLoader {
public:
    static std::vector<size_t> load_neuron_id_list(const std::string& path_to_file);

    static std::function<std::vector<std::pair<size_t, double>>(size_t)> load_external_stimulus(const std::string& path_to_file);

    static std::vector<std::string> split_string(const std::string &string, char delim);
};
