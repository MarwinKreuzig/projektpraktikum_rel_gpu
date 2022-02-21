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

#include "Types.h"

#include <filesystem>
#include <functional>
#include <utility>

class CalciumIO {
public:
    using initial_value_calculator = std::function<double(NeuronID::value_type)>;
    using target_value_calculator = std::function<double(NeuronID::value_type)>;

    static initial_value_calculator load_initial_function(const std::filesystem::path& path_to_file);

    static target_value_calculator load_target_function(const std::filesystem::path& path_to_file);

    static std::pair<initial_value_calculator, target_value_calculator> load_initial_and_target_function(const std::filesystem::path& path_to_file);
};
