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

#include <cstddef>

namespace Constants {
constexpr size_t number_oct = 8;
constexpr size_t uninitialized = 1111222233334444;

constexpr double theta = 0.3;
constexpr double sigma = 150.0;
constexpr size_t num_pend_vacant = 10;

constexpr size_t max_lvl_subdomains = 20;

constexpr size_t num_items_per_request = 6;

constexpr double eps = 0.00001;

constexpr size_t print_width = 12;

constexpr size_t plasticity_update_step = 100;
constexpr size_t logfile_update_step = 500;
} // namespace Constants
