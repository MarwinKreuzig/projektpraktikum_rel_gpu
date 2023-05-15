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

#include "util/ranges/Functional.hpp"

#include <range/v3/functional/not_fn.hpp>
#include <range/v3/range/operations.hpp>
#include <range/v3/view/filter.hpp>

namespace views {
inline constexpr auto filter_not_comment_not_empty_line = ranges::views::filter(ranges::not_fn(ranges::empty)) | ranges::views::filter(not_equal_to('#'), ranges::front);
} // namespace views
