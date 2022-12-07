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

#include "RelearnTest.hpp"

#include "util/Interval.h"

#include <climits>
#include <string>
#include <utility>

class StepParserTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }

    Interval generate_random_interval() {
        using int_type = Interval::step_type;

        constexpr auto min = std::numeric_limits<int_type>::min();
        constexpr auto max = std::numeric_limits<int_type>::max();

        const auto begin = get_random_integer<int_type>(min, max);
        const auto end = get_random_integer<int_type>(min, max);
        const auto frequency = get_random_integer<int_type>(min, max);

        return Interval{ std::min(begin, end), std::max(begin, end), frequency };
    }

    std::string codify_interval(const Interval& interval) {
        std::stringstream ss{};
        ss << interval.begin << '-' << interval.end << ':' << interval.frequency;
        return ss.str();
    }

    std::pair<Interval, std::string> generate_random_interval_description() {
        auto interval = generate_random_interval();
        auto description = codify_interval(interval);
        return { std::move(interval), std::move(description) };
    }
};
