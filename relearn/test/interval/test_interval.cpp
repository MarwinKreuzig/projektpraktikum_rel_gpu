/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_interval.h"

#include "adapter/interval/IntervalAdapter.h"
#include "adapter/random/RandomAdapter.h"

#include <sstream>

TEST_F(IntervalTest, testIntervalConstruction) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    Interval i{ begin, end, frequency };

    ASSERT_EQ(i.begin, begin);
    ASSERT_EQ(i.end, end);
    ASSERT_EQ(i.frequency, frequency);
}

TEST_F(IntervalTest, testIntervalIntersection) {
    Interval i1{ 10, 20, 1 };
    Interval i2{ 30, 40, 1 };

    ASSERT_FALSE(i1.check_for_intersection(i2));
    ASSERT_FALSE(i2.check_for_intersection(i1));

    Interval i3{ 100, 200, 4 };
    Interval i4{ 200, 300, 6 };

    ASSERT_TRUE(i3.check_for_intersection(i4));
    ASSERT_TRUE(i4.check_for_intersection(i3));

    Interval i5{ 1000, 10000, 5 };
    Interval i6{ 4000, 5000, 3 };

    ASSERT_TRUE(i5.check_for_intersection(i6));
    ASSERT_TRUE(i6.check_for_intersection(i5));

    Interval i7{ 2000, 3000, 70 };
    Interval i8{ 2500, 3500, 80 };

    ASSERT_TRUE(i7.check_for_intersection(i8));
    ASSERT_TRUE(i8.check_for_intersection(i7));
}

TEST_F(IntervalTest, testIntervalIntersecions) {
    std::vector<Interval> intervals{};

    intervals.emplace_back(IntervalAdapter::generate_random_interval(mt));
    intervals.emplace_back(IntervalAdapter::generate_random_interval(mt));
    intervals.emplace_back(IntervalAdapter::generate_random_interval(mt));

    auto golden_intersect = false;

    for (auto i = 0; i < 3; i++) {
        for (auto j = 0; j < 3; j++) {
            if (i == j) {
                continue;
            }

            golden_intersect |= intervals[i].check_for_intersection(intervals[j]);
        }
    }

    const auto all_intersect = Interval::check_intervals_for_intersection(intervals);

    ASSERT_EQ(golden_intersect, all_intersect);
}

TEST_F(IntervalTest, testParseInterval) {
    const auto& [golden_interval, description] = IntervalAdapter::generate_random_interval_description(mt);

    const auto& opt_interval = Interval::parse_interval(description);

    ASSERT_TRUE(opt_interval.has_value());

    const auto& interval = opt_interval.value();

    ASSERT_EQ(golden_interval.begin, interval.begin);
    ASSERT_EQ(golden_interval.end, interval.end);
    ASSERT_EQ(golden_interval.frequency, interval.frequency);
}

TEST_F(IntervalTest, testParseIntervalFail1) {
    const auto& opt_interval = Interval::parse_interval({});

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalTest, testParseIntervalFail2) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    std::stringstream ss{};
    ss << std::min(begin, end) << '-' << std::max(begin, end);

    const auto& description = ss.str();

    const auto& opt_interval = Interval::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalTest, testParseIntervalFail3) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    std::stringstream ss{};
    ss << begin << ':' << frequency;

    const auto& description = ss.str();

    const auto& opt_interval = Interval::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalTest, testParseIntervalFail4) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    std::stringstream ss{};
    ss << '-' << std::min(begin, end) << '-' << std::max(begin, end) << ':' << frequency;

    const auto& description = ss.str();

    const auto& opt_interval = Interval::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalTest, testParseIntervalFail5) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    std::stringstream ss{};
    ss << '-' << std::min(begin, end) << '-' << std::max(begin, end) << ':' << frequency << ':';

    const auto& description = ss.str();

    const auto& opt_interval = Interval::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalTest, testParseIntervalFail6) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto end = RandomAdapter::get_random_integer<int_type>(min, max, mt);
    const auto frequency = RandomAdapter::get_random_integer<int_type>(min, max, mt);

    std::stringstream ss{};
    ss << std::max(begin, end) << '-' << std::min(begin, end) << ':' << frequency;

    const auto& description = ss.str();

    const auto& opt_interval = Interval::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(IntervalTest, testParseIntervals1) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalAdapter::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        if (i != 9) {
            ss << ';';
        }
    }

    const auto& intervals = Interval::parse_description_as_intervals(ss.str());

    for (auto i = 0; i < 10; i++) {
        ASSERT_EQ(golden_intervals[i], intervals[i]);
    }
}

TEST_F(IntervalTest, testParseInterval2) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalAdapter::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        ss << ';';
    }

    const auto& intervals = Interval::parse_description_as_intervals(ss.str());

    ASSERT_EQ(intervals.size(), 10);
}

TEST_F(IntervalTest, testParseIntervalsFail1) {
    const auto& intervals = Interval::parse_description_as_intervals({});
    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalTest, testParseIntervalsFail2) {
    const auto& intervals = Interval::parse_description_as_intervals("sgahkllkrduf,'�.;f�lsa�df::SAfd--dfasdjf45");
    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalTest, testParseIntervalsFail3) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalAdapter::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        if (i != 9) {
            ss << ',';
        }
    }

    const auto& intervals = Interval::parse_description_as_intervals(ss.str());

    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalTest, testParseIntervalsFail4) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalAdapter::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        if (i != 9) {
            ss << ':';
        }
    }

    const auto& intervals = Interval::parse_description_as_intervals(ss.str());

    ASSERT_TRUE(intervals.empty());
}

TEST_F(IntervalTest, testParseIntervalsFail5) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = IntervalAdapter::generate_random_interval_description(mt);
        golden_intervals.emplace_back(interval);
        ss << description;

        ss << ';';
    }

    ss << "136546543135";

    const auto& intervals = Interval::parse_description_as_intervals(ss.str());

    ASSERT_EQ(intervals.size(), 10);
}
