#include "RelearnTest.hpp"

#include "gtest/gtest.h"

#include <sstream>

TEST_F(StepParserTest, testIntervalConstruction) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = get_random_integer<int_type>(min, max);
    const auto end = get_random_integer<int_type>(min, max);
    const auto frequency = get_random_integer<int_type>(min, max);

    Interval i{ begin, end, frequency };

    ASSERT_EQ(i.begin, begin);
    ASSERT_EQ(i.end, end);
    ASSERT_EQ(i.frequency, frequency);
}

TEST_F(StepParserTest, testIntervalIntersection) {
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

TEST_F(StepParserTest, testIntervalIntersecions) {
    std::vector<Interval> intervals{};

    intervals.emplace_back(generate_random_interval());
    intervals.emplace_back(generate_random_interval());
    intervals.emplace_back(generate_random_interval());

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

TEST_F(StepParserTest, testParseInterval) {
    const auto& [golden_interval, description] = generate_random_interval_description();

    const auto& opt_interval = Interval::parse_interval(description);

    ASSERT_TRUE(opt_interval.has_value());

    const auto& interval = opt_interval.value();

    ASSERT_EQ(golden_interval.begin, interval.begin);
    ASSERT_EQ(golden_interval.end, interval.end);
    ASSERT_EQ(golden_interval.frequency, interval.frequency);
}

TEST_F(StepParserTest, testParseIntervalFail1) {
    const auto& opt_interval = Interval::parse_interval({});

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(StepParserTest, testParseIntervalFail2) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = get_random_integer<int_type>(min, max);
    const auto end = get_random_integer<int_type>(min, max);

    std::stringstream ss{};
    ss << std::min(begin, end) << '-' << std::max(begin, end);

    const auto& description = ss.str();

    const auto& opt_interval = Interval::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(StepParserTest, testParseIntervalFail3) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = get_random_integer<int_type>(min, max);
    const auto frequency = get_random_integer<int_type>(min, max);

    std::stringstream ss{};
    ss << begin << ':' << frequency;

    const auto& description = ss.str();

    const auto& opt_interval = Interval::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(StepParserTest, testParseIntervalFail4) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = get_random_integer<int_type>(min, max);
    const auto end = get_random_integer<int_type>(min, max);
    const auto frequency = get_random_integer<int_type>(min, max);

    std::stringstream ss{};
    ss << '-' << std::min(begin, end) << '-' << std::max(begin, end) << ':' << frequency;

    const auto& description = ss.str();

    const auto& opt_interval = Interval::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(StepParserTest, testParseIntervalFail5) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    const auto begin = get_random_integer<int_type>(min, max);
    const auto end = get_random_integer<int_type>(min, max);
    const auto frequency = get_random_integer<int_type>(min, max);

    std::stringstream ss{};
    ss << '-' << std::min(begin, end) << '-' << std::max(begin, end) << ':' << frequency << ':';

    const auto& description = ss.str();

    const auto& opt_interval = Interval::parse_interval(description);

    ASSERT_FALSE(opt_interval.has_value());
}

TEST_F(StepParserTest, testParseIntervals1) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = generate_random_interval_description();
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

TEST_F(StepParserTest, testParseInterval2) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = generate_random_interval_description();
        golden_intervals.emplace_back(interval);
        ss << description;

        ss << ';';
    }

    const auto& intervals = Interval::parse_description_as_intervals(ss.str());

    ASSERT_EQ(intervals.size(), 10);
}

TEST_F(StepParserTest, testParseIntervalsFail1) {
    const auto& intervals = Interval::parse_description_as_intervals({});
    ASSERT_TRUE(intervals.empty());
}

TEST_F(StepParserTest, testParseIntervalsFail2) {
    const auto& intervals = Interval::parse_description_as_intervals("sgahkllkrduf,'�.;f�lsa�df::SAfd--dfasdjf45");
    ASSERT_TRUE(intervals.empty());
}

TEST_F(StepParserTest, testParseIntervalsFail3) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = generate_random_interval_description();
        golden_intervals.emplace_back(interval);
        ss << description;

        if (i != 9) {
            ss << ',';
        }
    }

    const auto& intervals = Interval::parse_description_as_intervals(ss.str());

    ASSERT_TRUE(intervals.empty());
}

TEST_F(StepParserTest, testParseIntervalsFail4) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = generate_random_interval_description();
        golden_intervals.emplace_back(interval);
        ss << description;

        if (i != 9) {
            ss << ':';
        }
    }

    const auto& intervals = Interval::parse_description_as_intervals(ss.str());

    ASSERT_TRUE(intervals.empty());
}

TEST_F(StepParserTest, testParseIntervalsFail5) {
    std::vector<Interval> golden_intervals{};
    std::stringstream ss{};

    for (auto i = 0; i < 10; i++) {
        const auto& [interval, description] = generate_random_interval_description();
        golden_intervals.emplace_back(interval);
        ss << description;

        ss << ';';
    }

    ss << "136546543135";

    const auto& intervals = Interval::parse_description_as_intervals(ss.str());

    ASSERT_EQ(intervals.size(), 10);
}

TEST_F(StepParserTest, testGenerateFunction1) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    auto function = StepParser::generate_step_check_function(std::vector<Interval>{});

    for (RelearnTypes::step_type step = 0; step < 10000; step++) {
        const auto result_1 = function(step);
        ASSERT_FALSE(result_1) << step;

        const auto random_step = get_random_integer<int_type>(min, max);
        const auto result_2 = function(random_step);
        ASSERT_FALSE(result_2) << random_step;
    }
}

TEST_F(StepParserTest, testGenerateFunction2) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    Interval i{ min, max, 1 };

    auto function = StepParser::generate_step_check_function({ i });

    for (RelearnTypes::step_type step = 0; step < 10000; step++) {
        const auto result_1 = function(step);
        ASSERT_TRUE(result_1) << step;

        const auto random_step = get_random_integer<int_type>(min, max);
        const auto result_2 = function(random_step);
        ASSERT_TRUE(result_2) << random_step;
    }
}

TEST_F(StepParserTest, testGenerateFunction3) {
    using int_type = Interval::step_type;

    constexpr auto min = std::numeric_limits<int_type>::min();
    constexpr auto max = std::numeric_limits<int_type>::max();

    Interval i1{ 0, 99, 10 };
    Interval i2{ 100, 999, 10 };
    Interval i3{ 1000, 2689, 10 };
    Interval i4{ 2690, 10000, 10 };

    auto function = StepParser::generate_step_check_function({ i1, i2, i3, i4 });

    for (RelearnTypes::step_type step = 0; step < 20000; step++) {
        const auto result_1 = function(step);
        ASSERT_EQ(result_1, (step <= 10000 && step % 10 == 0)) << step;

        const auto random_step = get_random_integer<int_type>(min, max);
        const auto result_2 = function(random_step);
        ASSERT_EQ(result_2, (random_step <= 10000 && random_step % 10 == 0)) << random_step;
    }
}

TEST_F(StepParserTest, testGenerateFunction4) {
    using int_type = Interval::step_type;

    constexpr auto min = 10000;
    constexpr auto max = 90000;

    auto begin = get_random_integer<int_type>(min, max);
    auto end = get_random_integer<int_type>(min, max);

    Interval i1{ 0, 99, 7 };
    Interval i2{ 100, 999, 7 };
    Interval i3{ 1000, 2689, 7 };
    Interval i4{ 2690, 9999, 7 };
    Interval i5{ std::min(begin, end), std::max(begin, end), 11 };

    std::stringstream ss{};
    ss << codify_interval(i1) << ';';
    ss << codify_interval(i2) << ';';
    ss << codify_interval(i3) << ';';
    ss << codify_interval(i4) << ';';
    ss << codify_interval(i5);

    auto function_1 = StepParser::generate_step_check_function({ i1, i2, i3, i4, i5 });
    auto function_2 = StepParser::generate_step_check_function(ss.str());

    for (RelearnTypes::step_type step = 0; step < 90000; step++) {
        const auto result_1 = function_1(step);
        const auto result_2 = function_2(step);
        ASSERT_EQ(result_1, result_2) << step;
    }
}
