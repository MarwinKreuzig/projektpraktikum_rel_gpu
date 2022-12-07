/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_misc.h"

#include "util/Utility.h"

TEST_F(MiscTest, testNumberDigitsInt) {
    using integer_type = int;

    std::vector<integer_type> small_integers{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    for (const auto val : small_integers) {
        ASSERT_EQ(1, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = get_random_integer<integer_type>(10, 99);
        ASSERT_EQ(2, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = get_random_integer<integer_type>(100, 999);
        ASSERT_EQ(3, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = get_random_integer<integer_type>(1000, 9999);
        ASSERT_EQ(4, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = get_random_integer<integer_type>(10000, 99999);
        ASSERT_EQ(5, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = get_random_integer<integer_type>(100000, 999999);
        ASSERT_EQ(6, Util::num_digits(val));
    }
}

TEST_F(MiscTest, testNumberDigitsUnsignedInt) {
    using integer_type = unsigned int;

    std::vector<integer_type> small_integers{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    for (const auto val : small_integers) {
        ASSERT_EQ(1, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = get_random_integer<integer_type>(10, 99);
        ASSERT_EQ(2, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = get_random_integer<integer_type>(100, 999);
        ASSERT_EQ(3, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = get_random_integer<integer_type>(1000, 9999);
        ASSERT_EQ(4, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = get_random_integer<integer_type>(10000, 99999);
        ASSERT_EQ(5, Util::num_digits(val));
    }

    for (auto i = 0; i < 10; i++) {
        const auto val = get_random_integer<integer_type>(100000, 999999);
        ASSERT_EQ(6, Util::num_digits(val));
    }
}

TEST_F(MiscTest, testFactorial) {
    constexpr auto fac0 = Util::factorial(0ULL);
    constexpr auto fac1 = Util::factorial(1ULL);
    constexpr auto fac2 = Util::factorial(2ULL);
    constexpr auto fac3 = Util::factorial(3ULL);
    constexpr auto fac4 = Util::factorial(4ULL);
    constexpr auto fac5 = Util::factorial(5ULL);
    constexpr auto fac6 = Util::factorial(6ULL);
    constexpr auto fac7 = Util::factorial(7ULL);
    constexpr auto fac8 = Util::factorial(8ULL);
    constexpr auto fac9 = Util::factorial(9ULL);
    constexpr auto fac10 = Util::factorial(10ULL);
    constexpr auto fac11 = Util::factorial(11ULL);
    constexpr auto fac12 = Util::factorial(12ULL);
    constexpr auto fac13 = Util::factorial(13ULL);
    constexpr auto fac14 = Util::factorial(14ULL);
    constexpr auto fac15 = Util::factorial(15ULL);

    ASSERT_EQ(fac0, 1ULL);
    ASSERT_EQ(fac1, 1ULL);
    ASSERT_EQ(fac2, 2ULL);
    ASSERT_EQ(fac3, 6ULL);
    ASSERT_EQ(fac4, 24ULL);
    ASSERT_EQ(fac5, 120ULL);
    ASSERT_EQ(fac6, 720ULL);
    ASSERT_EQ(fac7, 5040ULL);
    ASSERT_EQ(fac8, 40320ULL);
    ASSERT_EQ(fac9, 362880ULL);
    ASSERT_EQ(fac10, 3628800ULL);
    ASSERT_EQ(fac11, 39916800ULL);
    ASSERT_EQ(fac12, 479001600ULL);
    ASSERT_EQ(fac13, 6227020800ULL);
    ASSERT_EQ(fac14, 87178291200ULL);
    ASSERT_EQ(fac15, 1307674368000ULL);
}

TEST_F(MiscTest, testMinMaxAccEmpty) {
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<double>{}, std::vector<UpdateStatus>{}), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<float>{}, std::vector<UpdateStatus>{}), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<int>{}, std::vector<UpdateStatus>{}), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<size_t>{}, std::vector<UpdateStatus>{}), RelearnException);
}

TEST_F(MiscTest, testMinMaxAccSizeMismatch) {
    std::vector<UpdateStatus> update_status(3, UpdateStatus::Enabled);

    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<double>{ 4.0, 1.2 }, update_status), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<float>{ 0.8f }, update_status), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<int>{ 5, -4, 8, -6, 9 }, update_status), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<size_t>{ 10, 422, 5223, 554315 }, update_status), RelearnException);
}

TEST_F(MiscTest, testMinMaxAccSizeAllDisabled) {
    std::vector<UpdateStatus> update_status(3, UpdateStatus::Disabled);

    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<double>{ 4.0, 1.2, 5.2 }, update_status), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<float>{ 0.8f, -1.6f, 65423.8f }, update_status), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<int>{ 5, -4, 8 }, update_status), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<size_t>{ 10, 422, 5223 }, update_status), RelearnException);
}

TEST_F(MiscTest, testMinMaxAccSizeAllStatic) {
    std::vector<UpdateStatus> update_status(3, UpdateStatus::STATIC);

    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<double>{ 4.0, 1.2, 5.2 }, update_status), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<float>{ 0.8f, -1.6f, 65423.8f }, update_status), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<int>{ 5, -4, 8 }, update_status), RelearnException);
    ASSERT_THROW(auto val = Util::min_max_acc(std::vector<size_t>{ 10, 422, 5223 }, update_status), RelearnException);
}

TEST_F(MiscTest, testMinMaxAccDouble) {
    const auto number_enabled = get_random_number_neurons();
    const auto number_disabled = get_random_number_neurons();
    const auto number_static = get_random_number_neurons();

    const auto number_values = number_enabled + number_disabled + number_static;

    std::vector<UpdateStatus> update_status(number_enabled, UpdateStatus::Enabled);
    update_status.resize(number_enabled + number_disabled, UpdateStatus::Disabled);
    update_status.resize(number_values, UpdateStatus::STATIC);

    shuffle(update_status.begin(), update_status.end());

    std::vector<double> values{};
    values.reserve(number_values);

    auto min = std::numeric_limits<double>::max();
    auto max = std::numeric_limits<double>::min();
    auto sum = 0.0;

    for (auto i = 0; i < number_values; i++) {
        const auto random_value = get_random_double(-100000.0, 100000.0);

        if (update_status[i] == UpdateStatus::Enabled) {
            min = std::min(min, random_value);
            max = std::max(max, random_value);
            sum += random_value;
        }

        values.emplace_back(random_value);
    }

    const auto [minimum, maximum, accumulated, num] = Util::min_max_acc(values, update_status);

    ASSERT_EQ(minimum, min);
    ASSERT_EQ(maximum, max);
    ASSERT_NEAR(sum, accumulated, eps);
    ASSERT_EQ(number_enabled, num);
}

TEST_F(MiscTest, testMinMaxAccSizet) {
    const auto number_enabled = get_random_number_neurons();
    const auto number_disabled = get_random_number_neurons();
    const auto number_static = get_random_number_neurons();

    const auto number_values = number_enabled + number_disabled + number_static;

    std::vector<UpdateStatus> update_status(number_enabled, UpdateStatus::Enabled);
    update_status.resize(number_enabled + number_disabled, UpdateStatus::Disabled);
    update_status.resize(number_values, UpdateStatus::STATIC);

    shuffle(update_status.begin(), update_status.end());

    std::vector<size_t> values{};
    values.reserve(number_values);

    auto min = std::numeric_limits<size_t>::max();
    auto max = std::numeric_limits<size_t>::min();
    auto sum = size_t(0);

    for (auto i = 0; i < number_values; i++) {
        const auto random_value = get_random_integer<size_t>(std::numeric_limits<size_t>::min(), std::numeric_limits<size_t>::max());

        if (update_status[i] == UpdateStatus::Enabled) {
            min = std::min(min, random_value);
            max = std::max(max, random_value);
            sum += random_value;
        }

        values.emplace_back(random_value);
    }

    const auto [minimum, maximum, accumulated, num] = Util::min_max_acc(values, update_status);

    ASSERT_EQ(minimum, min);
    ASSERT_EQ(maximum, max);
    ASSERT_EQ(sum, accumulated);
    ASSERT_EQ(number_enabled, num);
}
