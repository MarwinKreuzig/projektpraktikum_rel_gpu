#include "../googletest/include/gtest/gtest.h"

#include <random>

#include "RelearnTest.hpp"

#include "../source/util/Vec3.h"
#include "../source/util/RelearnException.h"

constexpr const double lower_bound = -100.0;
constexpr const double upper_bound = 100.0;

TEST_F(VectorTest, test_vector_constructor_empty) {
    for (auto i = 0; i < iterations; i++) {
        Vec3<double> v{};

        ASSERT_EQ(0.0, v.get_x());
        ASSERT_EQ(0.0, v.get_y());
        ASSERT_EQ(0.0, v.get_z());
    }
}

TEST_F(VectorTest, test_vector_constructor_one) {
    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto val = urd(mt);

        Vec3<double> v{ val };

        ASSERT_EQ(val, v.get_x());
        ASSERT_EQ(val, v.get_y());
        ASSERT_EQ(val, v.get_z());
    }
}

TEST_F(VectorTest, test_vector_constructor_three) {
    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        Vec3<double> v{ x, y, z };

        ASSERT_EQ(x, v.get_x());
        ASSERT_EQ(y, v.get_y());
        ASSERT_EQ(z, v.get_z());
    }
}

TEST_F(VectorTest, test_vector_structured_binding) {
    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        Vec3<double> v{ x, y, z };

        const auto& [x2, y2, z2] = v;

        ASSERT_EQ(x, x2);
        ASSERT_EQ(y, y2);
        ASSERT_EQ(z, z2);

        auto& [x3, y3, z3] = v;

        ASSERT_EQ(x, x3);
        ASSERT_EQ(y, y3);
        ASSERT_EQ(z, z3);

        auto [x4, y4, z4] = v;

        ASSERT_EQ(x, x4);
        ASSERT_EQ(y, y4);
        ASSERT_EQ(z, z4);

        auto new_x = urd(mt);
        auto new_y = urd(mt);
        auto new_z = urd(mt);

        x3 = new_x;
        y3 = new_y;
        z3 = new_z;

        ASSERT_EQ(new_x, v.get_x());
        ASSERT_EQ(new_y, v.get_y());
        ASSERT_EQ(new_z, v.get_z());
    }
}

TEST_F(VectorTest, test_vector_copy_construct) {
    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        Vec3<double> v{ x, y, z };

        Vec3<double> w = v;

        ASSERT_EQ(x, w.get_x());
        ASSERT_EQ(y, w.get_y());
        ASSERT_EQ(z, w.get_z());
    }
}

TEST_F(VectorTest, test_vector_copy_assign) {
    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        Vec3<double> v{ x, y, z };

        Vec3<double> w{ x + 1.0, y + 1.0, z + 1.0 };
        w = v;

        ASSERT_EQ(x, w.get_x());
        ASSERT_EQ(y, w.get_y());
        ASSERT_EQ(z, w.get_z());
    }
}

TEST_F(VectorTest, test_vector_move_construct) {
    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        Vec3<double> v{ x, y, z };

        Vec3<double> w = std::move(v);

        ASSERT_EQ(x, w.get_x());
        ASSERT_EQ(y, w.get_y());
        ASSERT_EQ(z, w.get_z());
    }
}

TEST_F(VectorTest, test_vector_move_assign) {
    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        Vec3<double> v{ x, y, z };

        Vec3<double> w{ x + 1.0, y + 1.0, z + 1.0 };
        w = std::move(v);

        ASSERT_EQ(x, w.get_x());
        ASSERT_EQ(y, w.get_y());
        ASSERT_EQ(z, w.get_z());
    }
}

TEST_F(VectorTest, test_vector_operator_plus_vec) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x1 = urd(mt);
        auto y1 = urd(mt);
        auto z1 = urd(mt);

        Vec3<double> v{ x1, y1, z1 };

        auto x2 = urd(mt);
        auto y2 = urd(mt);
        auto z2 = urd(mt);

        Vec3<double> w{ x2, y2, z2 };

        auto sum = v + w;

        ASSERT_EQ(x1, v.get_x());
        ASSERT_EQ(y1, v.get_y());
        ASSERT_EQ(z1, v.get_z());

        ASSERT_EQ(x2, w.get_x());
        ASSERT_EQ(y2, w.get_y());
        ASSERT_EQ(z2, w.get_z());

        ASSERT_EQ(x1 + x2, sum.get_x());
        ASSERT_EQ(y1 + y2, sum.get_y());
        ASSERT_EQ(z1 + z2, sum.get_z());
    }
}

TEST_F(VectorTest, test_vector_operator_minus_vec) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x1 = urd(mt);
        auto y1 = urd(mt);
        auto z1 = urd(mt);

        Vec3<double> v{ x1, y1, z1 };

        auto x2 = urd(mt);
        auto y2 = urd(mt);
        auto z2 = urd(mt);

        Vec3<double> w{ x2, y2, z2 };

        auto diff = v - w;

        ASSERT_EQ(x1, v.get_x());
        ASSERT_EQ(y1, v.get_y());
        ASSERT_EQ(z1, v.get_z());

        ASSERT_EQ(x2, w.get_x());
        ASSERT_EQ(y2, w.get_y());
        ASSERT_EQ(z2, w.get_z());

        ASSERT_EQ(x1 - x2, diff.get_x());
        ASSERT_EQ(y1 - y2, diff.get_y());
        ASSERT_EQ(z1 - z2, diff.get_z());
    }
}

TEST_F(VectorTest, test_vector_operator_plus_scalar) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x1 = urd(mt);
        auto y1 = urd(mt);
        auto z1 = urd(mt);

        Vec3<double> v{ x1, y1, z1 };

        auto scalar = urd(mt);
        auto scalar_copy = scalar;

        auto sum = v + scalar;

        ASSERT_EQ(x1, v.get_x());
        ASSERT_EQ(y1, v.get_y());
        ASSERT_EQ(z1, v.get_z());

        ASSERT_EQ(x1 + scalar, sum.get_x());
        ASSERT_EQ(y1 + scalar, sum.get_y());
        ASSERT_EQ(z1 + scalar, sum.get_z());

        ASSERT_EQ(scalar, scalar_copy);
    }
}

TEST_F(VectorTest, test_vector_operator_minus_scalar) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x1 = urd(mt);
        auto y1 = urd(mt);
        auto z1 = urd(mt);

        Vec3<double> v{ x1, y1, z1 };

        auto scalar = urd(mt);
        auto scalar_copy = scalar;

        auto sum = v - scalar;

        ASSERT_EQ(x1, v.get_x());
        ASSERT_EQ(y1, v.get_y());
        ASSERT_EQ(z1, v.get_z());

        ASSERT_EQ(x1 - scalar, sum.get_x());
        ASSERT_EQ(y1 - scalar, sum.get_y());
        ASSERT_EQ(z1 - scalar, sum.get_z());

        ASSERT_EQ(scalar, scalar_copy);
    }
}

TEST_F(VectorTest, test_vector_operator_mul_scalar) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x1 = urd(mt);
        auto y1 = urd(mt);
        auto z1 = urd(mt);

        Vec3<double> v{ x1, y1, z1 };

        auto scalar = urd(mt);
        auto scalar_copy = scalar;

        auto prod = v * scalar;

        ASSERT_EQ(x1, v.get_x());
        ASSERT_EQ(y1, v.get_y());
        ASSERT_EQ(z1, v.get_z());

        ASSERT_EQ(x1 * scalar, prod.get_x());
        ASSERT_EQ(y1 * scalar, prod.get_y());
        ASSERT_EQ(z1 * scalar, prod.get_z());

        ASSERT_EQ(scalar, scalar_copy);
    }
}

TEST_F(VectorTest, test_vector_operator_div_scalar) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x1 = urd(mt);
        auto y1 = urd(mt);
        auto z1 = urd(mt);

        Vec3<double> v{ x1, y1, z1 };

        auto scalar = urd(mt);
        while (scalar == 0.0) {
            scalar = urd(mt);
        }

        auto scalar_copy = scalar;

        auto prod = v / scalar;

        ASSERT_EQ(x1, v.get_x());
        ASSERT_EQ(y1, v.get_y());
        ASSERT_EQ(z1, v.get_z());

        ASSERT_EQ(x1 / scalar, prod.get_x());
        ASSERT_EQ(y1 / scalar, prod.get_y());
        ASSERT_EQ(z1 / scalar, prod.get_z());

        ASSERT_EQ(scalar, scalar_copy);
    }
}

TEST_F(VectorTest, test_vector_operator_plus_assign_scalar) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x1 = urd(mt);
        auto y1 = urd(mt);
        auto z1 = urd(mt);

        Vec3<double> v{ x1, y1, z1 };

        auto scalar = urd(mt);
        auto scalar_copy = scalar;

        v += scalar;

        ASSERT_EQ(x1 + scalar, v.get_x());
        ASSERT_EQ(y1 + scalar, v.get_y());
        ASSERT_EQ(z1 + scalar, v.get_z());

        ASSERT_EQ(scalar, scalar_copy);
    }
}

TEST_F(VectorTest, test_vector_operator_plus_assign_vector) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        const auto x1 = urd(mt);
        const auto y1 = urd(mt);
        const auto z1 = urd(mt);

        Vec3<double> v{ x1, y1, z1 };

        const auto x2 = urd(mt);
        const auto y2 = urd(mt);
        const auto z2 = urd(mt);

        Vec3<double> w{ x2, y2, z2 };

        v += w;

        ASSERT_EQ(x1 + x2, v.get_x());
        ASSERT_EQ(y1 + y2, v.get_y());
        ASSERT_EQ(z1 + z2, v.get_z());
    }
}

TEST_F(VectorTest, test_vector_operator_minus_assign_vector) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        const auto x1 = urd(mt);
        const auto y1 = urd(mt);
        const auto z1 = urd(mt);

        Vec3<double> v{ x1, y1, z1 };

        const auto x2 = urd(mt);
        const auto y2 = urd(mt);
        const auto z2 = urd(mt);

        Vec3<double> w{ x2, y2, z2 };

        v -= w;

        ASSERT_EQ(x1 - x2, v.get_x());
        ASSERT_EQ(y1 - y2, v.get_y());
        ASSERT_EQ(z1 - z2, v.get_z());
    }
}

TEST_F(VectorTest, test_vector_operator_mul_assign_scalar) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x1 = urd(mt);
        auto y1 = urd(mt);
        auto z1 = urd(mt);

        Vec3<double> v{ x1, y1, z1 };

        auto scalar = urd(mt);
        auto scalar_copy = scalar;

        v *= scalar;

        ASSERT_EQ(x1 * scalar, v.get_x());
        ASSERT_EQ(y1 * scalar, v.get_y());
        ASSERT_EQ(z1 * scalar, v.get_z());

        ASSERT_EQ(scalar, scalar_copy);
    }
}

TEST_F(VectorTest, test_vector_volume) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        Vec3<double> v{ x, y, z };

        auto volume = x * y * z;

        ASSERT_EQ(volume, v.get_volume());

        ASSERT_EQ(x, v.get_x());
        ASSERT_EQ(y, v.get_y());
        ASSERT_EQ(z, v.get_z());
    }
}

TEST_F(VectorTest, test_vector_equal) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        Vec3<double> v_1{ x, y, z };
        Vec3<double> v_2{ x, y, z };

        auto is_equal = v_1 == v_2;

        ASSERT_TRUE(is_equal);
    }

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = x + 1;
        auto z = y + 1;

        Vec3<double> v_1{ x, y, z };
        Vec3<double> v_2{ x, z, y };
        Vec3<double> v_3{ y, x, z };
        Vec3<double> v_4{ y, z, x };
        Vec3<double> v_5{ z, x, y };
        Vec3<double> v_6{ z, y, x };

        auto is_equal_2 = v_1 == v_2;
        auto is_equal_3 = v_1 == v_3;
        auto is_equal_4 = v_1 == v_4;
        auto is_equal_5 = v_1 == v_5;
        auto is_equal_6 = v_1 == v_6;

        ASSERT_FALSE(is_equal_2);
        ASSERT_FALSE(is_equal_3);
        ASSERT_FALSE(is_equal_4);
        ASSERT_FALSE(is_equal_5);
        ASSERT_FALSE(is_equal_6);
    }
}

TEST_F(VectorTest, test_vector_componentwise_min_max) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x_1 = urd(mt);
        auto y_1 = urd(mt);
        auto z_1 = urd(mt);

        auto x_2 = urd(mt);
        auto y_2 = urd(mt);
        auto z_2 = urd(mt);

        auto x_max = std::max(x_1, x_2);
        auto x_min = std::min(x_1, x_2);
        auto y_max = std::max(y_1, y_2);
        auto y_min = std::min(y_1, y_2);
        auto z_max = std::max(z_1, z_2);
        auto z_min = std::min(z_1, z_2);

        Vec3<double> v_1{ x_1, y_1, z_1 };
        Vec3<double> v_2{ x_2, y_2, z_2 };

        Vec3<double> v_max{ x_max, y_max, z_max };
        Vec3<double> v_min{ x_min, y_min, z_min };

        v_1.calculate_componentwise_maximum(v_2);
        auto is_equal = v_1 == v_max;
        ASSERT_TRUE(is_equal);

        v_1 = Vec3<double>(x_1, y_1, z_1);

        v_1.calculate_componentwise_minimum(v_2);
        is_equal = v_1 == v_min;
        ASSERT_TRUE(is_equal);
    }
}

TEST_F(VectorTest, test_vector_min_max) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        Vec3<double> v{ x, y, z };

        auto max = v.get_maximum();
        auto min = v.get_minimum();

        auto is_there = (x == max) || (y == max) || (z == max);
        auto is_bound = (x <= max) || (y <= max) || (z <= max);

        auto is_equal = is_there && is_bound;

        ASSERT_TRUE(is_equal);

        is_there = (x == min) || (y == min) || (z == min);
        is_bound = (x >= min) || (y >= min) || (z >= min);

        is_equal = is_there && is_bound;

        ASSERT_TRUE(is_equal);
    }
}

TEST_F(VectorTest, test_vector_componentwise_floor) {

    std::uniform_real_distribution<double> urd(0.0001, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        auto x_floor = floor(x);
        auto y_floor = floor(y);
        auto z_floor = floor(z);

        auto x_size_t = static_cast<size_t>(x_floor);
        auto y_size_t = static_cast<size_t>(y_floor);
        auto z_size_t = static_cast<size_t>(z_floor);

        Vec3<double> v{ x, y, z };
        Vec3<size_t> v_size_t{ x_size_t, y_size_t, z_size_t };

        Vec3<size_t> v_floored = v.floor_componentwise();

        auto is_equal = v_size_t == v_floored;

        ASSERT_TRUE(is_equal);
    }
}

TEST_F(VectorTest, test_vector_componentwise_floor_assert) {
    std::uniform_real_distribution<double> urd(0.0001, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        x *= -1.0;

        Vec3<double> v{ x, y, z };
        ASSERT_THROW(Vec3<size_t> v_floored = v.floor_componentwise(), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        y *= -1.0;

        Vec3<double> v{ x, y, z };
        ASSERT_THROW(Vec3<size_t> v_floored = v.floor_componentwise(), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        z *= -1.0;

        Vec3<double> v{ x, y, z };
        ASSERT_THROW(Vec3<size_t> v_floored = v.floor_componentwise(), RelearnException);
    }
}

TEST_F(VectorTest, test_vector_norm) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);
    std::uniform_real_distribution<double> urd_p(1.0, 10.1);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        auto p = urd_p(mt);

        Vec3<double> v{ x, y, z };

        auto v_normed = v.calculate_p_norm(p);

        auto x_p = pow(abs(x), p);
        auto y_p = pow(abs(y), p);
        auto z_p = pow(abs(z), p);

        auto sum = x_p + y_p + z_p;
        auto p_norm = pow(sum, 1.0 / p);

        ASSERT_NEAR(v_normed, p_norm, eps);
    }
}

TEST_F(VectorTest, test_vector_norm_assert) {
    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);
    std::uniform_real_distribution<double> urd_bad_p(lower_bound, 1.0);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        auto p = urd_bad_p(mt);

        Vec3<double> v{ x, y, z };

        ASSERT_THROW(auto v_normed = v.calculate_p_norm(p), RelearnException);
    }
}

TEST_F(VectorTest, test_vector_rounding) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);
    std::uniform_real_distribution<double> urd_multiple(1.0, 10.1);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        auto multiple = urd_multiple(mt);

        auto x_div = x / multiple;
        auto y_div = y / multiple;
        auto z_div = z / multiple;

        Vec3<double> v{ x, y, z };

        v.round_to_larger_multiple(multiple);

        auto x_rounded = v.get_x();
        auto y_rounded = v.get_y();
        auto z_rounded = v.get_z();

        auto x_rounded_div = x_rounded / multiple;
        auto y_rounded_div = y_rounded / multiple;
        auto z_rounded_div = z_rounded / multiple;

        auto x_in_expected = x_div <= x_rounded_div && x_rounded_div <= x_div + 1;
        auto y_in_expected = y_div <= y_rounded_div && y_rounded_div <= y_div + 1;
        auto z_in_expected = z_div <= z_rounded_div && z_rounded_div <= z_div + 1;

        auto x_rounded_div_rounded = round(x_rounded_div);
        auto y_rounded_div_rounded = round(y_rounded_div);
        auto z_rounded_div_rounded = round(z_rounded_div);

        ASSERT_TRUE(x_in_expected);
        ASSERT_TRUE(y_in_expected);
        ASSERT_TRUE(z_in_expected);

        ASSERT_NEAR(x_rounded_div, x_rounded_div_rounded, eps);
        ASSERT_NEAR(y_rounded_div, y_rounded_div_rounded, eps);
        ASSERT_NEAR(z_rounded_div, z_rounded_div_rounded, eps);
    }
}

TEST_F(VectorTest, test_vector_order) {

    std::uniform_real_distribution<double> urd(lower_bound, upper_bound);
    std::uniform_real_distribution<double> urd_offset(1.0, 10.1);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        Vec3<double> v_1{ x, y, z };
        Vec3<double> v_2{ x, y, z };

        auto is_smaller = v_1 < v_2;
        ASSERT_FALSE(is_smaller);
    }

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        auto offset = urd_offset(mt);

        Vec3<double> v_1{ x, y, z };
        Vec3<double> v_2{ x + offset, y, z };

        auto is_smaller = v_1 < v_2;
        ASSERT_TRUE(is_smaller);

        is_smaller = v_2 < v_1;
        ASSERT_FALSE(is_smaller);
    }

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        auto offset = urd_offset(mt);

        Vec3<double> v_1{ x, y, z };
        Vec3<double> v_2{ x, y + offset, z };

        auto is_smaller = v_1 < v_2;
        ASSERT_TRUE(is_smaller);

        is_smaller = v_2 < v_1;
        ASSERT_FALSE(is_smaller);
    }

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        auto offset = urd_offset(mt);

        Vec3<double> v_1{ x, y, z };
        Vec3<double> v_2{ x, y, z + offset };

        auto is_smaller = v_1 < v_2;
        ASSERT_TRUE(is_smaller);

        is_smaller = v_2 < v_1;
        ASSERT_FALSE(is_smaller);
    }

    for (auto i = 0; i < iterations; i++) {
        auto x_1 = urd(mt);
        auto y_1 = urd(mt);
        auto z_1 = urd(mt);

        auto x_2 = urd(mt);
        auto y_2 = urd(mt);
        auto z_2 = urd(mt);

        Vec3<double> v_1{ x_1, y_1, z_1 };
        Vec3<double> v_2{ x_2, y_2, z_2 };

        auto is_smaller_1_2 = v_1 < v_2;
        auto is_smaller_2_1 = v_2 < v_1;

        auto connected = is_smaller_1_2 || is_smaller_2_1;

        if (connected) {
            ASSERT_FALSE(is_smaller_1_2 && is_smaller_2_1);
        } else {
            auto is_equal = v_1 == v_2;
            ASSERT_TRUE(is_equal);
        }
    }
}

TEST_F(VectorTest, test_vector_cast) {

    std::uniform_real_distribution<double> urd(0.0, upper_bound);

    for (auto i = 0; i < iterations; i++) {
        auto x = urd(mt);
        auto y = urd(mt);
        auto z = urd(mt);

        auto x_size_t = static_cast<size_t>(x);
        auto y_size_t = static_cast<size_t>(y);
        auto z_size_t = static_cast<size_t>(z);

        Vec3<double> v{ x, y, z };
        Vec3<size_t> v_casted = static_cast<Vec3<size_t>>(v);
        Vec3<size_t> v_size_t{ x_size_t, y_size_t, z_size_t };

        auto is_equal = v_casted == v_size_t;
        ASSERT_TRUE(is_equal);
    }
}
