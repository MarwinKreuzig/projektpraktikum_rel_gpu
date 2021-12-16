#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/util/Vec3.h"

#include <sstream>

TEST_F(VectorTest, testVectorEmptyConstructor) {
    const Vec3<double> v{};

    ASSERT_EQ(0.0, v.get_x());
    ASSERT_EQ(0.0, v.get_y());
    ASSERT_EQ(0.0, v.get_z());
}

TEST_F(VectorTest, testVectorSameValueConstructor) {
    const auto val = get_random_vector_element();

    const Vec3<double> v{ val };

    std::stringstream ss{};
    ss << val;

    ASSERT_EQ(val, v.get_x()) << ss.str();
    ASSERT_EQ(val, v.get_y()) << ss.str();
    ASSERT_EQ(val, v.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorThreeValuesConstructor) {
    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    const Vec3<double> v{ x, y, z };

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    ASSERT_EQ(x, v.get_x()) << ss.str();
    ASSERT_EQ(y, v.get_y()) << ss.str();
    ASSERT_EQ(z, v.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorStructuredBinding) {
    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    Vec3<double> v{ x, y, z };

    const auto& [x2, y2, z2] = v;

    ASSERT_EQ(x, x2) << ss.str();
    ASSERT_EQ(y, y2);
    ASSERT_EQ(z, z2);

    auto& [x3, y3, z3] = v;

    ASSERT_EQ(x, x3) << ss.str();
    ASSERT_EQ(y, y3) << ss.str();
    ASSERT_EQ(z, z3) << ss.str();

    auto [x4, y4, z4] = v;

    ASSERT_EQ(x, x4) << ss.str();
    ASSERT_EQ(y, y4) << ss.str();
    ASSERT_EQ(z, z4) << ss.str();

    const auto new_x = get_random_vector_element();
    const auto new_y = get_random_vector_element();
    const auto new_z = get_random_vector_element();

    ss << new_x << ' ' << new_y << ' ' << new_z;

    x3 = new_x;
    y3 = new_y;
    z3 = new_z;

    ASSERT_EQ(new_x, v.get_x()) << ss.str();
    ASSERT_EQ(new_y, v.get_y()) << ss.str();
    ASSERT_EQ(new_z, v.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorCopyConstructor) {
    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const Vec3<double> v{ x, y, z };

    const Vec3<double> w = v;

    ASSERT_EQ(x, w.get_x()) << ss.str();
    ASSERT_EQ(y, w.get_y()) << ss.str();
    ASSERT_EQ(z, w.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorCopyAssignment) {
    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const Vec3<double> v{ x, y, z };

    Vec3<double> w{ x + 1.0, y + 1.0, z + 1.0 };
    w = v;

    ASSERT_EQ(x, w.get_x()) << ss.str();
    ASSERT_EQ(y, w.get_y()) << ss.str();
    ASSERT_EQ(z, w.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorMoveConstructor) {
    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    Vec3<double> v{ x, y, z };

    const Vec3<double> w = std::move(v);

    ASSERT_EQ(x, w.get_x()) << ss.str();
    ASSERT_EQ(y, w.get_y()) << ss.str();
    ASSERT_EQ(z, w.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorMoveAssignment) {
    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    Vec3<double> v{ x, y, z };

    Vec3<double> w{ x + 1.0, y + 1.0, z + 1.0 };
    w = std::move(v);

    ASSERT_EQ(x, w.get_x()) << ss.str();
    ASSERT_EQ(y, w.get_y()) << ss.str();
    ASSERT_EQ(z, w.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorSetComponents) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << x2 << ' ' << y2 << ' ' << z2 << '\n';

    Vec3<double> v{ x1, y1, z1 };

    v.set_x(x2);
    v.set_y(y2);
    v.set_z(z2);

    ASSERT_EQ(x2, v.get_x()) << ss.str();
    ASSERT_EQ(y2, v.get_y()) << ss.str();
    ASSERT_EQ(z2, v.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorPlusVector) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const Vec3<double> v{ x1, y1, z1 };

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << x2 << ' ' << y2 << ' ' << z2 << '\n';

    const Vec3<double> w{ x2, y2, z2 };

    const auto sum = v + w;

    ASSERT_EQ(x1, v.get_x()) << ss.str();
    ASSERT_EQ(y1, v.get_y()) << ss.str();
    ASSERT_EQ(z1, v.get_z()) << ss.str();

    ASSERT_EQ(x2, w.get_x()) << ss.str();
    ASSERT_EQ(y2, w.get_y()) << ss.str();
    ASSERT_EQ(z2, w.get_z()) << ss.str();

    ASSERT_EQ(x1 + x2, sum.get_x()) << ss.str();
    ASSERT_EQ(y1 + y2, sum.get_y()) << ss.str();
    ASSERT_EQ(z1 + z2, sum.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorMinusVector) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const Vec3<double> v{ x1, y1, z1 };

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << x2 << ' ' << y2 << ' ' << z2 << '\n';

    const Vec3<double> w{ x2, y2, z2 };

    const auto diff = v - w;

    ASSERT_EQ(x1, v.get_x()) << ss.str();
    ASSERT_EQ(y1, v.get_y()) << ss.str();
    ASSERT_EQ(z1, v.get_z()) << ss.str();

    ASSERT_EQ(x2, w.get_x()) << ss.str();
    ASSERT_EQ(y2, w.get_y()) << ss.str();
    ASSERT_EQ(z2, w.get_z()) << ss.str();

    ASSERT_EQ(x1 - x2, diff.get_x()) << ss.str();
    ASSERT_EQ(y1 - y2, diff.get_y()) << ss.str();
    ASSERT_EQ(z1 - z2, diff.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorPlusScalar) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const Vec3<double> v{ x1, y1, z1 };

    const auto scalar = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << "scalar: " << scalar;

    const auto sum = v + scalar;

    ASSERT_EQ(x1, v.get_x()) << ss.str();
    ASSERT_EQ(y1, v.get_y()) << ss.str();
    ASSERT_EQ(z1, v.get_z()) << ss.str();

    ASSERT_EQ(x1 + scalar, sum.get_x()) << ss.str();
    ASSERT_EQ(y1 + scalar, sum.get_y()) << ss.str();
    ASSERT_EQ(z1 + scalar, sum.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorMinusScalar) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const Vec3<double> v{ x1, y1, z1 };

    const auto scalar = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << "scalar: " << scalar;

    const auto sum = v - scalar;

    ASSERT_EQ(x1, v.get_x()) << ss.str();
    ASSERT_EQ(y1, v.get_y()) << ss.str();
    ASSERT_EQ(z1, v.get_z()) << ss.str();

    ASSERT_EQ(x1 - scalar, sum.get_x()) << ss.str();
    ASSERT_EQ(y1 - scalar, sum.get_y()) << ss.str();
    ASSERT_EQ(z1 - scalar, sum.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorMultiplyScalar) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const Vec3<double> v{ x1, y1, z1 };
    const auto scalar = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << "scalar: " << scalar;

    const auto prod = v * scalar;

    ASSERT_EQ(x1, v.get_x()) << ss.str();
    ASSERT_EQ(y1, v.get_y()) << ss.str();
    ASSERT_EQ(z1, v.get_z()) << ss.str();

    ASSERT_EQ(x1 * scalar, prod.get_x()) << ss.str();
    ASSERT_EQ(y1 * scalar, prod.get_y()) << ss.str();
    ASSERT_EQ(z1 * scalar, prod.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorDivideScalar) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const Vec3<double> v{ x1, y1, z1 };

    auto scalar = get_random_vector_element();
    while (scalar == 0.0) {
        scalar = get_random_vector_element();
    }

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << "scalar: " << scalar;

    const auto prod = v / scalar;

    ASSERT_EQ(x1, v.get_x()) << ss.str();
    ASSERT_EQ(y1, v.get_y()) << ss.str();
    ASSERT_EQ(z1, v.get_z()) << ss.str();

    ASSERT_EQ(x1 / scalar, prod.get_x()) << ss.str();
    ASSERT_EQ(y1 / scalar, prod.get_y()) << ss.str();
    ASSERT_EQ(z1 / scalar, prod.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorPlusAssignVector) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    Vec3<double> v{ x1, y1, z1 };

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const Vec3<double> w{ x2, y2, z2 };

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << x2 << ' ' << y2 << ' ' << z2 << '\n';

    v += w;

    ASSERT_EQ(x1 + x2, v.get_x()) << ss.str();
    ASSERT_EQ(y1 + y2, v.get_y()) << ss.str();
    ASSERT_EQ(z1 + z2, v.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorPlusAssignScalar) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    Vec3<double> v{ x1, y1, z1 };

    const auto scalar = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << "scalar: " << scalar;

    v += scalar;

    ASSERT_EQ(x1 + scalar, v.get_x()) << ss.str();
    ASSERT_EQ(y1 + scalar, v.get_y()) << ss.str();
    ASSERT_EQ(z1 + scalar, v.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorMinusAssignVector) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    Vec3<double> v{ x1, y1, z1 };

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const Vec3<double> w{ x2, y2, z2 };

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << x2 << ' ' << y2 << ' ' << z2 << '\n';

    v -= w;

    ASSERT_EQ(x1 - x2, v.get_x()) << ss.str();
    ASSERT_EQ(y1 - y2, v.get_y()) << ss.str();
    ASSERT_EQ(z1 - z2, v.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorMinusAssignScalar) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    Vec3<double> v{ x1, y1, z1 };

    const auto scalar = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << "scalar: " << scalar;

    v -= scalar;

    ASSERT_EQ(x1 - scalar, v.get_x()) << ss.str();
    ASSERT_EQ(y1 - scalar, v.get_y()) << ss.str();
    ASSERT_EQ(z1 - scalar, v.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorMultiplyAssignScalar) {
    auto x1 = get_random_vector_element();
    auto y1 = get_random_vector_element();
    auto z1 = get_random_vector_element();

    Vec3<double> v{ x1, y1, z1 };

    const auto scalar = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << "scalar: " << scalar;

    v *= scalar;

    ASSERT_EQ(x1 * scalar, v.get_x()) << ss.str();
    ASSERT_EQ(y1 * scalar, v.get_y()) << ss.str();
    ASSERT_EQ(z1 * scalar, v.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorOperatorDivideAssignScalar) {
    auto x1 = get_random_vector_element();
    auto y1 = get_random_vector_element();
    auto z1 = get_random_vector_element();

    Vec3<double> v{ x1, y1, z1 };

    auto scalar = get_random_vector_element();
    while (scalar == 0.0) {
        scalar = get_random_vector_element();
    }

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << "scalar: " << scalar;

    v /= scalar;

    ASSERT_EQ(x1 / scalar, v.get_x()) << ss.str();
    ASSERT_EQ(y1 / scalar, v.get_y()) << ss.str();
    ASSERT_EQ(z1 / scalar, v.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorVolume) {
    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    const Vec3<double> v{ x, y, z };

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const auto volume = x * y * z;

    ASSERT_EQ(volume, v.get_volume()) << ss.str();

    ASSERT_EQ(x, v.get_x()) << ss.str();
    ASSERT_EQ(y, v.get_y()) << ss.str();
    ASSERT_EQ(z, v.get_z()) << ss.str();
}

TEST_F(VectorTest, testVectorEqual) {
    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const Vec3<double> v_1{ x, y, z };
    const Vec3<double> v_2{ x, y, z };

    const auto is_equal = v_1 == v_2;

    ASSERT_TRUE(is_equal) << ss.str();
}

TEST_F(VectorTest, testVectorUnequal) {
    const auto x = get_random_vector_element();
    const auto y = x + 1;
    const auto z = y + 1;

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const Vec3<double> v_1{ x, y, z };
    const Vec3<double> v_2{ x, z, y };
    const Vec3<double> v_3{ y, x, z };
    const Vec3<double> v_4{ y, z, x };
    const Vec3<double> v_5{ z, x, y };
    const Vec3<double> v_6{ z, y, x };

    const auto is_equal_2 = v_1 == v_2;
    const auto is_equal_3 = v_1 == v_3;
    const auto is_equal_4 = v_1 == v_4;
    const auto is_equal_5 = v_1 == v_5;
    const auto is_equal_6 = v_1 == v_6;

    ASSERT_FALSE(is_equal_2) << ss.str();
    ASSERT_FALSE(is_equal_3) << ss.str();
    ASSERT_FALSE(is_equal_4) << ss.str();
    ASSERT_FALSE(is_equal_5) << ss.str();
    ASSERT_FALSE(is_equal_6) << ss.str();
}

TEST_F(VectorTest, testVectorComponentwiseMinMax) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << x2 << ' ' << y2 << ' ' << z2 << '\n';

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    Vec3<double> v_1{ x1, y1, z1 };
    Vec3<double> v_2{ x2, y2, z2 };

    const Vec3<double> v_max{ x_max, y_max, z_max };
    const Vec3<double> v_min{ x_min, y_min, z_min };

    v_1.calculate_componentwise_maximum(v_2);
    const auto is_equal_max = v_1 == v_max;
    ASSERT_TRUE(is_equal_max) << ss.str();

    v_1 = Vec3<double>(x1, y1, z1);

    v_1.calculate_componentwise_minimum(v_2);
    const auto is_equal_min = v_1 == v_min;
    ASSERT_TRUE(is_equal_min) << ss.str();
}

TEST_F(VectorTest, testVectorMinMax) {
    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const Vec3<double> v{ x, y, z };

    const auto max = v.get_maximum();
    const auto min = v.get_minimum();

    const auto is_there_max = (x == max) || (y == max) || (z == max);
    const auto is_bound_max = (x <= max) || (y <= max) || (z <= max);

    const auto is_equal_max = is_there_max && is_bound_max;

    ASSERT_TRUE(is_equal_max) << ss.str();

    const auto is_there_min = (x == min) || (y == min) || (z == min);
    const auto is_bound_min = (x >= min) || (y >= min) || (z >= min);

    const auto is_equal_min = is_there_min && is_bound_min;

    ASSERT_TRUE(is_equal_min) << ss.str();
}

TEST_F(VectorTest, testVectorComponentwiseFloor) {
    const auto x = std::abs(get_random_vector_element()) + eps;
    const auto y = std::abs(get_random_vector_element()) + eps;
    const auto z = std::abs(get_random_vector_element()) + eps;

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const auto x_floor = floor(x);
    const auto y_floor = floor(y);
    const auto z_floor = floor(z);

    const auto x_size_t = static_cast<size_t>(x_floor);
    const auto y_size_t = static_cast<size_t>(y_floor);
    const auto z_size_t = static_cast<size_t>(z_floor);

    const Vec3<double> v{ x, y, z };
    const Vec3<size_t> v_size_t{ x_size_t, y_size_t, z_size_t };

    const Vec3<size_t> v_floored = v.floor_componentwise();

    const auto is_equal = v_size_t == v_floored;

    ASSERT_TRUE(is_equal) << ss.str();
}

TEST_F(VectorTest, testVectorComponentwiseFloorExceptionX) {
    auto x = std::abs(get_random_vector_element()) + eps;
    const auto y = std::abs(get_random_vector_element()) + eps;
    const auto z = std::abs(get_random_vector_element()) + eps;

    x *= -1.0;

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const Vec3<double> v{ x, y, z };
    ASSERT_THROW(Vec3<size_t> v_floored = v.floor_componentwise(), RelearnException) << ss.str();
}

TEST_F(VectorTest, testVectorComponentwiseFloorExceptionY) {
    const auto x = std::abs(get_random_vector_element()) + eps;
    auto y = std::abs(get_random_vector_element()) + eps;
    const auto z = std::abs(get_random_vector_element()) + eps;

    y *= -1.0;

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const Vec3<double> v{ x, y, z };
    ASSERT_THROW(Vec3<size_t> v_floored = v.floor_componentwise(), RelearnException) << ss.str();
}

TEST_F(VectorTest, testVectorComponentwiseFloorExceptionZ) {
    const auto x = std::abs(get_random_vector_element()) + eps;
    const auto y = std::abs(get_random_vector_element()) + eps;
    auto z = std::abs(get_random_vector_element()) + eps;

    z *= -1.0;

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const Vec3<double> v{ x, y, z };
    ASSERT_THROW(Vec3<size_t> v_floored = v.floor_componentwise(), RelearnException) << ss.str();
}

TEST_F(VectorTest, testVectorNorm) {
    std::uniform_real_distribution<double> urd_p(1.0, 10.1);

    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    const auto p = urd_p(mt);

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;
    ss << "p is: " << p;

    const Vec3<double> v{ x, y, z };

    const auto v_normed = v.calculate_p_norm(p);

    const auto x_p = pow(abs(x), p);
    const auto y_p = pow(abs(y), p);
    const auto z_p = pow(abs(z), p);

    const auto sum = x_p + y_p + z_p;
    const auto p_norm = pow(sum, 1.0 / p);

    ASSERT_NEAR(v_normed, p_norm, eps) << ss.str();
}

TEST_F(VectorTest, testVectorNorm2) {
    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const Vec3<double> v{ x, y, z };

    const auto v_normed = v.calculate_2_norm();
    const auto v_normed_squared = v.calculate_squared_2_norm();

    const auto xx = x * x;
    const auto yy = y * y;
    const auto zz = z * z;

    const auto sum = xx + yy + zz;
    const auto root = sqrt(sum);

    ASSERT_NEAR(v_normed, root, eps) << ss.str();
    ASSERT_NEAR(v_normed_squared, sum, eps) << ss.str();
}

TEST_F(VectorTest, testVectorNormException) {
    std::uniform_real_distribution<double> urd_bad_p(-10.0, 1.0);

    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    const auto p = urd_bad_p(mt);

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;
    ss << "p is: " << p;

    const Vec3<double> v{ x, y, z };

    ASSERT_THROW(auto v_normed = v.calculate_p_norm(p), RelearnException) << ss.str();
}

TEST_F(VectorTest, testVectorRound) {
    std::uniform_real_distribution<double> urd_multiple(1.0, 10.1);

    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    const auto multiple = urd_multiple(mt);

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;
    ss << "multiple is: " << multiple;

    const auto x_div = x / multiple;
    const auto y_div = y / multiple;
    const auto z_div = z / multiple;

    Vec3<double> v{ x, y, z };

    v.round_to_larger_multiple(multiple);

    const auto x_rounded = v.get_x();
    const auto y_rounded = v.get_y();
    const auto z_rounded = v.get_z();

    const auto x_rounded_div = x_rounded / multiple;
    const auto y_rounded_div = y_rounded / multiple;
    const auto z_rounded_div = z_rounded / multiple;

    const auto x_in_expected = x_div <= x_rounded_div && x_rounded_div <= x_div + 1;
    const auto y_in_expected = y_div <= y_rounded_div && y_rounded_div <= y_div + 1;
    const auto z_in_expected = z_div <= z_rounded_div && z_rounded_div <= z_div + 1;

    const auto x_rounded_div_rounded = round(x_rounded_div);
    const auto y_rounded_div_rounded = round(y_rounded_div);
    const auto z_rounded_div_rounded = round(z_rounded_div);

    ASSERT_TRUE(x_in_expected) << ss.str();
    ASSERT_TRUE(y_in_expected) << ss.str();
    ASSERT_TRUE(z_in_expected) << ss.str();

    ASSERT_NEAR(x_rounded_div, x_rounded_div_rounded, eps) << ss.str();
    ASSERT_NEAR(y_rounded_div, y_rounded_div_rounded, eps) << ss.str();
    ASSERT_NEAR(z_rounded_div, z_rounded_div_rounded, eps) << ss.str();
}

TEST_F(VectorTest, testVectorOrderEqual) {
    std::uniform_real_distribution<double> urd_offset(1.0, 10.1);

    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const Vec3<double> v_1{ x, y, z };
    const Vec3<double> v_2{ x, y, z };

    const auto is_smaller = v_1 < v_2;
    ASSERT_FALSE(is_smaller) << ss.str();
}

TEST_F(VectorTest, testVectorOrderSmallerX) {
    std::uniform_real_distribution<double> urd_offset(1.0, 10.1);

    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    const auto offset = urd_offset(mt);

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;
    ss << "offset is: " << offset;

    const Vec3<double> v_1{ x, y, z };
    const Vec3<double> v_2{ x + offset, y, z };

    const auto is_smaller_1_2 = v_1 < v_2;
    ASSERT_TRUE(is_smaller_1_2) << ss.str();

    const auto is_smaller_2_1 = v_2 < v_1;
    ASSERT_FALSE(is_smaller_2_1) << ss.str();
}

TEST_F(VectorTest, testVectorOrderSmallerY) {
    std::uniform_real_distribution<double> urd_offset(1.0, 10.1);

    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    const auto offset = urd_offset(mt);

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;
    ss << "offset is: " << offset;

    const Vec3<double> v_1{ x, y, z };
    const Vec3<double> v_2{ x, y + offset, z };

    const auto is_smaller_1_2 = v_1 < v_2;
    ASSERT_TRUE(is_smaller_1_2) << ss.str();

    const auto is_smaller_2_1 = v_2 < v_1;
    ASSERT_FALSE(is_smaller_2_1) << ss.str();
}

TEST_F(VectorTest, testVectorOrderSmallerZ) {
    std::uniform_real_distribution<double> urd_offset(1.0, 10.1);

    const auto x = get_random_vector_element();
    const auto y = get_random_vector_element();
    const auto z = get_random_vector_element();

    const auto offset = urd_offset(mt);

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;
    ss << "offset is: " << offset;

    const Vec3<double> v_1{ x, y, z };
    const Vec3<double> v_2{ x, y, z + offset };

    const auto is_smaller_1_2 = v_1 < v_2;
    ASSERT_TRUE(is_smaller_1_2) << ss.str();

    const auto is_smaller_2_1 = v_2 < v_1;
    ASSERT_FALSE(is_smaller_2_1) << ss.str();
}

TEST_F(VectorTest, testVectorOrder) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << x2 << ' ' << y2 << ' ' << z2 << '\n';

    const Vec3<double> v_1{ x1, y1, z1 };
    const Vec3<double> v_2{ x2, y2, z2 };

    const auto is_smaller_1_2 = v_1 < v_2;
    const auto is_smaller_2_1 = v_2 < v_1;

    const auto connected = is_smaller_1_2 || is_smaller_2_1;

    if (connected) {
        ASSERT_FALSE(is_smaller_1_2 && is_smaller_2_1) << ss.str();
    } else {
        const auto is_equal = v_1 == v_2;
        ASSERT_TRUE(is_equal) << ss.str();
    }
}

TEST_F(VectorTest, testVectorCast) {
    const auto x = std::abs(get_random_vector_element());
    const auto y = std::abs(get_random_vector_element());
    const auto z = std::abs(get_random_vector_element());

    std::stringstream ss{};
    ss << x << ' ' << y << ' ' << z;

    const auto x_size_t = static_cast<size_t>(x);
    const auto y_size_t = static_cast<size_t>(y);
    const auto z_size_t = static_cast<size_t>(z);

    const Vec3<double> v{ x, y, z };
    const Vec3<size_t> v_casted = static_cast<Vec3<size_t>>(v);
    const Vec3<size_t> v_size_t{ x_size_t, y_size_t, z_size_t };

    const auto is_equal = v_casted == v_size_t;
    ASSERT_TRUE(is_equal) << ss.str();
}

TEST_F(VectorTest, testVectorInBoxNoThrow) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    std::stringstream ss{};
    ss << x1 << ' ' << y1 << ' ' << z1 << '\n';
    ss << x2 << ' ' << y2 << ' ' << z2 << '\n';

    const Vec3<double> v{ x1, y1, z1 };
    const Vec3<double> w{ x2, y2, z2 };

    ASSERT_NO_THROW(w.check_in_box(v, v)) << ss.str();
}

TEST_F(VectorTest, testVectorInBoxTrue) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const Vec3<double> v_min{ x_min, y_min, z_min };
    const Vec3<double> v_max{ x_max, y_max, z_max };

    std::uniform_real_distribution<double> urd_x(x_min, x_max);
    std::uniform_real_distribution<double> urd_y(y_min, y_max);
    std::uniform_real_distribution<double> urd_z(z_min, z_max);

    const auto x3 = urd_x(mt);
    const auto y3 = urd_y(mt);
    const auto z3 = urd_z(mt);

    std::stringstream ss{};
    ss << x_min << ' ' << y_min << ' ' << z_min << '\n';
    ss << x_max << ' ' << y_max << ' ' << z_max << '\n';
    ss << x3 << ' ' << y3 << ' ' << z3;

    const Vec3<double> v{ x3, y3, z3 };

    const auto is_in_box = v.check_in_box(v_min, v_max);
    ASSERT_TRUE(is_in_box) << ss.str();
}

TEST_F(VectorTest, testVectorInBoxFalseXlarge) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const Vec3<double> v_min{ x_min, y_min, z_min };
    const Vec3<double> v_max{ x_max, y_max, z_max };

    std::uniform_real_distribution<double> urd_x(x_min, x_max);
    std::uniform_real_distribution<double> urd_y(y_min, y_max);
    std::uniform_real_distribution<double> urd_z(z_min, z_max);

    const auto x3 = urd_x(mt) + x_max - x_min;
    const auto y3 = urd_y(mt);
    const auto z3 = urd_z(mt);

    std::stringstream ss{};
    ss << x_min << ' ' << y_min << ' ' << z_min << '\n';
    ss << x_max << ' ' << y_max << ' ' << z_max << '\n';
    ss << x3 << ' ' << y3 << ' ' << z3;

    const Vec3<double> v{ x3, y3, z3 };

    const auto is_in_box = v.check_in_box(v_min, v_max);
    ASSERT_FALSE(is_in_box) << ss.str();
}

TEST_F(VectorTest, testVectorInBoxFalseYlarge) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const Vec3<double> v_min{ x_min, y_min, z_min };
    const Vec3<double> v_max{ x_max, y_max, z_max };

    std::uniform_real_distribution<double> urd_x(x_min, x_max);
    std::uniform_real_distribution<double> urd_y(y_min, y_max);
    std::uniform_real_distribution<double> urd_z(z_min, z_max);

    const auto x3 = urd_x(mt);
    const auto y3 = urd_y(mt) + y_max - y_min;
    const auto z3 = urd_z(mt);

    std::stringstream ss{};
    ss << x_min << ' ' << y_min << ' ' << z_min << '\n';
    ss << x_max << ' ' << y_max << ' ' << z_max << '\n';
    ss << x3 << ' ' << y3 << ' ' << z3;

    const Vec3<double> v{ x3, y3, z3 };

    const auto is_in_box = v.check_in_box(v_min, v_max);
    ASSERT_FALSE(is_in_box) << ss.str();
}

TEST_F(VectorTest, testVectorInBoxFalseZlarge) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const Vec3<double> v_min{ x_min, y_min, z_min };
    const Vec3<double> v_max{ x_max, y_max, z_max };

    std::uniform_real_distribution<double> urd_x(x_min, x_max);
    std::uniform_real_distribution<double> urd_y(y_min, y_max);
    std::uniform_real_distribution<double> urd_z(z_min, z_max);

    const auto x3 = urd_x(mt);
    const auto y3 = urd_y(mt);
    const auto z3 = urd_z(mt) + z_max - z_min;

    std::stringstream ss{};
    ss << x_min << ' ' << y_min << ' ' << z_min << '\n';
    ss << x_max << ' ' << y_max << ' ' << z_max << '\n';
    ss << x3 << ' ' << y3 << ' ' << z3;

    const Vec3<double> v{ x3, y3, z3 };

    const auto is_in_box = v.check_in_box(v_min, v_max);
    ASSERT_FALSE(is_in_box) << ss.str();
}

TEST_F(VectorTest, testVectorInBoxFalseXsmall) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const Vec3<double> v_min{ x_min, y_min, z_min };
    const Vec3<double> v_max{ x_max, y_max, z_max };

    std::uniform_real_distribution<double> urd_x(x_min, x_max);
    std::uniform_real_distribution<double> urd_y(y_min, y_max);
    std::uniform_real_distribution<double> urd_z(z_min, z_max);

    const auto x3 = urd_x(mt) - x_max + x_min;
    const auto y3 = urd_y(mt);
    const auto z3 = urd_z(mt);

    std::stringstream ss{};
    ss << x_min << ' ' << y_min << ' ' << z_min << '\n';
    ss << x_max << ' ' << y_max << ' ' << z_max << '\n';
    ss << x3 << ' ' << y3 << ' ' << z3;

    const Vec3<double> v{ x3, y3, z3 };

    const auto is_in_box = v.check_in_box(v_min, v_max);
    ASSERT_FALSE(is_in_box) << ss.str();
}

TEST_F(VectorTest, testVectorInBoxFalseYsmall) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const Vec3<double> v_min{ x_min, y_min, z_min };
    const Vec3<double> v_max{ x_max, y_max, z_max };

    std::uniform_real_distribution<double> urd_x(x_min, x_max);
    std::uniform_real_distribution<double> urd_y(y_min, y_max);
    std::uniform_real_distribution<double> urd_z(z_min, z_max);

    const auto x3 = urd_x(mt);
    const auto y3 = urd_y(mt) - y_max + y_min;
    const auto z3 = urd_z(mt);

    std::stringstream ss{};
    ss << x_min << ' ' << y_min << ' ' << z_min << '\n';
    ss << x_max << ' ' << y_max << ' ' << z_max << '\n';
    ss << x3 << ' ' << y3 << ' ' << z3;

    const Vec3<double> v{ x3, y3, z3 };

    const auto is_in_box = v.check_in_box(v_min, v_max);
    ASSERT_FALSE(is_in_box) << ss.str();
}

TEST_F(VectorTest, testVectorInBoxFalseZsmall) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const Vec3<double> v_min{ x_min, y_min, z_min };
    const Vec3<double> v_max{ x_max, y_max, z_max };

    std::uniform_real_distribution<double> urd_x(x_min, x_max);
    std::uniform_real_distribution<double> urd_y(y_min, y_max);
    std::uniform_real_distribution<double> urd_z(z_min, z_max);

    const auto x3 = urd_x(mt);
    const auto y3 = urd_y(mt);
    const auto z3 = urd_z(mt) - z_max + z_min;

    std::stringstream ss{};
    ss << x_min << ' ' << y_min << ' ' << z_min << '\n';
    ss << x_max << ' ' << y_max << ' ' << z_max << '\n';
    ss << x3 << ' ' << y3 << ' ' << z3;

    const Vec3<double> v{ x3, y3, z3 };

    const auto is_in_box = v.check_in_box(v_min, v_max);
    ASSERT_FALSE(is_in_box) << ss.str();
}

TEST_F(VectorTest, testVectorInBoxExceptionX) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const Vec3<double> v_min{ x_max, y_min, z_min };
    const Vec3<double> v_max{ x_min, y_max, z_max };

    std::uniform_real_distribution<double> urd_x(x_min, x_max);
    std::uniform_real_distribution<double> urd_y(y_min, y_max);
    std::uniform_real_distribution<double> urd_z(z_min, z_max);

    const auto x3 = urd_x(mt);
    const auto y3 = urd_y(mt);
    const auto z3 = urd_z(mt);

    std::stringstream ss{};
    ss << x_min << ' ' << y_min << ' ' << z_min << '\n';
    ss << x_max << ' ' << y_max << ' ' << z_max << '\n';
    ss << x3 << ' ' << y3 << ' ' << z3;

    const Vec3<double> v{ x3, y3, z3 };

    ASSERT_THROW(const auto is_in_box = v.check_in_box(v_min, v_max), RelearnException) << ss.str();
}

TEST_F(VectorTest, testVectorInBoxExceptionY) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const Vec3<double> v_min{ x_min, y_max, z_min };
    const Vec3<double> v_max{ x_max, y_min, z_max };

    std::uniform_real_distribution<double> urd_x(x_min, x_max);
    std::uniform_real_distribution<double> urd_y(y_min, y_max);
    std::uniform_real_distribution<double> urd_z(z_min, z_max);

    const auto x3 = urd_x(mt);
    const auto y3 = urd_y(mt);
    const auto z3 = urd_z(mt);

    std::stringstream ss{};
    ss << x_min << ' ' << y_min << ' ' << z_min << '\n';
    ss << x_max << ' ' << y_max << ' ' << z_max << '\n';
    ss << x3 << ' ' << y3 << ' ' << z3;

    const Vec3<double> v{ x3, y3, z3 };

    ASSERT_THROW(const auto is_in_box = v.check_in_box(v_min, v_max), RelearnException) << ss.str();
}

TEST_F(VectorTest, testVectorInBoxExceptionZ) {
    const auto x1 = get_random_vector_element();
    const auto y1 = get_random_vector_element();
    const auto z1 = get_random_vector_element();

    const auto x2 = get_random_vector_element();
    const auto y2 = get_random_vector_element();
    const auto z2 = get_random_vector_element();

    const auto x_max = std::max(x1, x2);
    const auto x_min = std::min(x1, x2);
    const auto y_max = std::max(y1, y2);
    const auto y_min = std::min(y1, y2);
    const auto z_max = std::max(z1, z2);
    const auto z_min = std::min(z1, z2);

    const Vec3<double> v_min{ x_min, y_min, z_max };
    const Vec3<double> v_max{ x_max, y_max, z_min };

    std::uniform_real_distribution<double> urd_x(x_min, x_max);
    std::uniform_real_distribution<double> urd_y(y_min, y_max);
    std::uniform_real_distribution<double> urd_z(z_min, z_max);

    const auto x3 = urd_x(mt);
    const auto y3 = urd_y(mt);
    const auto z3 = urd_z(mt);

    std::stringstream ss{};
    ss << x_min << ' ' << y_min << ' ' << z_min << '\n';
    ss << x_max << ' ' << y_max << ' ' << z_max << '\n';
    ss << x3 << ' ' << y3 << ' ' << z3;

    const Vec3<double> v{ x3, y3, z3 };

    ASSERT_THROW(const auto is_in_box = v.check_in_box(v_min, v_max), RelearnException) << ss.str();
}
