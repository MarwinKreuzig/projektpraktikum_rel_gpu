#include "../googletest/include/gtest/gtest.h"

#include <random>

#include "../source/Vec3.h"


TEST(TestVector, test_vector_constructor_empty) {
	for (auto i = 0; i < 10; i++) {
		Vec3<double> v{ };

		EXPECT_EQ(0.0, v.x);
		EXPECT_EQ(0.0, v.y);
		EXPECT_EQ(0.0, v.z);
	}
}

TEST(TestVector, test_vector_constructor_one) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto val = urd(mt);

		Vec3<double> v{ val };

		EXPECT_EQ(val, v.x);
		EXPECT_EQ(val, v.y);
		EXPECT_EQ(val, v.z);
	}
}

TEST(TestVector, test_vector_constructor_three) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x = urd(mt);
		auto y = urd(mt);
		auto z = urd(mt);

		Vec3<double> v{ x, y, z };

		EXPECT_EQ(x, v.x);
		EXPECT_EQ(y, v.y);
		EXPECT_EQ(z, v.z);
	}
}

TEST(TestVector, test_vector_copy_construct) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x = urd(mt);
		auto y = urd(mt);
		auto z = urd(mt);

		Vec3<double> v{ x, y, z };

		Vec3<double> w = v;

		EXPECT_EQ(x, w.x);
		EXPECT_EQ(y, w.y);
		EXPECT_EQ(z, w.z);
	}
}

TEST(TestVector, test_vector_copy_assign) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x = urd(mt);
		auto y = urd(mt);
		auto z = urd(mt);

		Vec3<double> v{ x, y, z };

		Vec3<double> w{ x + 1.0, y + 1.0, z + 1.0 };
		w = v;

		EXPECT_EQ(x, w.x);
		EXPECT_EQ(y, w.y);
		EXPECT_EQ(z, w.z);
	}
}

TEST(TestVector, test_vector_move_construct) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x = urd(mt);
		auto y = urd(mt);
		auto z = urd(mt);

		Vec3<double> v{ x, y, z };

		Vec3<double> w = std::move(v);

		EXPECT_EQ(x, w.x);
		EXPECT_EQ(y, w.y);
		EXPECT_EQ(z, w.z);
	}
}

TEST(TestVector, test_vector_move_assign) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x = urd(mt);
		auto y = urd(mt);
		auto z = urd(mt);

		Vec3<double> v{ x, y, z };

		Vec3<double> w{ x + 1.0, y + 1.0, z + 1.0 };
		w = std::move(v);

		EXPECT_EQ(x, w.x);
		EXPECT_EQ(y, w.y);
		EXPECT_EQ(z, w.z);
	}
}

TEST(TestVector, test_vector_operator_index_read) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x = urd(mt);
		auto y = urd(mt);
		auto z = urd(mt);

		Vec3<double> v{ x, y, z };

		EXPECT_EQ(x, v[0]);
		EXPECT_EQ(y, v[1]);
		EXPECT_EQ(z, v[2]);
	}
}

TEST(TestVector, test_vector_operator_index_write) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x = urd(mt);
		auto y = urd(mt);
		auto z = urd(mt);

		Vec3<double> v{ };

		v[0] = x;
		v[1] = y;
		v[2] = z;

		EXPECT_EQ(x, v[0]);
		EXPECT_EQ(y, v[1]);
		EXPECT_EQ(z, v[2]);
	}
}

TEST(TestVector, test_vector_operator_plus_vec) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x1 = urd(mt);
		auto y1 = urd(mt);
		auto z1 = urd(mt);

		Vec3<double> v{ x1, y1, z1 };

		auto x2 = urd(mt);
		auto y2 = urd(mt);
		auto z2 = urd(mt);

		Vec3<double> w{ x2, y2, z2 };

		auto sum = v + w;

		EXPECT_EQ(x1, v.x);
		EXPECT_EQ(y1, v.y);
		EXPECT_EQ(z1, v.z);

		EXPECT_EQ(x2, w.x);
		EXPECT_EQ(y2, w.y);
		EXPECT_EQ(z2, w.z);

		EXPECT_EQ(x1 + x2, sum.x);
		EXPECT_EQ(y1 + y2, sum.y);
		EXPECT_EQ(z1 + z2, sum.z);
	}
}

TEST(TestVector, test_vector_operator_minus_vec) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x1 = urd(mt);
		auto y1 = urd(mt);
		auto z1 = urd(mt);

		Vec3<double> v{ x1, y1, z1 };

		auto x2 = urd(mt);
		auto y2 = urd(mt);
		auto z2 = urd(mt);

		Vec3<double> w{ x2, y2, z2 };

		auto diff = v - w;

		EXPECT_EQ(x1, v.x);
		EXPECT_EQ(y1, v.y);
		EXPECT_EQ(z1, v.z);

		EXPECT_EQ(x2, w.x);
		EXPECT_EQ(y2, w.y);
		EXPECT_EQ(z2, w.z);

		EXPECT_EQ(x1 - x2, diff.x);
		EXPECT_EQ(y1 - y2, diff.y);
		EXPECT_EQ(z1 - z2, diff.z);
	}
}

TEST(TestVector, test_vector_operator_plus_scalar) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x1 = urd(mt);
		auto y1 = urd(mt);
		auto z1 = urd(mt);

		Vec3<double> v{ x1, y1, z1 };

		auto scalar = urd(mt);
		auto scalar_copy = scalar;

		auto sum = v + scalar;

		EXPECT_EQ(x1, v.x);
		EXPECT_EQ(y1, v.y);
		EXPECT_EQ(z1, v.z);

		EXPECT_EQ(x1 + scalar, sum.x);
		EXPECT_EQ(y1 + scalar, sum.y);
		EXPECT_EQ(z1 + scalar, sum.z);

		EXPECT_EQ(scalar, scalar_copy);
	}
}

TEST(TestVector, test_vector_operator_mul_scalar) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x1 = urd(mt);
		auto y1 = urd(mt);
		auto z1 = urd(mt);

		Vec3<double> v{ x1, y1, z1 };

		auto scalar = urd(mt);
		auto scalar_copy = scalar;

		auto prod = v * scalar;

		EXPECT_EQ(x1, v.x);
		EXPECT_EQ(y1, v.y);
		EXPECT_EQ(z1, v.z);

		EXPECT_EQ(x1 * scalar, prod.x);
		EXPECT_EQ(y1 * scalar, prod.y);
		EXPECT_EQ(z1 * scalar, prod.z);

		EXPECT_EQ(scalar, scalar_copy);
	}
}

TEST(TestVector, test_vector_operator_plus_assign_scalar) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x1 = urd(mt);
		auto y1 = urd(mt);
		auto z1 = urd(mt);

		Vec3<double> v{ x1, y1, z1 };

		auto scalar = urd(mt);
		auto scalar_copy = scalar;

		v += scalar;

		EXPECT_EQ(x1 + scalar, v.x);
		EXPECT_EQ(y1 + scalar, v.y);
		EXPECT_EQ(z1 + scalar, v.z);

		EXPECT_EQ(scalar, scalar_copy);
	}
}

TEST(TestVector, test_vector_operator_mul_assign_scalar) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x1 = urd(mt);
		auto y1 = urd(mt);
		auto z1 = urd(mt);

		Vec3<double> v{ x1, y1, z1 };

		auto scalar = urd(mt);
		auto scalar_copy = scalar;

		v *= scalar;

		EXPECT_EQ(x1 * scalar, v.x);
		EXPECT_EQ(y1 * scalar, v.y);
		EXPECT_EQ(z1 * scalar, v.z);

		EXPECT_EQ(scalar, scalar_copy);
	}
}

TEST(TestVector, test_vector_volume) {
	std::mt19937 mt;
	std::uniform_real_distribution<double> urd(-100.0, 100.0);

	mt.seed(rand());

	for (auto i = 0; i < 10; i++) {
		auto x = urd(mt);
		auto y = urd(mt);
		auto z = urd(mt);

		Vec3<double> v{ x, y, z };

		auto volume = x * y * z;

		EXPECT_EQ(volume, v.get_volume());

		EXPECT_EQ(x, v.x);
		EXPECT_EQ(y, v.y);
		EXPECT_EQ(z, v.z);
	}
}

