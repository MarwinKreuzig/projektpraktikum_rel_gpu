#include "../googletest/include/gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <random>

#include "RelearnTest.hpp"

#include "../source/structure/SpaceFillingCurve.h"
#include "../source/util/RelearnException.h"

TEST_F(SpaceFillingCurveTest, test_morton_constructor) {
    Morton morton{};

    std::uniform_int_distribution<unsigned short> uid_refinement(0, Constants::max_lvl_subdomains);

    for (auto i = 0; i < iterations; i++) {
        const auto ref_level = static_cast<uint8_t>(uid_refinement(mt));
        morton.set_refinement_level(ref_level);
        const auto res = morton.get_refinement_level();

        ASSERT_EQ(ref_level, res);
    }
}

TEST_F(SpaceFillingCurveTest, test_morton_translation_brute_force) {
    Morton morton{};

    std::uniform_int_distribution<unsigned short> uid_refinement(0, 5);

    for (auto i = 0; i < iterations; i++) {
        const auto ref_level = static_cast<uint8_t>(uid_refinement(mt));
        morton.set_refinement_level(ref_level);

        const size_t num_boxes_per_dimension = size_t(1) << ref_level;
        const size_t total_num_boxes = num_boxes_per_dimension * num_boxes_per_dimension * num_boxes_per_dimension;

        for (size_t x = 0; x < num_boxes_per_dimension; x++) {
            for (size_t y = 0; y < num_boxes_per_dimension; y++) {
                for (size_t z = 0; z < num_boxes_per_dimension; z++) {
                    Vec3s index3d{ x, y, z };

                    const auto index1d = morton.map_3d_to_1d(index3d);
                    const auto res = morton.map_1d_to_3d(index1d);

                    ASSERT_EQ(index3d, res);
                    ASSERT_LT(index1d, total_num_boxes);
                }
            }
        }
    }
}

TEST_F(SpaceFillingCurveTest, test_morton_translation_stochastic) {
    Morton morton{};

    std::uniform_int_distribution<unsigned short> uid_refinement(5, Constants::max_lvl_subdomains);

    for (auto i = 0; i < iterations; i++) {
        const auto ref_level = static_cast<uint8_t>(uid_refinement(mt));
        morton.set_refinement_level(ref_level);

        const size_t num_boxes_per_dimension = size_t(1) << ref_level;
        const size_t total_num_boxes = num_boxes_per_dimension * num_boxes_per_dimension * num_boxes_per_dimension;

        std::uniform_int_distribution<size_t> uid_idx(0, num_boxes_per_dimension - 1);

        for (auto rep = 0; rep < 1000; rep++) {
            const size_t x = uid_idx(mt);
            const size_t y = uid_idx(mt);
            const size_t z = uid_idx(mt);

            Morton::BoxCoordinates index3d{ x, y, z };

            const auto index1d = morton.map_3d_to_1d(index3d);
            const auto res = morton.map_1d_to_3d(index1d);

            ASSERT_EQ(index3d, res) << ref_level << x << y << z;
            ASSERT_LT(index1d, total_num_boxes) << ref_level << x << y << z;
        }
    }
}

TEST_F(SpaceFillingCurveTest, test_space_filling_curve_morton_constructor) {
    SpaceFillingCurve<Morton> sfc{};

    std::uniform_int_distribution<unsigned short> uid_refinement(0, Constants::max_lvl_subdomains);

    for (auto i = 0; i < iterations; i++) {
        const auto ref_level = static_cast<uint8_t>(uid_refinement(mt));
        sfc.set_refinement_level(ref_level);
        const auto res = sfc.get_refinement_level();

        ASSERT_EQ(ref_level, res);
    }

    for (auto i = 0; i < iterations; i++) {
        const auto ref_level = static_cast<uint8_t>(uid_refinement(mt));
        SpaceFillingCurve<Morton> sfc2{ ref_level };

        const auto res = sfc2.get_refinement_level();

        ASSERT_EQ(ref_level, res);

        const auto ref_level_2 = static_cast<uint8_t>(uid_refinement(mt));
        sfc2.set_refinement_level(ref_level_2);
        const auto res2 = sfc2.get_refinement_level();

        ASSERT_EQ(ref_level_2, res2);
    }
}

TEST_F(SpaceFillingCurveTest, test_space_filling_curve_morton_constructor_exception) {
    std::uniform_int_distribution<unsigned short> uid_refinement(Constants::max_lvl_subdomains + 1, std::numeric_limits<uint8_t>::max());

    for (auto i = 0; i < iterations; i++) {
        const auto ref_level = static_cast<uint8_t>(uid_refinement(mt));

        ASSERT_THROW(SpaceFillingCurve<Morton> sfc(ref_level);, RelearnException);
    }
}

TEST_F(SpaceFillingCurveTest, test_space_filling_curve_morton_set_refinement_exception) {
    std::uniform_int_distribution<unsigned short> uid_refinement(Constants::max_lvl_subdomains + 1, std::numeric_limits<uint8_t>::max());
    SpaceFillingCurve<Morton> sfc{};

    for (auto i = 0; i < iterations; i++) {
        const auto ref_level = static_cast<uint8_t>(uid_refinement(mt));

        ASSERT_THROW(sfc.set_refinement_level(ref_level);, RelearnException);
    }
}

TEST_F(SpaceFillingCurveTest, test_space_filling_curve_morton_translation_brute_force) {
    SpaceFillingCurve<Morton> sfc{};

    std::uniform_int_distribution<unsigned short> uid_refinement(0, 5);

    for (auto i = 0; i < iterations; i++) {
        const auto ref_level = static_cast<uint8_t>(uid_refinement(mt));
        sfc.set_refinement_level(ref_level);

        const size_t num_boxes_per_dimension = size_t(1) << ref_level;
        const size_t total_num_boxes = num_boxes_per_dimension * num_boxes_per_dimension * num_boxes_per_dimension;

        for (size_t x = 0; x < num_boxes_per_dimension; x++) {
            for (size_t y = 0; y < num_boxes_per_dimension; y++) {
                for (size_t z = 0; z < num_boxes_per_dimension; z++) {
                    Vec3s index3d{ x, y, z };

                    const auto index1d = sfc.map_3d_to_1d(index3d);
                    const auto res = sfc.map_1d_to_3d(index1d);

                    ASSERT_EQ(index3d, res);
                    ASSERT_LT(index1d, total_num_boxes);
                }
            }
        }
    }
}

TEST_F(SpaceFillingCurveTest, test_space_filling_curve_morton_translation_stochastic) {
    SpaceFillingCurve<Morton> sfc{};

    std::uniform_int_distribution<unsigned short> uid_refinement(5, Constants::max_lvl_subdomains);

    for (auto i = 0; i < iterations; i++) {
        const auto ref_level = static_cast<uint8_t>(uid_refinement(mt));
        sfc.set_refinement_level(ref_level);

        const size_t num_boxes_per_dimension = size_t(1) << ref_level;
        const size_t total_num_boxes = num_boxes_per_dimension * num_boxes_per_dimension * num_boxes_per_dimension;

        std::uniform_int_distribution<size_t> uid_idx(0, num_boxes_per_dimension - 1);

        for (auto rep = 0; rep < 1000; rep++) {
            const size_t x = uid_idx(mt);
            const size_t y = uid_idx(mt);
            const size_t z = uid_idx(mt);

            Vec3s index3d{ x, y, z };

            const auto index1d = sfc.map_3d_to_1d(index3d);
            const auto res = sfc.map_1d_to_3d(index1d);

            ASSERT_EQ(index3d, res) << ref_level << x << y << z;
            ASSERT_LT(index1d, total_num_boxes) << ref_level << x << y << z;
        }
    }
}
