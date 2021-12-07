#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/structure/SpaceFillingCurve.h"
#include "../source/util/RelearnException.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <sstream>

TEST_F(SpaceFillingCurveTest, testMortonConstructor) {
    Morton morton{};

    for (auto refinement_level = 0; refinement_level < max_refinement_level; refinement_level++) {
        std::stringstream ss{};
        ss << "Refinement level: " << refinement_level;

        morton.set_refinement_level(refinement_level);
        const auto res = morton.get_random_refinement_level();

        ASSERT_EQ(refinement_level, res) << ss.str();
    }
}

TEST_F(SpaceFillingCurveTest, testMortonTranslationBruteForce) {
    Morton morton{};

    for (auto refinement_level = 0; refinement_level < small_refinement_level; refinement_level++) {
        morton.set_refinement_level(refinement_level);

        std::stringstream ss{};
        ss << "Refinement level: " << refinement_level;

        const size_t num_boxes_per_dimension = size_t(1) << refinement_level;
        const size_t total_num_boxes = num_boxes_per_dimension * num_boxes_per_dimension * num_boxes_per_dimension;

        for (size_t x = 0; x < num_boxes_per_dimension; x++) {
            for (size_t y = 0; y < num_boxes_per_dimension; y++) {
                for (size_t z = 0; z < num_boxes_per_dimension; z++) {
                    Vec3s index3d{ x, y, z };

                    const auto index1d = morton.map_3d_to_1d(index3d);
                    const auto res = morton.map_1d_to_3d(index1d);

                    ASSERT_EQ(index3d, res) << ss.str();
                    ASSERT_LT(index1d, total_num_boxes) << ss.str();
                }
            }
        }
    }
}

TEST_F(SpaceFillingCurveTest, testMortonTranslationStochastic) {
    Morton morton{};
    std::stringstream ss{};

    for (auto i = 0; i < iterations; i++) {
        const auto refinement_level = get_large_refinement_level();
        morton.set_refinement_level(refinement_level);

        const size_t num_boxes_per_dimension = size_t(1) << refinement_level;
        const size_t total_num_boxes = num_boxes_per_dimension * num_boxes_per_dimension * num_boxes_per_dimension;

        std::uniform_int_distribution<size_t> uid_idx(0, num_boxes_per_dimension - 1);

        for (auto rep = 0; rep < 1000; rep++) {
            const size_t x = uid_idx(mt);
            const size_t y = uid_idx(mt);
            const size_t z = uid_idx(mt);

            Morton::BoxCoordinates index3d{ x, y, z };

            const auto index1d = morton.map_3d_to_1d(index3d);
            const auto res = morton.map_1d_to_3d(index1d);

            ss.clear();
            ss << x << ' ' << y << ' ' << z << ' ';
            ss << "Refinement level: " << refinement_level;

            ASSERT_EQ(index3d, res) << ss.str();
            ASSERT_LT(index1d, total_num_boxes) << ss.str();
        }
    }
}

TEST_F(SpaceFillingCurveTest, testSpaceFillingCurveMortonConstructor) {
    SpaceFillingCurve<Morton> sfc{};

    for (auto refinement_level = 0; refinement_level < max_refinement_level; refinement_level++) {
        std::stringstream ss{};
        ss << "Refinement level: " << refinement_level;

        sfc.set_refinement_level(refinement_level);
        const auto res = sfc.get_random_refinement_level();

        ASSERT_EQ(refinement_level, res) << ss.str();
    }

    for (auto i = 0; i < iterations; i++) {
        const auto refinement_level = get_random_refinement_level();
        const auto refinement_level_2 = get_random_refinement_level();

        std::stringstream ss{};
        ss << "Refinement level: " << refinement_level << '\n';
        ss << "Refinement level 2: " << refinement_level;

        SpaceFillingCurve<Morton> sfc2{ refinement_level };

        const auto res = sfc2.get_random_refinement_level();

        ASSERT_EQ(refinement_level, res) << ss.str();

        sfc2.set_refinement_level(refinement_level_2);
        const auto res2 = sfc2.get_random_refinement_level();

        ASSERT_EQ(refinement_level_2, res2) << ss.str();
    }
}

TEST_F(SpaceFillingCurveTest, testSpaceFillingCurveMortonConstructorException) {
    for (auto i = 0; i < iterations; i++) {
        const auto refinement_level = get_random_refinement_level() + Constants::max_lvl_subdomains + 1;

        std::stringstream ss{};
        ss << "Refinement level: " << refinement_level;

        ASSERT_THROW(SpaceFillingCurve<Morton> sfc(refinement_level);, RelearnException) << ss.str();
    }
}

TEST_F(SpaceFillingCurveTest, testSpaceFillingCurveMortonSetRefinementException) {
    SpaceFillingCurve<Morton> sfc{};

    for (auto i = 0; i < iterations; i++) {
        const auto refinement_level = get_random_refinement_level() + Constants::max_lvl_subdomains + 1;

        std::stringstream ss{};
        ss << "Refinement level: " << refinement_level;

        ASSERT_THROW(sfc.set_refinement_level(refinement_level);, RelearnException) << ss.str();
    }
}

TEST_F(SpaceFillingCurveTest, testSpaceFillingCurveMortonTranslationBruteForce) {
    SpaceFillingCurve<Morton> sfc{};

    for (auto refinement_level = 0; refinement_level < small_refinement_level; refinement_level++) {
        sfc.set_refinement_level(refinement_level);

        std::stringstream ss{};
        ss << "Refinement level: " << refinement_level;

        const size_t num_boxes_per_dimension = size_t(1) << refinement_level;
        const size_t total_num_boxes = num_boxes_per_dimension * num_boxes_per_dimension * num_boxes_per_dimension;

        for (size_t x = 0; x < num_boxes_per_dimension; x++) {
            for (size_t y = 0; y < num_boxes_per_dimension; y++) {
                for (size_t z = 0; z < num_boxes_per_dimension; z++) {
                    Vec3s index3d{ x, y, z };

                    const auto index1d = sfc.map_3d_to_1d(index3d);
                    const auto res = sfc.map_1d_to_3d(index1d);

                    ASSERT_EQ(index3d, res) << ss.str();
                    ASSERT_LT(index1d, total_num_boxes) << ss.str();
                }
            }
        }
    }
}

TEST_F(SpaceFillingCurveTest, testSpaceFillingCurveMortonTranslationStochastic) {
    SpaceFillingCurve<Morton> sfc{};
    std::stringstream ss{};

    for (auto i = 0; i < iterations; i++) {
        const auto refinement_level = get_large_refinement_level();
        sfc.set_refinement_level(refinement_level);

        const size_t num_boxes_per_dimension = size_t(1) << refinement_level;
        const size_t total_num_boxes = num_boxes_per_dimension * num_boxes_per_dimension * num_boxes_per_dimension;

        std::uniform_int_distribution<size_t> uid_idx(0, num_boxes_per_dimension - 1);

        for (auto rep = 0; rep < 1000; rep++) {
            const size_t x = uid_idx(mt);
            const size_t y = uid_idx(mt);
            const size_t z = uid_idx(mt);

            Vec3s index3d{ x, y, z };

            const auto index1d = sfc.map_3d_to_1d(index3d);
            const auto res = sfc.map_1d_to_3d(index1d);

            ss.clear();
            ss << x << ' ' << y << ' ' << z << ' ';
            ss << "Refinement level: " << refinement_level;

            ASSERT_EQ(index3d, res) << ss.str();
            ASSERT_LT(index1d, total_num_boxes) << ss.str();
        }
    }
}
