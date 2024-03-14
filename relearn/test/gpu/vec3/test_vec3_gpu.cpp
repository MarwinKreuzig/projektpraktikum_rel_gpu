/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_vec3_gpu.h"
#include "../../../source/gpu/structure/GpuDataStructures.h"

#include <sstream>
#include <random>

TEST_F(Vec3TestGpu, testVectorGpuEmptyConstructor) {
    const gpu::Vec3d v{};

    EXPECT_NEAR(v.x, 0.0, 0.001);
    EXPECT_NEAR(v.y, 0.0, 0.001);
    EXPECT_NEAR(v.z, 0.0, 0.001);
}

TEST_F(Vec3TestGpu, testVectorGpuConstructor) {

    for (int i = 0; i < 100; ++i) {
        double x = (double)rand();
        double y = (double)rand();
        double z = (double)rand();
        const gpu::Vec3d v{ x, y, z };

        EXPECT_NEAR(v.x, x, 0.001);
        EXPECT_NEAR(v.y, y, 0.001);
        EXPECT_NEAR(v.z, z, 0.001);
    }
}

TEST_F(Vec3TestGpu, testVectorGpuEqual) {

    for (int i = 0; i < 100; ++i) {
        double x = (double)rand();
        double y = (double)rand();
        double z = (double)rand();
        const gpu::Vec3d v1{ x, y, z };
        const gpu::Vec3d v2{ x, y, z };

        EXPECT_TRUE(v1 == v2);
    }
}

TEST_F(Vec3TestGpu, testVectorGpuUnequal) {

    for (int i = 0; i < 100; ++i) {
        double x1 = (double)rand();

        double x2 = (double)rand();
        while (x1 == x2) {
            x2 = (double)rand();
        }

        double y = (double)rand();
        double z = (double)rand();

        const gpu::Vec3d v1{ x1, y, z };
        const gpu::Vec3d v2{ x2, y, z };

        EXPECT_TRUE(v1 != v2);
    }

    for (int i = 0; i < 100; ++i) {
        double x = (double)rand();
        double y1 = (double)rand();

        double y2 = (double)rand();
        while (y1 == y2) {
            y2 = (double)rand();
        }

        double z = (double)rand();

        const gpu::Vec3d v1{ x, y1, z };
        const gpu::Vec3d v2{ x, y2, z };

        EXPECT_TRUE(v1 != v2);
    }

    for (int i = 0; i < 100; ++i) {
        double x = (double)rand();
        double y = (double)rand();
        double z1 = (double)rand();

        double z2 = (double)rand();
        while (z1 == z2) {
            z2 = (double)rand();
        }

        const gpu::Vec3d v1{ x, y, z1 };
        const gpu::Vec3d v2{ x, y, z2 };

        EXPECT_TRUE(v1 != v2);
    }
}