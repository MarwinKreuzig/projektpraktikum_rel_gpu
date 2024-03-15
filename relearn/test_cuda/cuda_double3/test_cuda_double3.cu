/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cuda_double3/test_cuda_double3.cuh"

#include "../harness/adapter/random/RandomAdapter.h"

#include "../../source/gpu/structure/CudaDouble3.cuh"
#include "RelearnGPUException.h"
#include <random>

TEST_F(CudaDouble3Test, testCudaDouble3EmptyConstructor) {
    const gpu::Vector::CudaDouble3 v{};

    EXPECT_NEAR(v.get_x(), 0.0, 0.001);
    EXPECT_NEAR(v.get_y(), 0.0, 0.001);
    EXPECT_NEAR(v.get_z(), 0.0, 0.001);
}

TEST_F(CudaDouble3Test, testCudaDouble3Constructor) {

    for (int i = 0; i < 100; ++i) {
        double x = (double)rand();
        double y = (double)rand();
        double z = (double)rand();

        const gpu::Vector::CudaDouble3 v{ x, y, z };

        EXPECT_NEAR(v.get_x(), x, 0.001);
        EXPECT_NEAR(v.get_y(), y, 0.001);
        EXPECT_NEAR(v.get_z(), z, 0.001);
    }
}

TEST_F(CudaDouble3Test, testCudaDouble3Equal) {

    for (int i = 0; i < 100; ++i) {
        double x = (double)rand();
        double y = (double)rand();
        double z = (double)rand();
        const gpu::Vector::CudaDouble3 v1{ x, y, z };
        const gpu::Vector::CudaDouble3 v2{ x, y, z };

        EXPECT_TRUE(v1 == v2);
    }
}

TEST_F(CudaDouble3Test, testCudaDouble3Unequal) {

    for (int i = 0; i < 100; ++i) {
        double x1 = (double)rand();

        double x2 = (double)rand();
        while (x1 == x2) {
            x2 = (double)rand();
        }

        double y = (double)rand();
        double z = (double)rand();

        const gpu::Vector::CudaDouble3 v1{ x1, y, z };
        const gpu::Vector::CudaDouble3 v2{ x2, y, z };

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

        const gpu::Vector::CudaDouble3 v1{ x, y1, z };
        const gpu::Vector::CudaDouble3 v2{ x, y2, z };

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

        const gpu::Vector::CudaDouble3 v1{ x, y, z1 };
        const gpu::Vector::CudaDouble3 v2{ x, y, z2 };

        EXPECT_TRUE(v1 != v2);
    }
}

TEST_F(CudaDouble3Test, testCudaDouble3Sub) {

    for (int i = 0; i < 100; ++i) {
        double x1 = (double)rand();
        double x2 = (double)rand();
        double x = x1 - x2;

        double y1 = (double)rand();
        double y2 = (double)rand();
        double y = y1 - y2;

        double z1 = (double)rand();
        double z2 = (double)rand();
        double z = z1 - z2;

        const gpu::Vector::CudaDouble3 v1{ x1, y1, z1 };
        const gpu::Vector::CudaDouble3 v2{ x2, y2, z2 };
        const gpu::Vector::CudaDouble3 v = v1 - v2;

        EXPECT_EQ(x, v.get_x());
        EXPECT_EQ(y, v.get_y());
        EXPECT_EQ(z, v.get_z());
    }
}

TEST_F(CudaDouble3Test, testCudaDouble3SubInt) {

    for (int i = 0; i < 100; ++i) {
        int s = (int)rand();

        double x = (double)rand();
        double xs = x - s;

        double y = (double)rand();
        double ys = y - s;

        double z = (double)rand();
        double zs = z - s;

        const gpu::Vector::CudaDouble3 v{ x, y, z };
        const gpu::Vector::CudaDouble3 vs = v - s;

        EXPECT_EQ(xs, vs.get_x());
        EXPECT_EQ(ys, vs.get_y());
        EXPECT_EQ(zs, vs.get_z());
    }
}

TEST_F(CudaDouble3Test, testCudaDouble3Add) {

    for (int i = 0; i < 100; ++i) {
        double x1 = (double)rand();
        double x2 = (double)rand();
        double x = x1 + x2;

        double y1 = (double)rand();
        double y2 = (double)rand();
        double y = y1 + y2;

        double z1 = (double)rand();
        double z2 = (double)rand();
        double z = z1 + z2;

        const gpu::Vector::CudaDouble3 v1{ x1, y1, z1 };
        const gpu::Vector::CudaDouble3 v2{ x2, y2, z2 };
        const gpu::Vector::CudaDouble3 v = v1 + v2;

        EXPECT_EQ(x, v.get_x());
        EXPECT_EQ(y, v.get_y());
        EXPECT_EQ(z, v.get_z());
    }
}

TEST_F(CudaDouble3Test, testCudaDouble3AddInt) {

    for (int i = 0; i < 100; ++i) {
        int s = (int)rand();

        double x = (double)rand();
        double xs = x + s;

        double y = (double)rand();
        double ys = y + s;

        double z = (double)rand();
        double zs = z + s;

        const gpu::Vector::CudaDouble3 v{ x, y, z };
        const gpu::Vector::CudaDouble3 vs = v + s;

        EXPECT_EQ(xs, vs.get_x());
        EXPECT_EQ(ys, vs.get_y());
        EXPECT_EQ(zs, vs.get_z());
    }
}

TEST_F(CudaDouble3Test, testCudaDouble3Mul) {

    for (int i = 0; i < 100; ++i) {
        double x1 = (double)rand();
        double x2 = (double)rand();
        double x = x1 * x2;

        double y1 = (double)rand();
        double y2 = (double)rand();
        double y = y1 * y2;

        double z1 = (double)rand();
        double z2 = (double)rand();
        double z = z1 * z2;

        const gpu::Vector::CudaDouble3 v1{ x1, y1, z1 };
        const gpu::Vector::CudaDouble3 v2{ x2, y2, z2 };
        const gpu::Vector::CudaDouble3 v = v1 * v2;

        EXPECT_EQ(x, v.get_x());
        EXPECT_EQ(y, v.get_y());
        EXPECT_EQ(z, v.get_z());
    }
}

TEST_F(CudaDouble3Test, testCudaDouble3MulInt) {

    for (int i = 0; i < 100; ++i) {
        int s = (int)rand();

        double x = (double)rand();
        double y = (double)rand();
        double z = (double)rand();
        const gpu::Vector::CudaDouble3 v{ x, y, z };

        const gpu::Vector::CudaDouble3 vs = v * s;

        EXPECT_EQ(v.get_x() * s, vs.get_x());
        EXPECT_EQ(v.get_y() * s, vs.get_y());
        EXPECT_EQ(v.get_z() * s, vs.get_z());
    }
}

TEST_F(CudaDouble3Test, testCudaDouble3Max) {

    for (int i = 0; i < 100; ++i) {
        double x = (double)rand();
        double y = (double)rand();
        double z = (double)rand();
        const gpu::Vector::CudaDouble3 v{ x, y, z };

        double m = std::max(std::max(x, y), z);

        EXPECT_EQ(m, v.max());
    }
}

TEST_F(CudaDouble3Test, testCudaDouble3Min) {

    for (int i = 0; i < 100; ++i) {
        double x = (double)rand();
        double y = (double)rand();
        double z = (double)rand();
        const gpu::Vector::CudaDouble3 v{ x, y, z };

        double m = std::min(std::min(x, y), z);

        EXPECT_EQ(m, v.min());
    }
}
