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
#include "../../../source/gpu/structure/VectorTypes.h"

#include <sstream>

TEST_F(Vec3TestGpu, testVectorGpuEmptyConstructor) {
    const gpu::Vec3d v{};


    ASSERT_EQ(0.0, v.x);
    ASSERT_EQ(0.0, v.y);
    ASSERT_EQ(0.0, v.z);
}