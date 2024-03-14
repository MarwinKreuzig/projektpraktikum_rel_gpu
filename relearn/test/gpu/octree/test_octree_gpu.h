#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "../../RelearnTest.hpp"

#include "util/Vec3.h"
#include "gpu/structure/GpuDataStructures.h"

/**
 * @brief converts a gpu::Vec3 to an util::Vec3
 * @param gpu::Vec3 to convert
 * @return converted util::Vec3
 */
const auto convert_gpu_vec_to_vec(const gpu::Vec3d gpu_vec);

/**
 * @brief converts an util::Vec3 to a gpu::Vec3
 * @param util::Vec3 to convert
 * @return converted gpu::Vec3
 */
const auto convert_vec_to_gpu_vec(const Vec3d cpu_vec);

/**
 * @brief applies ASSERT_DOUBLE_EQ() to all elements of a gpu::Vec3
 */
const auto assert_eq_vec(const Vec3d vec1, const Vec3d vec2);

template <typename AdditionalCellAttributes>
class OctreeTestGpu : public RelearnMemoryTest {
};