/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include "../util/Vec3.h"

#include <optional>

/**
 * This type is used to represent a virtual plasticity element,
 * i.e., axons and dendrites, when it comes to combining multiple of them
 * in the octree
 */
struct VirtualPlasticityElement {
    // All elements have the same position
    std::optional<Vec3d> position{};
    unsigned int num_free_elements{ 0 };
};

using Axons = VirtualPlasticityElement;
using Dendrites = VirtualPlasticityElement;
