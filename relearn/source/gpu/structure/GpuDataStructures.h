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

#include "utils/GpuTypes.h"

namespace gpu {
/**
 * Data structures here exist, since they are understood by both the GPU and the CPU, making easy data transfer possible this way
 */
template <typename T>
struct Vec3 {
    /**
     * Vector struct mainly used to easily transfer to the Vec3 structure in non cuda code
     */

    T x;
    T y;
    T z;

    Vec3(T x, T y, T z)
        : x(x)
        , y(y)
        , z(z) { }

    Vec3() { }

    bool operator!=(const Vec3& rhs) const {
        return (x != rhs.x) || (y != rhs.y) || (z != rhs.z);
    }

    bool operator==(const Vec3& rhs) const {
        return (x == rhs.x) && (y == rhs.y) && (z == rhs.z);
    }
};

using Vec3d = Vec3<double>;

struct Synapse {
    /**
     * Stores Synapses created on the GPU and can be read by both cuda and non cuda code
     */

    RelearnGPUTypes::neuron_id_type target_id;
    RelearnGPUTypes::neuron_id_type source_id;
    int weight;

    Synapse(RelearnGPUTypes::neuron_id_type target_id, RelearnGPUTypes::neuron_id_type source_id, int weight)
        : target_id(target_id)
        , source_id(source_id)
        , weight(weight) { }
};
};