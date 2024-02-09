/*
* This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#if CUDA_FOUND

#include "main.h"
#include "gpu/Octree.cuh"

static void BM_octree_copy(benchmark::State& state) {
    const auto number_neurons = state.range(0);


}