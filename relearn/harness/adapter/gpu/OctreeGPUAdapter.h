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

#include "gpu/utils/Interface.h"
#include "gpu/algorithm/kernel/KernelGPUInterface.h"

#include <random>

// These exist in order to allow stuff that normally won't compile in the cuda tests
// to be used over the declared functions here
class OctreeGPUAdapter {
public:
    // Returns the max_depth of the tree and the handle to the gpu version of the octree
    static std::pair<uint16_t, std::shared_ptr<gpu::algorithm::OctreeHandle>> construct_random_octree(std::mt19937& mt, size_t number_neurons);

    // Returns the max_depth of the tree, the handle to the gpu version of the octree and does get_nodes_to_consider on the cpu for a specified neuron on the tree
    static std::tuple<uint16_t, std::shared_ptr<gpu::algorithm::OctreeHandle>, std::vector<uint64_t>> construct_random_octree_and_gather(std::mt19937& mt, size_t number_neurons, uint64_t neuron_index, SignalType signal_type, double acceptance_criterion);

    static const std::shared_ptr<gpu::kernel::KernelHandle> get_kernel();
};