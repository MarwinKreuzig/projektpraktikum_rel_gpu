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

#include "harness/simulation/SimulationAdapter.h"
#include "harness/neurons/NeuronsAdapter.h"
#include "gpu/Octree.cuh"


/**
 * @brief Tests the overhead of parsing and copying the cpu octree for the gpu
 */
static void BM_octree_copy(benchmark::State& state) {
    state.PauseTiming();

    size_t number_neurons_type number_neurons = state.range(0);

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(this->mt);
    const auto neuron_positions = SimulationAdapter::get_random_positions(rng, number_neurons);
    const auto neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons);

    OctreeImplementation<BarnesHutCell> octree(min, max, level_of_branch_nodes);

    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    octree.initializes_leaf_nodes(neurons_to_place.size());

    state.ResumeTiming();

    octree.construct_on_gpu(neurons_to_place.size());

    const std::shared_ptr<gpu::algorithm::OctreeHandle> gpu_handle = octree.get_gpu_handle();
    gpu::algorithm::OctreeCPUCopy octree_cpu_copy(neurons_to_place.size(), gpu_handle->get_number_virtual_neurons());
    gpu_handle->copy_to_cpu(octree_cpu_copy);
}