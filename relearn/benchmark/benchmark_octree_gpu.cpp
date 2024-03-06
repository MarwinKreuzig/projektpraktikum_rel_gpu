/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "main.h"

#include "../harness/adapter/simulation/SimulationAdapter.h"
#include "../harness/adapter/neurons/NeuronsAdapter.h"

#include "../source/algorithm/Algorithms.h"
#include "../source/algorithm/Cells.h"
#include "../source/structure/Cell.h"
#include "../source/structure/Octree.h"
#include "../source/util/RelearnException.h"
#include "../source/util/Vec3.h"

#include "gpu/utils/GpuTypes.h"

/**
 * @brief Tests the overhead of parsing and copying the cpu octree to the gpu
 */
static void BM_octree_copy(benchmark::State& state) {
    using number_neurons_type = RelearnTypes::number_neurons_type;

    number_neurons_type number_neurons = state.range(0);
    std::mt19937 mt;

    state.PauseTiming();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(mt);
    auto neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons, mt);
    OctreeImplementation<BarnesHutCell> octree(min, max, level_of_branch_nodes);

    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    octree.initializes_leaf_nodes(neurons_to_place.size());

    state.ResumeTiming();

    octree.construct_on_gpu(neurons_to_place.size());
    const std::shared_ptr<gpu::algorithm::OctreeHandle> gpu_handle = octree.get_gpu_handle();
    auto octree_cpu_copy = gpu_handle->copy_to_host(neurons_to_place.size(), gpu_handle->get_number_virtual_neurons());

    //    for (auto _ : state) {
    //        octree.construct_on_gpu(neurons_to_place.size());
    //        const std::shared_ptr<gpu::algorithm::OctreeHandle> gpu_handle = octree.get_gpu_handle();
    //        auto octree_cpu_copy = gpu_handle->copy_to_host(neurons_to_place.size(), gpu_handle->get_number_virtual_neurons());
    //    }
}

BENCHMARK(BM_octree_copy)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 1000 })->Iterations(static_few_iterations);
