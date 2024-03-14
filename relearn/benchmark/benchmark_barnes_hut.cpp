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

#include "gpu/algorithm/BarnesHutGPU.h"
#include "util/MemoryHolder.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "enums/UpdateStatus.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"

#include "algorithm/Algorithms.h"
#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/Cells.h"
#include "structure/Cell.h"
#include "structure/Octree.h"
#include "util/NeuronID.h"
#include "util/Vec3.h"
#include "util/ranges/Functional.hpp"

#include <memory>
#include <range/v3/view/filter.hpp>

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <stack>
#include <tuple>
#include <vector>

#include <range/v3/algorithm/sort.hpp>
#include <range/v3/view/map.hpp>

#include "neurons/Neurons.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/input/FiredStatusCommunicationMap.h"
#include "neurons/input/SynapticInputCalculators.h"
#include "neurons/input/BackgroundActivityCalculators.h"
#include "algorithm/BarnesHutInternal/BarnesHut.h"
#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/Internal/ExchangingAlgorithm.h"
#include "neurons/CalciumCalculator.h"
#include "neurons/input/BackgroundActivityCalculators.h"
#include "neurons/input/Stimulus.h"
#include "neurons/input/SynapticInputCalculator.h"
#include "neurons/input/SynapticInputCalculators.h"
#include "neurons/models/AEIFModel.h"
#include "neurons/models/FitzHughNagumoModel.h"
#include "neurons/models/IzhikevichModel.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/models/PoissonModel.h"
#include "neurons/models/SynapticElements.h"
#include "neurons/Neurons.h"
#include "structure/Partition.h"
#include "util/Utility.h"
#include "util/ranges/Functional.hpp"

#include <utility>

static void BM_BarnesHutGPU_Update_Connectivity(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto num_gather = state.range(1);
    const auto neurons_per_thread = state.range(2);

    const Vec3d min{ -5000., -5000., -5000. };
    const Vec3d max{ 5000., 5000., 5000. };

    const uint8_t level_of_branch_nodes = 3;

    auto octree_shared_ptr = std::make_shared<OctreeImplementation<BarnesHutCell>>(min, max, level_of_branch_nodes);

    size_t num_additional_ids = 500;

    std::random_device dev;
    std::mt19937 mt(dev());

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, mt);

    const auto my_rank = MPIWrapper::get_my_rank();
    for (const auto& [position, id] : neurons_to_place) {
        octree_shared_ptr->insert(position, id);
    }

    octree_shared_ptr->initializes_leaf_nodes(neurons_to_place.size());

    octree_shared_ptr->construct_on_gpu(neurons_to_place.size());
    
    auto cast = std::static_pointer_cast<OctreeImplementation<BarnesHutCell>>(octree_shared_ptr);
    auto barnes_hut_gpu = std::make_shared<BarnesHutGPU>(std::move(cast), num_gather, neurons_per_thread);
    auto cast2 = std::static_pointer_cast<OctreeImplementation<BarnesHutCell>>(octree_shared_ptr);
    auto barnes_hut_cpu = std::make_shared<BarnesHut>(std::move(cast2));

    
    auto axs = SynapticElementsAdapter::create_axons(neurons_to_place.size(), 5, 10, mt);
    auto dends_ex = SynapticElementsAdapter::create_dendrites(neurons_to_place.size(), SignalType::Excitatory, 0, 2, mt);
    auto dends_in = SynapticElementsAdapter::create_dendrites(neurons_to_place.size(), SignalType::Inhibitory, 0, 2, mt);

    std::shared_ptr<NeuronsExtraInfo> extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(neurons_to_place.size());

    barnes_hut_gpu->set_neuron_extra_infos(extra_infos);
    barnes_hut_gpu->set_synaptic_elements(axs, dends_ex, dends_in);
    barnes_hut_cpu->set_neuron_extra_infos(extra_infos);
    barnes_hut_cpu->set_synaptic_elements(axs, dends_ex, dends_in);

    std::vector<SignalType> signal_types(neurons_to_place.size());

    for (const auto neuron_id : NeuronID::range_id(neurons_to_place.size())) {
        const auto& signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

        signal_types[neuron_id] = signal_type;
    }

    std::vector<Vec3d> neuron_positions(neurons_to_place.size());
    auto convert_pair = [](const std::pair<Vec3d, NeuronID>& p) -> Vec3d {
        return p.first;
    };
    std::transform(neurons_to_place.begin(), neurons_to_place.end(), neuron_positions.begin(), convert_pair);

    axs->set_signal_types(std::move(signal_types));
    extra_infos->set_positions(std::move(neuron_positions));

    barnes_hut_cpu->update_octree();

    octree_shared_ptr->update_gpu_octree_structure();

    for (auto _ : state) {
        const auto& tuple = barnes_hut_gpu->update_connectivity(neurons_to_place.size());
        state.PauseTiming();

        const auto& local_synapses = std::get<0>(tuple);

        int weight_sum = 0;
        for (auto& synapse : local_synapses) {
            weight_sum += synapse.get_weight();
        }

        benchmark::DoNotOptimize(weight_sum);

        barnes_hut_cpu->update_octree();

        octree_shared_ptr->update_gpu_octree_structure();

        state.ResumeTiming();
    }

    MemoryHolder<BarnesHutCell>::make_all_available();
}

static void BM_BarnesHut_Update_Connectivity(benchmark::State& state) {

    const auto number_neurons = state.range(0);

    const Vec3d min{ -5000., -5000., -5000. };
    const Vec3d max{ 5000., 5000., 5000. };

    const uint8_t level_of_branch_nodes = 3;

    auto octree_shared_ptr = std::make_shared<OctreeImplementation<BarnesHutCell>>(min, max, level_of_branch_nodes);

    size_t num_additional_ids = 500;

    std::random_device dev;
    std::mt19937 mt(dev());

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, mt);

    const auto my_rank = MPIWrapper::get_my_rank();
    for (const auto& [position, id] : neurons_to_place) {
        octree_shared_ptr->insert(position, id);
    }

    octree_shared_ptr->initializes_leaf_nodes(neurons_to_place.size());
    
    auto cast = std::static_pointer_cast<OctreeImplementation<BarnesHutCell>>(octree_shared_ptr);
    auto barnes_hut_cpu = std::make_shared<BarnesHut>(std::move(cast));

    auto axs = SynapticElementsAdapter::create_axons(neurons_to_place.size(), 5, 10, mt);
    auto dends_ex = SynapticElementsAdapter::create_dendrites(neurons_to_place.size(), SignalType::Excitatory, 0, 2, mt);
    auto dends_in = SynapticElementsAdapter::create_dendrites(neurons_to_place.size(), SignalType::Inhibitory, 0, 2, mt);

    std::shared_ptr<NeuronsExtraInfo> extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(neurons_to_place.size());

    barnes_hut_cpu->set_neuron_extra_infos(extra_infos);
    barnes_hut_cpu->set_synaptic_elements(axs, dends_ex, dends_in);

    std::vector<SignalType> signal_types(neurons_to_place.size());

    for (const auto neuron_id : NeuronID::range_id(neurons_to_place.size())) {
        const auto& signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

        signal_types[neuron_id] = signal_type;
    }

    std::vector<Vec3d> neuron_positions(neurons_to_place.size());
    auto convert_pair = [](const std::pair<Vec3d, NeuronID>& p) -> Vec3d {
        return p.first;
    };
    std::transform(neurons_to_place.begin(), neurons_to_place.end(), neuron_positions.begin(), convert_pair);
    
    axs->set_signal_types(std::move(signal_types));
    extra_infos->set_positions(std::move(neuron_positions));


    barnes_hut_cpu->update_octree();


    for (auto _ : state) {
        const auto& tuple = barnes_hut_cpu->update_connectivity(neurons_to_place.size());
        state.PauseTiming();

        const auto& local_synapses = std::get<0>(tuple);

        int weight_sum = 0;
        for (auto& synapse : local_synapses) {
            weight_sum += synapse.get_weight();
        }

        benchmark::DoNotOptimize(weight_sum);

        barnes_hut_cpu->update_octree();

        state.ResumeTiming();
    }
}

BENCHMARK(BM_BarnesHutGPU_Update_Connectivity)->Unit(benchmark::kMillisecond)->Args({800000, 10000, 4})->Iterations(10);

//BENCHMARK(BM_BarnesHut_Update_Connectivity)->Unit(benchmark::kMillisecond)->Args({800000})->Iterations(1);