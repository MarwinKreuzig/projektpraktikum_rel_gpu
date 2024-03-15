/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "OctreeGPUAdapter.h"

#include "adapter/random/RandomAdapter.h"

#include "Config.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/octree/OctreeAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"

#include "enums/UpdateStatus.h"

#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/BarnesHutInternal/BarnesHut.h"
#include "algorithm/Internal/ExchangingAlgorithm.h"
#include "algorithm/Kernel/Kernel.h"

#include "structure/Cell.h"
#include "structure/Octree.h"
#include "structure/Partition.h"

#include "util/Vec3.h"
#include "util/Utility.h"

#include "gpu/algorithm/BarnesHutGPU.h"

#include "neurons/Neurons.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/input/FiredStatusCommunicationMap.h"
#include "neurons/input/SynapticInputCalculators.h"
#include "neurons/input/BackgroundActivityCalculators.h"
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

#include <range/v3/view/filter.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/view/map.hpp>

#include <memory>
#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <stack>
#include <tuple>
#include <vector>

std::pair<uint16_t, std::shared_ptr<gpu::algorithm::OctreeHandle>> OctreeGPUAdapter::construct_random_octree(std::mt19937& mt, size_t number_neurons) {
    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    const auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(mt);

    auto octree_shared_ptr = std::make_shared<OctreeImplementation<BarnesHutCell>>(min, max, level_of_branch_nodes);

    uint16_t max_level = 0;
    auto get_synaptic_count = [&mt]() { return RandomAdapter::get_random_integer<typename OctreeNode<BarnesHutCell>::counter_type>(1, 2, mt); };

    OctreeNode<BarnesHutCell> root{};
    root.set_level(0);
    root.set_rank(MPIRank::root_rank());

    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_size(min, max);
    root.set_cell_neuron_position(SimulationAdapter::get_random_position_in_box(min, max, mt));

    for (const auto id : NeuronID::range(1, number_neurons)) {
        auto* ptr = root.insert(SimulationAdapter::get_random_position_in_box(min, max, mt), id);
        if (ptr->get_level() > max_level) {
            max_level = ptr->get_level();
        }
    }

    std::stack<OctreeNode<BarnesHutCell>*> stack{};
    stack.push(&root);

    while (!stack.empty()) {
        auto* current = stack.top();
        stack.pop();

        if (current->is_leaf()) {
            if constexpr (OctreeNode<BarnesHutCell>::has_excitatory_dendrite) {
                current->set_cell_number_excitatory_dendrites(get_synaptic_count());
            }

            if constexpr (OctreeNode<BarnesHutCell>::has_inhibitory_dendrite) {
                current->set_cell_number_inhibitory_dendrites(get_synaptic_count());
            }

            if constexpr (OctreeNode<BarnesHutCell>::has_excitatory_axon) {
                current->set_cell_number_excitatory_axons(get_synaptic_count());
            }

            if constexpr (OctreeNode<BarnesHutCell>::has_inhibitory_axon) {
                current->set_cell_number_inhibitory_axons(get_synaptic_count());
            }
            continue;
        }

        for (auto* child : current->get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }
    }

    OctreeNodeUpdater<BarnesHutCell>::update_tree(&root);

    *(octree_shared_ptr->get_root()) = root;

    octree_shared_ptr->initializes_leaf_nodes(number_neurons);

    octree_shared_ptr->construct_on_gpu(number_neurons);

    return std::make_pair(max_level, octree_shared_ptr->get_gpu_handle());
}

std::tuple<uint16_t, std::shared_ptr<gpu::algorithm::OctreeHandle>, std::vector<uint64_t>> OctreeGPUAdapter::construct_random_octree_and_gather(std::mt19937& mt, size_t number_neurons, uint64_t neuron_index, SignalType signal_type, double acceptance_criterion) {
    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    const auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(mt);

    auto octree_shared_ptr = std::make_shared<OctreeImplementation<BarnesHutCell>>(min, max, level_of_branch_nodes);

    uint16_t max_level = 0;
    auto get_synaptic_count = [&mt]() { return RandomAdapter::get_random_integer<typename OctreeNode<BarnesHutCell>::counter_type>(1, 2, mt); };

    OctreeNode<BarnesHutCell> root{};
    root.set_level(0);
    root.set_rank(MPIRank::root_rank());

    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_size(min, max);
    root.set_cell_neuron_position(SimulationAdapter::get_random_position_in_box(min, max, mt));

    for (const auto id : NeuronID::range(1, number_neurons)) {
        auto* ptr = root.insert(SimulationAdapter::get_random_position_in_box(min, max, mt), id);
        if (ptr->get_level() > max_level) {
            max_level = ptr->get_level();
        }
    }

    std::stack<OctreeNode<BarnesHutCell>*> stack{};
    stack.push(&root);

    while (!stack.empty()) {
        auto* current = stack.top();
        stack.pop();

        if (current->is_leaf()) {
            if constexpr (OctreeNode<BarnesHutCell>::has_excitatory_dendrite) {
                current->set_cell_number_excitatory_dendrites(get_synaptic_count());
            }

            if constexpr (OctreeNode<BarnesHutCell>::has_inhibitory_dendrite) {
                current->set_cell_number_inhibitory_dendrites(get_synaptic_count());
            }

            if constexpr (OctreeNode<BarnesHutCell>::has_excitatory_axon) {
                current->set_cell_number_excitatory_axons(get_synaptic_count());
            }

            if constexpr (OctreeNode<BarnesHutCell>::has_inhibitory_axon) {
                current->set_cell_number_inhibitory_axons(get_synaptic_count());
            }
            continue;
        }

        for (auto* child : current->get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }
    }

    OctreeNodeUpdater<BarnesHutCell>::update_tree(&root);

    *(octree_shared_ptr->get_root()) = root;

    octree_shared_ptr->initializes_leaf_nodes(number_neurons);

    octree_shared_ptr->construct_on_gpu(number_neurons);

    gpu::Vec3d source_pos = octree_shared_ptr->get_gpu_handle()->get_node_position(neuron_index, signal_type);
    std::vector<OctreeNode<BarnesHutCell>*> nodes_gathered = BarnesHutBase<BarnesHutCell>::get_nodes_to_consider(Vec3d(source_pos.x, source_pos.y, source_pos.z), &root, ElementType::Dendrite, signal_type, acceptance_criterion);

    std::vector<uint64_t> gathered_indices(nodes_gathered.size());
    auto get_index = [](OctreeNode<BarnesHutCell>* node) -> uint64_t {
        return node->get_index_on_device();
    };

    std::transform(nodes_gathered.begin(), nodes_gathered.end(), gathered_indices.begin(), get_index);

    return std::make_tuple(max_level, octree_shared_ptr->get_gpu_handle(), gathered_indices);
}

const std::shared_ptr<gpu::kernel::KernelHandle> OctreeGPUAdapter::get_kernel() {
    return Kernel<BarnesHutCell>::get_gpu_handle();
}