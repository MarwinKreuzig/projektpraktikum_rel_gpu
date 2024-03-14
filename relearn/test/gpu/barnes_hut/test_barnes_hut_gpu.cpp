/*
* This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_barnes_hut_gpu.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/octree/OctreeAdapter.h"
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

#include "gpu/algorithm/BarnesHutGPU.h"

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

// Basically just tests if the output of update_connectivity is well formed
TEST_F(BarnesHutTestGpu, testBarnesHutUpdateConnectivity) {
    using AdditionalCellAttributes = BarnesHutCell;

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(this->mt);

    auto octree_shared_ptr = std::make_shared<OctreeImplementation<BarnesHutCell>>(min, max, level_of_branch_nodes);

    size_t number_neurons = NeuronIdAdapter::get_random_number_neurons(this->mt);
    size_t num_additional_ids = NeuronIdAdapter::get_random_number_neurons(this->mt);

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, this->mt);

    const auto my_rank = MPIWrapper::get_my_rank();
    for (const auto& [position, id] : neurons_to_place) {
        octree_shared_ptr->insert(position, id);
    }

    octree_shared_ptr->initializes_leaf_nodes(neurons_to_place.size());

    octree_shared_ptr->construct_on_gpu(neurons_to_place.size());
    
    auto cast = std::static_pointer_cast<OctreeImplementation<BarnesHutCell>>(octree_shared_ptr);
    auto barnes_hut_gpu = std::make_shared<BarnesHutGPU>(std::move(cast));
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

    auto synapses = barnes_hut_gpu->update_connectivity(neurons_to_place.size());
    auto local_synapses = std::get<0>(synapses);

    ASSERT_GT(local_synapses.size(), 0);

    std::vector<uint64_t> source_sum(neurons_to_place.size(), 0);
    std::vector<uint64_t> target_sum(neurons_to_place.size(), 0);
    for (int i = 0; i < local_synapses.size(); i++) {
        ASSERT_TRUE(local_synapses[i].get_weight() == -1 || local_synapses[i].get_weight() == 1);
        ASSERT_NE(local_synapses[i].get_source().get_neuron_id(), local_synapses[i].get_target().get_neuron_id()) << " i: " << i;
        ASSERT_LT(local_synapses[i].get_source().get_neuron_id(), neurons_to_place.size());
        ASSERT_LT(local_synapses[i].get_target().get_neuron_id(), neurons_to_place.size());

        source_sum[local_synapses[i].get_source().get_neuron_id()] += 1;
        target_sum[local_synapses[i].get_target().get_neuron_id()] += 1;
    }

    for (int i = 0; i < neurons_to_place.size(); i++) {
        ASSERT_LE(source_sum[i], axs->get_free_elements(NeuronID(i)));
        ASSERT_LE(target_sum[i], dends_ex->get_free_elements(NeuronID(i)) + dends_in->get_free_elements(NeuronID(i)));
    }
}
