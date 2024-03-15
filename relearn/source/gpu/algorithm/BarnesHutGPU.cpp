/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BarnesHutGPU.h"
#include "../../util/Vec3.h"
#include "../../algorithm/Kernel/Kernel.h"
#include "../utils/GpuTypes.h"

using number_elements_type = RelearnGPUTypes::number_elements_type;
using neuron_id_type = RelearnGPUTypes::neuron_id_type;

BarnesHutGPU::BarnesHutGPU(const std::shared_ptr<OctreeImplementation<BarnesHutCell>>& octree)
    : BarnesHut(octree) {
    ElementType element_type;
    if (Cell<AdditionalCellAttributes>::has_excitatory_dendrite) {
        element_type = ElementType::Dendrite;
    } else {
        element_type = ElementType::Axon;
    }

    // Determine initial synapse space to allocate
    for (auto* leaf_node : global_tree->get_leaf_nodes()) {
        auto current_cell = leaf_node->get_cell();

        synapse_space += current_cell.get_number_elements_for(element_type, SignalType::Excitatory);
        synapse_space += current_cell.get_number_elements_for(element_type, SignalType::Inhibitory);
    }

    gpu_handle = gpu::algorithm::create_barnes_hut_data(synapse_space, global_tree->get_leaf_nodes().size());
}

BarnesHutGPU::BarnesHutGPU(const std::shared_ptr<OctreeImplementation<BarnesHutCell>>& octree, uint64_t num_nodes_gathered_before_pick,
    RelearnGPUTypes::number_neurons_type neurons_per_thread)
    : BarnesHut(octree) {
    ElementType element_type;
    if (Cell<AdditionalCellAttributes>::has_excitatory_dendrite) {
        element_type = ElementType::Dendrite;
    } else {
        element_type = ElementType::Axon;
    }

    // Determine initial synapse space to allocate
    for (auto* leaf_node : global_tree->get_leaf_nodes()) {
        auto current_cell = leaf_node->get_cell();

        synapse_space += current_cell.get_number_elements_for(element_type, SignalType::Excitatory);
        synapse_space += current_cell.get_number_elements_for(element_type, SignalType::Inhibitory);
    }

    gpu_handle = gpu::algorithm::create_barnes_hut_data(synapse_space, num_nodes_gathered_before_pick, neurons_per_thread);
}

void BarnesHutGPU::update_leaf_nodes() {
    gpu::kernel::update_leaf_nodes(global_tree->get_gpu_handle()->get_device_pointer(),
        axons->get_gpu_handle()->get_device_pointer(),
        excitatory_dendrites->get_gpu_handle()->get_device_pointer(),
        inhibitory_dendrites->get_gpu_handle()->get_device_pointer(),
        extra_infos->get_gpu_handle()->get_device_pointer(),
        global_tree->get_leaf_nodes().size(),
        update_leaf_nodes_kernel_grid_size,
        update_leaf_nodes_kernel_block_size);
}

void BarnesHutGPU::update_octree() {
    BarnesHutGPU::update_leaf_nodes();

    // This does the bottom up update through the octree itself
    global_tree->get_gpu_handle()->update_virtual_neurons();
}

[[nodiscard]] std::tuple<PlasticLocalSynapses, PlasticDistantInSynapses, PlasticDistantOutSynapses> BarnesHutGPU::update_connectivity(number_neurons_type number_neurons) {

    ElementType element_type;
    if (Cell<AdditionalCellAttributes>::has_excitatory_dendrite) {
        element_type = ElementType::Dendrite;
    } else {
        element_type = ElementType::Axon;
    }

    number_elements_type num_free_elements_ex = global_tree->get_gpu_handle()->get_total_excitatory_elements();
    number_elements_type num_free_elements_in = global_tree->get_gpu_handle()->get_total_inhibitory_elements();
    unsigned int new_synapse_space = num_free_elements_ex + num_free_elements_in;

    // Check if more synapse space needs to be allocated, based on new number of total elements
    if (new_synapse_space > synapse_space) {
        synapse_space = new_synapse_space;
        gpu_handle->update_synapse_allocation(synapse_space);
    }

    // Initially or when number_neurons has changed, new allocation needs to happen
    if (this->number_neurons != global_tree->get_leaf_nodes().size()) {
        this->number_neurons = global_tree->get_leaf_nodes().size();
        gpu_handle->update_kernel_allocation_sizes(this->number_neurons, global_tree->get_max_level());
    }

    // Once everything is on the GPU, these will not have to be given as a return parameter and they should be directly processed on the GPU
    std::vector<gpu::Synapse> gpu_synapses = gpu::kernel::update_connectivity_gpu(gpu_handle.get(),
        global_tree->get_gpu_handle()->get_device_pointer(),
        axons->get_gpu_handle()->get_device_pointer(),
        extra_infos->get_gpu_handle()->get_device_pointer(),
        Kernel<BarnesHutCell>::get_gpu_handle()->get_device_pointer());

    PlasticLocalSynapses cpu_synapses(gpu_synapses.size(), PlasticLocalSynapse(NeuronID(0), NeuronID(0), 0));

    auto convert_gpu_synapse_to_cpu = [](const gpu::Synapse& synapse) -> PlasticLocalSynapse {
        return PlasticLocalSynapse(NeuronID(synapse.target_id), NeuronID(synapse.source_id), synapse.weight);
    };

    std::transform(gpu_synapses.begin(), gpu_synapses.end(), cpu_synapses.begin(), convert_gpu_synapse_to_cpu);
    return std::make_tuple(cpu_synapses, PlasticDistantInSynapses(), PlasticDistantOutSynapses());
}

void BarnesHutGPU::set_acceptance_criterion(double acceptance_criterion) {
    BarnesHut::set_acceptance_criterion(acceptance_criterion);

    gpu_handle->set_acceptance_criterion(acceptance_criterion);
}