#pragma once

#include <vector>
#include <array>
#include "GpuDataStructures.h"

namespace gpu::algorithm {

struct OctreeCPUCopy {
    /**
     *  Represents an Octree data structure for copying CPU octree to GPU.
     */
    // We need the leaf nodes to be allocated after depth first traversal for efficient prefix traversal
    // But we need the virtual neurons to be sorted in breadth-first way, in order to make the tree update phase more efficient

    std::vector<RelearnGPUTypes::neuron_id_type> neuron_ids;

    std::vector<RelearnGPUTypes::neuron_index_type> child_indices;
        
    std::vector<unsigned int> num_children;

    std::vector<gpu::Vec3d> minimum_cell_position;
    std::vector<gpu::Vec3d> maximum_cell_position;

    std::vector<gpu::Vec3d> position_excitatory_element;
    std::vector<gpu::Vec3d> position_inhibitory_element;

    std::vector<RelearnGPUTypes::number_elements_type> num_free_elements_excitatory;
    std::vector<RelearnGPUTypes::number_elements_type> num_free_elements_inhibitory;

    OctreeCPUCopy(const RelearnGPUTypes::number_neurons_type number_neurons, RelearnGPUTypes::number_neurons_type number_virtual_neurons) {
        neuron_ids.resize(number_neurons, 0);

        child_indices.resize(number_virtual_neurons * 8, 0);

        num_children.resize(number_virtual_neurons, 0);

        minimum_cell_position.resize(number_neurons + number_virtual_neurons);
        maximum_cell_position.resize(number_neurons + number_virtual_neurons);

        position_excitatory_element.resize(number_neurons + number_virtual_neurons);
        position_inhibitory_element.resize(number_neurons + number_virtual_neurons);

        num_free_elements_excitatory.resize(number_neurons + number_virtual_neurons, 0);
        num_free_elements_inhibitory.resize(number_neurons + number_virtual_neurons, 0);
    }
};
};
