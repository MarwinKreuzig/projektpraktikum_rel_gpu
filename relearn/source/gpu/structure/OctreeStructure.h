#pragma once

#include <vector>
#include <array>
#include "VectorTypes.h"
// Including this causes include errors of propagated includes. Not really sure what to do to fix this.

namespace gpu::algorithm {
    struct OctreeCPUCopy {
        
        // We need the neuron-nodes to be allocated after depth first traversal for efficient prefix traversal
        // But we need the virtual neurons to be sorted in breadth-first way, in order to make the tree update phase more efficient

        std::vector<uint64_t> neuron_ids;

        std::array<std::vector<uint64_t>, 8> child_indices;
        
        std::vector<unsigned int> num_children;

        std::vector<gpu::Vec3d> minimum_cell_position;
        std::vector<gpu::Vec3d> minimum_cell_position_virtual;

        std::vector<gpu::Vec3d> maximum_cell_position;
        std::vector<gpu::Vec3d> maximum_cell_position_virtual;

        std::vector<gpu::Vec3d> position_excitatory_element;
        std::vector<gpu::Vec3d> position_excitatory_element_virtual;
        std::vector<gpu::Vec3d> position_inhibitory_element;
        std::vector<gpu::Vec3d> position_inhibitory_element_virtual;

        std::vector<unsigned int> num_free_elements_excitatory;
        std::vector<unsigned int> num_free_elements_excitatory_virtual;
        std::vector<unsigned int> num_free_elements_inhibitory;
        std::vector<unsigned int> num_free_elements_inhibitory_virtual;

        OctreeCPUCopy(const RelearnGPUTypes::number_neurons_type number_neurons, RelearnGPUTypes::number_neurons_type number_virtual_neurons) {

            neuron_ids.reserve(number_neurons);

            child_indices[0].reserve(number_virtual_neurons);
            child_indices[1].reserve(number_virtual_neurons);
            child_indices[2].reserve(number_virtual_neurons);
            child_indices[3].reserve(number_virtual_neurons);
            child_indices[4].reserve(number_virtual_neurons);
            child_indices[5].reserve(number_virtual_neurons);
            child_indices[6].reserve(number_virtual_neurons);
            child_indices[7].reserve(number_virtual_neurons);

            num_children.reserve(number_virtual_neurons);

            minimum_cell_position.reserve(number_neurons);
            minimum_cell_position_virtual.reserve(number_virtual_neurons);
            maximum_cell_position.reserve(number_neurons);
            maximum_cell_position_virtual.reserve(number_virtual_neurons);

            position_excitatory_element.reserve(number_neurons);
            position_excitatory_element_virtual.reserve(number_virtual_neurons);
            position_inhibitory_element.reserve(number_neurons);
            position_inhibitory_element_virtual.reserve(number_virtual_neurons);

            num_free_elements_excitatory.reserve(number_neurons);
            num_free_elements_excitatory_virtual.reserve(number_virtual_neurons);
            num_free_elements_inhibitory.reserve(number_neurons);
            num_free_elements_inhibitory_virtual.reserve(number_virtual_neurons);
        }
    };
};