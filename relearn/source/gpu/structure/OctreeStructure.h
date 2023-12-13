#pragma once

#include <vector>
#include <array>
#include "../../util/Vec3.h"

// Using Vec3d causes an include error here, but potentially we dont even want to use Vec3d, since its data alignment might not fit to the Cuda vector types
// A better option would be something like glm::vec3, or another aligned custom vector type TODO
namespace gpu::algorithm {
    struct OctreeCPUCopy {
        
        // We need the neuron-nodes to be allocated after depth first traversal for efficient prefix traversal
        // But we need the virtual neurons to be sorted in breadth-first way, in order to make the tree update phase more efficient

        std::vector<uint64_t> neuron_ids;

        /*std::vector<uint64_t> child_index_1;
        std::vector<uint64_t> child_index_2;
        std::vector<uint64_t> child_index_3;
        std::vector<uint64_t> child_index_4;
        std::vector<uint64_t> child_index_5;
        std::vector<uint64_t> child_index_6;
        std::vector<uint64_t> child_index_7;
        std::vector<uint64_t> child_index_8;*/
        std::array<std::vector<uint64_t>, 8> child_indices;
        
        std::vector<unsigned int> num_children;

        /*std::vector<double> minimum_position_x;
        std::vector<double> minimum_position_y;
        std::vector<double> minimum_position_z;*/

        std::vector<Vec3d> minimum_cell_position;
        std::vector<Vec3d> minimum_cell_position_virtual;

        /*std::vector<double> maximum_position_x;
        std::vector<double> maximum_position_y;
        std::vector<double> maximum_position_z;*/

        std::vector<Vec3d> maximum_cell_position;
        std::vector<Vec3d> maximum_cell_position_virtual;

        /*std::vector<double> position_x_excitatory_dendrite;
        std::vector<double> position_y_excitatory_dendrite;
        std::vector<double> position_z_excitatory_dendrite;
        std::vector<double> position_x_inhibitory_dendrite;
        std::vector<double> position_y_inhibitory_dendrite;
        std::vector<double> position_z_inhibitory_dendrite;
        std::vector<double> position_x_excitatory_axon;
        std::vector<double> position_y_excitatory_axon;
        std::vector<double> position_z_excitatory_axon;
        std::vector<double> position_x_inhibitory_axon;
        std::vector<double> position_y_inhibitory_axon;
        std::vector<double> position_z_inhibitory_axon;*/

        std::vector<Vec3d> position_excitatory_element;
        std::vector<Vec3d> position_excitatory_element_virtual;
        std::vector<Vec3d> position_inhibitory_element;
        std::vector<Vec3d> position_inhibitory_element_virtual;

        /*std::vector<unsigned int> num_free_elements_excitatory_dendrite;
        std::vector<unsigned int> num_free_elements_inhibitory_dendrite;
        std::vector<unsigned int> num_free_elements_excitatory_axon;
        std::vector<unsigned int> num_free_elements_inhibitory_axon;*/

        std::vector<unsigned int> num_free_elements_excitatory;
        std::vector<unsigned int> num_free_elements_excitatory_virtual;
        std::vector<unsigned int> num_free_elements_inhibitory;
        std::vector<unsigned int> num_free_elements_inhibitory_virtual;

        OctreeCPUCopy(const RelearnGPUTypes::number_neurons_type number_neurons, RelearnGPUTypes::number_neurons_type number_virtual_neurons) {
                
            /*neuron_ids = new uint64_t[number_neurons];

            child_index_1 = new uint64_t[number_virtual_neurons];
            child_index_2 = new uint64_t[number_virtual_neurons];
            child_index_3 = new uint64_t[number_virtual_neurons];
            child_index_4 = new uint64_t[number_virtual_neurons];
            child_index_5 = new uint64_t[number_virtual_neurons];
            child_index_6 = new uint64_t[number_virtual_neurons];
            child_index_7 = new uint64_t[number_virtual_neurons];
            child_index_8 = new uint64_t[number_virtual_neurons];

            minimum_position_x = new double[number_virtual_neurons + number_neurons];
            minimum_position_y = new double[number_virtual_neurons + number_neurons];
            minimum_position_z = new double[number_virtual_neurons + number_neurons];
            maximum_position_x = new double[number_virtual_neurons + number_neurons];
            maximum_position_y = new double[number_virtual_neurons + number_neurons];
            maximum_position_z = new double[number_virtual_neurons + number_neurons];

            position_x_excitatory_dendrite = new double[number_virtual_neurons + number_neurons];
            position_y_excitatory_dendrite = new double[number_virtual_neurons + number_neurons];
            position_z_excitatory_dendrite = new double[number_virtual_neurons + number_neurons];
            position_x_inhibitory_dendrite = new double[number_virtual_neurons + number_neurons];
            position_y_inhibitory_dendrite = new double[number_virtual_neurons + number_neurons];
            position_z_inhibitory_dendrite = new double[number_virtual_neurons + number_neurons];
            position_x_excitatory_axon = new double[number_virtual_neurons + number_neurons];
            position_y_excitatory_axon = new double[number_virtual_neurons + number_neurons];
            position_z_excitatory_axon = new double[number_virtual_neurons + number_neurons];
            position_x_inhibitory_axon = new double[number_virtual_neurons + number_neurons];
            position_y_inhibitory_axon = new double[number_virtual_neurons + number_neurons];
            position_z_inhibitory_axon = new double[number_virtual_neurons + number_neurons];
            num_free_elements_excitatory_dendrite = new unsigned int[number_virtual_neurons + number_neurons];
            num_free_elements_inhibitory_dendrite = new unsigned int[number_virtual_neurons + number_neurons];
            num_free_elements_excitatory_axon = new unsigned int[number_virtual_neurons + number_neurons];
            num_free_elements_inhibitory_axon = new unsigned int[number_virtual_neurons + number_neurons];*/

            neuron_ids.reserve(number_neurons);

            /*child_index_1.reserve(number_virtual_neurons);
            child_index_2.reserve(number_virtual_neurons);
            child_index_3.reserve(number_virtual_neurons);
            child_index_4.reserve(number_virtual_neurons);
            child_index_5.reserve(number_virtual_neurons);
            child_index_6.reserve(number_virtual_neurons);
            child_index_7.reserve(number_virtual_neurons);
            child_index_8.reserve(number_virtual_neurons);*/
            child_indices[0].reserve(number_virtual_neurons);
            child_indices[1].reserve(number_virtual_neurons);
            child_indices[2].reserve(number_virtual_neurons);
            child_indices[3].reserve(number_virtual_neurons);
            child_indices[4].reserve(number_virtual_neurons);
            child_indices[5].reserve(number_virtual_neurons);
            child_indices[6].reserve(number_virtual_neurons);
            child_indices[7].reserve(number_virtual_neurons);

            num_children.reserve(number_virtual_neurons);

            /*minimum_position_x.reserve(number_virtual_neurons + number_neurons);
            minimum_position_y.reserve(number_virtual_neurons + number_neurons);
            minimum_position_z.reserve(number_virtual_neurons + number_neurons);
            maximum_position_x.reserve(number_virtual_neurons + number_neurons);
            maximum_position_y.reserve(number_virtual_neurons + number_neurons);
            maximum_position_z.reserve(number_virtual_neurons + number_neurons);*/

            minimum_cell_position.reserve(number_neurons);
            minimum_cell_position_virtual.reserve(number_virtual_neurons);
            maximum_cell_position.reserve(number_neurons);
            maximum_cell_position_virtual.reserve(number_virtual_neurons);

            /*position_x_excitatory_dendrite.reserve(number_virtual_neurons + number_neurons);
            position_y_excitatory_dendrite.reserve(number_virtual_neurons + number_neurons);
            position_z_excitatory_dendrite.reserve(number_virtual_neurons + number_neurons);
            position_x_inhibitory_dendrite.reserve(number_virtual_neurons + number_neurons);
            position_y_inhibitory_dendrite.reserve(number_virtual_neurons + number_neurons);
            position_z_inhibitory_dendrite.reserve(number_virtual_neurons + number_neurons);
            position_x_excitatory_axon.reserve(number_virtual_neurons + number_neurons);
            position_y_excitatory_axon.reserve(number_virtual_neurons + number_neurons);
            position_z_excitatory_axon.reserve(number_virtual_neurons + number_neurons);
            position_x_inhibitory_axon.reserve(number_virtual_neurons + number_neurons);
            position_y_inhibitory_axon.reserve(number_virtual_neurons + number_neurons);
            position_z_inhibitory_axon.reserve(number_virtual_neurons + number_neurons);*/

            position_excitatory_element.reserve(number_neurons);
            position_excitatory_element_virtual.reserve(number_virtual_neurons);
            position_inhibitory_element.reserve(number_neurons);
            position_inhibitory_element_virtual.reserve(number_virtual_neurons);

            /*num_free_elements_excitatory_dendrite.reserve(number_virtual_neurons + number_neurons);
            num_free_elements_inhibitory_dendrite.reserve(number_virtual_neurons + number_neurons);
            num_free_elements_excitatory_axon.reserve(number_virtual_neurons + number_neurons);
            num_free_elements_inhibitory_axon.reserve(number_virtual_neurons + number_neurons);*/

            num_free_elements_excitatory.reserve(number_neurons);
            num_free_elements_excitatory_virtual.reserve(number_virtual_neurons);
            num_free_elements_inhibitory.reserve(number_neurons);
            num_free_elements_inhibitory_virtual.reserve(number_virtual_neurons);
        }

        /*~OctreeCPUCopy(const RelearnGPUTypes::number_neurons_type number_neurons, const RelearnGPUTypes::number_neurons_type number_virtual_neurons) {
                
            delete[] neuron_ids;

            delete[] child_index_1;
            delete[] child_index_2;
            delete[] child_index_3;
            delete[] child_index_4;
            delete[] child_index_5;
            delete[] child_index_6;
            delete[] child_index_7;
            delete[] child_index_8;

            delete[] minimum_position_x;
            delete[] minimum_position_y;
            delete[] minimum_position_z;
            delete[] maximum_position_x;
            delete[] maximum_position_y;
            delete[] maximum_position_z;

            delete[] position_x_excitatory_dendrite;
            delete[] position_y_excitatory_dendrite;
            delete[] position_z_excitatory_dendrite;
            delete[] position_x_inhibitory_dendrite;
            delete[] position_y_inhibitory_dendrite;
            delete[] position_z_inhibitory_dendrite;

            delete[] position_x_excitatory_axon;
            delete[] position_y_excitatory_axon;
            delete[] position_z_excitatory_axon;
            delete[] position_x_inhibitory_axon;
            delete[] position_y_inhibitory_axon;
            delete[] position_z_inhibitory_axon;

            delete[] num_free_elements_excitatory_dendrite;
            delete[] num_free_elements_inhibitory_dendrite;
            delete[] num_free_elements_excitatory_axon;
            delete[] num_free_elements_inhibitory_axon;
        }*/
    };
};