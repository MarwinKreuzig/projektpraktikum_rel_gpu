#pragma once

#include <vector>

namespace gpu::algorithm {
    struct OctreeCPUCopy {
        
        std::vector<uint64_t> neuron_ids;

        /*std::vector<uint64_t> child_index_1;
        std::vector<uint64_t> child_index_2;
        std::vector<uint64_t> child_index_3;
        std::vector<uint64_t> child_index_4;
        std::vector<uint64_t> child_index_5;
        std::vector<uint64_t> child_index_6;
        std::vector<uint64_t> child_index_7;
        std::vector<uint64_t> child_index_8;*/
        std::vector<uint64_t> child_indices;

        std::vector<double> minimum_position_x;
        std::vector<double> minimum_position_y;
        std::vector<double> minimum_position_z;
        std::vector<double> maximum_position_x;
        std::vector<double> maximum_position_y;
        std::vector<double> maximum_position_z;

        std::vector<double> position_x_excitatory_dendrite;
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
        std::vector<double> position_z_inhibitory_axon;

        std::vector<unsigned int> num_free_elements_excitatory_dendrite;
        std::vector<unsigned int> num_free_elements_inhibitory_dendrite;
        std::vector<unsigned int> num_free_elements_excitatory_axon;
        std::vector<unsigned int> num_free_elements_inhibitory_axon;

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
            child_indices.reserve(number_virtual_neurons * 8);

            minimum_position_x.reserve(number_virtual_neurons + number_neurons);
            minimum_position_y.reserve(number_virtual_neurons + number_neurons);
            minimum_position_z.reserve(number_virtual_neurons + number_neurons);
            maximum_position_x.reserve(number_virtual_neurons + number_neurons);
            maximum_position_y.reserve(number_virtual_neurons + number_neurons);
            maximum_position_z.reserve(number_virtual_neurons + number_neurons);

            position_x_excitatory_dendrite.reserve(number_virtual_neurons + number_neurons);
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
            position_z_inhibitory_axon.reserve(number_virtual_neurons + number_neurons);
            num_free_elements_excitatory_dendrite.reserve(number_virtual_neurons + number_neurons);
            num_free_elements_inhibitory_dendrite.reserve(number_virtual_neurons + number_neurons);
            num_free_elements_excitatory_axon.reserve(number_virtual_neurons + number_neurons);
            num_free_elements_inhibitory_axon.reserve(number_virtual_neurons + number_neurons);
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