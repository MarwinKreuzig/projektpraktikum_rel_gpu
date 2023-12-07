#pragma once

#include "Commons.cuh"
#include "gpu/GpuTypes.h"
#include "gpu/Interface.h"
#include "NeuronsExtraInfos.cuh"

#include "CudaArray.cuh"
#include "CudaVector.cuh"

#include <numeric>

namespace gpu::algorithm {

    struct Octree {
        uint64_t* neuron_ids;

        /*uint64_t* child_index_1;
        uint64_t* child_index_2;
        uint64_t* child_index_3;
        uint64_t* child_index_4;
        uint64_t* child_index_5;
        uint64_t* child_index_6;
        uint64_t* child_index_7;
        uint64_t* child_index_8;*/
        uint64_t* child_indices;

        // maybe this can be cuda vectors, depends on if access stays coalesced
        double* minimum_position_x;
        double* minimum_position_y;
        double* minimum_position_z;
        double* maximum_position_x;
        double* maximum_position_y;
        double* maximum_position_z;

        // maybe this can be cuda vectors, depends on if access stays coalesced
        double* position_x_excitatory_dendrite;
        double* position_y_excitatory_dendrite;
        double* position_z_excitatory_dendrite;
        double* position_x_inhibitory_dendrite;
        double* position_y_inhibitory_dendrite;
        double* position_z_inhibitory_dendrite;
        double* position_x_excitatory_axon;
        double* position_y_excitatory_axon;
        double* position_z_excitatory_axon;
        double* position_x_inhibitory_axon;
        double* position_y_inhibitory_axon;
        double* position_z_inhibitory_axon;

        unsigned int* num_free_elements_excitatory_dendrite;
        unsigned int* num_free_elements_inhibitory_dendrite;
        unsigned int* num_free_elements_excitatory_axon;
        unsigned int* num_free_elements_inhibitory_axon;

        Octree(const RelearnGPUTypes::number_neurons_type number_neurons, RelearnGPUTypes::number_neurons_type number_virtual_neurons) {
            
            neuron_ids = (uint64_t*)cuda_malloc(number_neurons * sizeof(uint64_t));

            /*child_index_1 = (uint64_t*)cuda_malloc(number_virtual_neurons * sizeof(uint64_t));
            child_index_2 = (uint64_t*)cuda_malloc(number_virtual_neurons * sizeof(uint64_t));
            child_index_3 = (uint64_t*)cuda_malloc(number_virtual_neurons * sizeof(uint64_t));
            child_index_4 = (uint64_t*)cuda_malloc(number_virtual_neurons * sizeof(uint64_t));
            child_index_5 = (uint64_t*)cuda_malloc(number_virtual_neurons * sizeof(uint64_t));
            child_index_6 = (uint64_t*)cuda_malloc(number_virtual_neurons * sizeof(uint64_t));
            child_index_7 = (uint64_t*)cuda_malloc(number_virtual_neurons * sizeof(uint64_t));
            child_index_8 = (uint64_t*)cuda_malloc(number_virtual_neurons * sizeof(uint64_t));*/
            child_indices = (uint64_t*)cuda_malloc(number_virtual_neurons * sizeof(uint64_t) * 8);

            minimum_position_x = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            minimum_position_y = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            minimum_position_z = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            maximum_position_x = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            maximum_position_y = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            maximum_position_z = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));

            position_x_excitatory_dendrite = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            position_y_excitatory_dendrite = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            position_z_excitatory_dendrite = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            position_x_inhibitory_dendrite = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            position_y_inhibitory_dendrite = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            position_z_inhibitory_dendrite = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            position_x_excitatory_axon = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            position_y_excitatory_axon = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            position_z_excitatory_axon = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            position_x_inhibitory_axon = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            position_y_inhibitory_axon = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            position_z_inhibitory_axon = (double*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double));
            num_free_elements_excitatory_dendrite = (unsigned int*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(unsigned int));
            num_free_elements_inhibitory_dendrite = (unsigned int*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(unsigned int));
            num_free_elements_excitatory_axon = (unsigned int*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(unsigned int));
            num_free_elements_inhibitory_axon = (unsigned int*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(unsigned int));
        }

        // brauchen vielleicht destructor der de-allocated?
    };

    class OctreeHandleImpl : public OctreeHandle {
    public:
        OctreeHandleImpl(const RelearnGPUTypes::number_neurons_type number_neurons, const RelearnGPUTypes::number_neurons_type number_virtual_neurons) {
            octree_dev_ptrs = new Octree(number_neurons, number_virtual_neurons);
        }

        ~OctreeHandleImpl() {
            delete octree_dev_ptrs;
        }

        void copy_to_GPU(OctreeCPUCopy&& octreeCPUCopy) {
            // copy the structure into GPU memory and save the device pointers in octree_dev_ptrs
        }

    private:
        Octree* octree_dev_ptrs;
    };

    std::unique_ptr<OctreeHandle> createOctree(const RelearnGPUTypes::number_neurons_type number_neurons, const RelearnGPUTypes::number_neurons_type number_virtual_neurons) {
        return std::make_shared<OctreeHandleImpl>(number_neurons, number_virtual_neurons);
    }
};