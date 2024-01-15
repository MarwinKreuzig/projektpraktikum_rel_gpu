#pragma once

#include "Commons.cuh"
#include "utils/GpuTypes.h"
#include "utils/Interface.h"

#include "CudaArray.cuh"
#include "CudaVector.cuh"

namespace gpu::algorithm {

    // Do it like this: indexes are always in the combined range with neurons and virtual neurons with neurons in the front
    // when trying to access child_indices, num_neurons has to be subtracted from the index.
    // TODO BRIEF, add the above comment to brief
    struct Octree {
        uint64_t* neuron_ids;

        uint64_t* child_indices;

        // we need this, since we can't use -1 to indicate an invalid child_indices entry
        unsigned int* num_children;

        double3* minimum_cell_position;
        double3* maximum_cell_position;

        double3* position_excitatory_element;
        double3* position_inhibitory_element;

        // depending on if barnes hut or inverse barnes hut will be done, these are either dendrites or axons
        unsigned int* num_free_elements_excitatory;
        unsigned int* num_free_elements_inhibitory;

        /**
        * @brief Allocates the necessary memory to hold all the data that is needed for the Octree on the GPU
        * @param number_neurons Number of neurons, influences how much memory will be allocated on the GPU
        * @param number_virtual_neurons Number of virtual neurons, influences how much memory will be allocated on the GPU
        */
        Octree(const RelearnGPUTypes::number_neurons_type number_neurons, RelearnGPUTypes::number_neurons_type number_virtual_neurons) {

            neuron_ids = (uint64_t*)cuda_malloc(number_neurons * sizeof(uint64_t));

            child_indices = (uint64_t*)cuda_malloc(number_virtual_neurons * sizeof(uint64_t) * 8);

            num_children = (unsigned int*)cuda_malloc(number_virtual_neurons * sizeof(unsigned int));

            minimum_cell_position = (double3*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double3));
            maximum_cell_position = (double3*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double3));

            position_excitatory_element = (double3*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double3));
            position_inhibitory_element = (double3*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(double3));

            num_free_elements_excitatory = (unsigned int*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(unsigned int));
            num_free_elements_inhibitory = (unsigned int*)cuda_malloc((number_virtual_neurons + number_neurons) * sizeof(unsigned int));
        }

        ~Octree() {

            cudaFree(neuron_ids);
            cudaFree(child_indices);
            cudaFree(num_children);
            cudaFree(minimum_cell_position);
            cudaFree(maximum_cell_position);
            cudaFree(position_excitatory_element);
            cudaFree(position_inhibitory_element);
            cudaFree(num_free_elements_excitatory);
            cudaFree(num_free_elements_inhibitory);
        }
    };

    class OctreeHandleImpl : public OctreeHandle {
    public:

        OctreeHandleImpl(const RelearnGPUTypes::number_neurons_type number_neurons, const RelearnGPUTypes::number_neurons_type number_virtual_neurons);

        ~OctreeHandleImpl();

        [[nodiscard]] RelearnGPUTypes::number_neurons_type get_number_virtual_neurons() const;
        [[nodiscard]] RelearnGPUTypes::number_neurons_type get_number_neurons() const;

        void copy_to_gpu(OctreeCPUCopy&& octree_cpu_copy);
        void copy_to_cpu(OctreeCPUCopy& octree_cpu_copy);

    private:
        Octree* octree_dev_ptrs;

        RelearnGPUTypes::number_neurons_type number_neurons;
        RelearnGPUTypes::number_neurons_type number_virtual_neurons;
    };
};
