#pragma once

#include "Commons.cuh"
#include "gpu/GpuTypes.h"
#include "gpu/Interface.h"
#include "NeuronsExtraInfos.cuh"

#include "CudaArray.cuh"
#include "CudaVector.cuh"

#include <numeric>

namespace gpu::algorithm {

    // Do it like this: indexes are always in the combined range with neurons and virtual neurons with neurons in the front
    // when trying to access child_indices, num_neurons has to be subtracted from the index.
    struct Octree {
        uint64_t* neuron_ids;

        uint64_t* child_indices;

        // we need this, since we can't use -1 to indicate that there is no child there
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

        /**
        * @brief Allocates the necessary memory for the octree on the GPU and saves the device pointers to this memory
        * @param number_neurons Number of neurons, influences how much memory will be allocated on the GPU
        * @param number_virtual_neurons Number of virtual neurons, influences how much memory will be allocated on the GPU
        */
        OctreeHandleImpl(const RelearnGPUTypes::number_neurons_type number_neurons, const RelearnGPUTypes::number_neurons_type number_virtual_neurons) {
            octree_dev_ptrs = new Octree(number_neurons, number_virtual_neurons);
        }

        ~OctreeHandleImpl() {
            delete octree_dev_ptrs;
        }

        /**
        * @brief Copies the GPU data structure version of the octree, which was constructed on the CPU, to the GPU
        * @param octreeCPUCopy Struct which holds the octree data to be copied on to the GPU
        */
        void copy_to_GPU(OctreeCPUCopy&& octreeCPUCopy) {
            // the virtual data might have to be flipped in order to adhere to the requirements in the paper, depending on how marvin parses it

            cuda_memcpy_to_device((void*)octree_dev_ptrs->neuron_ids, (void*)octreeCPUCopy.neuron_ids.data(), sizeof(uint64_t), octreeCPUCopy.neuron_ids.size());

            cuda_memcpy_to_device((void*)octree_dev_ptrs->child_indices, (void*)octreeCPUCopy.child_indices[0].data(), sizeof(uint64_t), octreeCPUCopy.child_indices[0].size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octreeCPUCopy.child_indices[0].size()), (void*)octreeCPUCopy.child_indices[1].data(), sizeof(uint64_t), octreeCPUCopy.child_indices[1].size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octreeCPUCopy.child_indices[1].size()), (void*)octreeCPUCopy.child_indices[2].data(), sizeof(uint64_t), octreeCPUCopy.child_indices[2].size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octreeCPUCopy.child_indices[2].size()), (void*)octreeCPUCopy.child_indices[3].data(), sizeof(uint64_t), octreeCPUCopy.child_indices[3].size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octreeCPUCopy.child_indices[3].size()), (void*)octreeCPUCopy.child_indices[4].data(), sizeof(uint64_t), octreeCPUCopy.child_indices[4].size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octreeCPUCopy.child_indices[4].size()), (void*)octreeCPUCopy.child_indices[5].data(), sizeof(uint64_t), octreeCPUCopy.child_indices[5].size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octreeCPUCopy.child_indices[5].size()), (void*)octreeCPUCopy.child_indices[6].data(), sizeof(uint64_t), octreeCPUCopy.child_indices[6].size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octreeCPUCopy.child_indices[6].size()), (void*)octreeCPUCopy.child_indices[7].data(), sizeof(uint64_t), octreeCPUCopy.child_indices[7].size());

            cuda_memcpy_to_device((void*)octree_dev_ptrs->num_children, (void*)octreeCPUCopy.num_children.data(), sizeof(unsigned int), octreeCPUCopy.num_children.size());

            cuda_memcpy_to_device((void*)octree_dev_ptrs->minimum_cell_position, (void*)octreeCPUCopy.minimum_cell_position.data(), sizeof(double3), octreeCPUCopy.minimum_cell_position.size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->minimum_cell_position + octreeCPUCopy.minimum_cell_position.size()), (void*)octreeCPUCopy.minimum_cell_position_virtual.data(), sizeof(double3), octreeCPUCopy.minimum_cell_position_virtual.size());

            cuda_memcpy_to_device((void*)octree_dev_ptrs->maximum_cell_position, (void*)octreeCPUCopy.maximum_cell_position.data(), sizeof(double3), octreeCPUCopy.maximum_cell_position.size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->maximum_cell_position + octreeCPUCopy.maximum_cell_position.size()), (void*)octreeCPUCopy.maximum_cell_position_virtual.data(), sizeof(double3), octreeCPUCopy.maximum_cell_position_virtual.size());

            cuda_memcpy_to_device((void*)octree_dev_ptrs->position_excitatory_element, (void*)octreeCPUCopy.position_excitatory_element.data(), sizeof(double3), octreeCPUCopy.position_excitatory_element.size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->position_excitatory_element + octreeCPUCopy.position_excitatory_element.size()), (void*)octreeCPUCopy.position_excitatory_element_virtual.data(), sizeof(double3), octreeCPUCopy.position_excitatory_element_virtual.size());

            cuda_memcpy_to_device((void*)octree_dev_ptrs->position_inhibitory_element, (void*)octreeCPUCopy.position_inhibitory_element.data(), sizeof(double3), octreeCPUCopy.position_inhibitory_element.size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->position_inhibitory_element + octreeCPUCopy.position_inhibitory_element.size()), (void*)octreeCPUCopy.position_inhibitory_element_virtual.data(), sizeof(double3), octreeCPUCopy.position_inhibitory_element_virtual.size());

            cuda_memcpy_to_device((void*)octree_dev_ptrs->num_free_elements_excitatory, (void*)octreeCPUCopy.num_free_elements_excitatory.data(), sizeof(unsigned int), octreeCPUCopy.num_free_elements_excitatory.size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->num_free_elements_excitatory + octreeCPUCopy.num_free_elements_excitatory.size()), (void*)octreeCPUCopy.num_free_elements_excitatory_virtual.data(), sizeof(unsigned int), octreeCPUCopy.num_free_elements_excitatory_virtual.size());

            cuda_memcpy_to_device((void*)octree_dev_ptrs->num_free_elements_inhibitory, (void*)octreeCPUCopy.num_free_elements_inhibitory.data(), sizeof(unsigned int), octreeCPUCopy.num_free_elements_inhibitory.size());
            cuda_memcpy_to_device((void*)(octree_dev_ptrs->num_free_elements_inhibitory + octreeCPUCopy.num_free_elements_inhibitory.size()), (void*)octreeCPUCopy.num_free_elements_inhibitory_virtual.data(), sizeof(unsigned int), octreeCPUCopy.num_free_elements_inhibitory_virtual.size());
        }

    private:
        Octree* octree_dev_ptrs;
    };

    /**
    * @brief Returns a shared pointer to a newly created handle to the Octree on the GPU
    * @param number_neurons Number of neurons, influences how much memory will be allocated on the GPU
    * @param number_virtual_neurons Number of virtual neurons, influences how much memory will be allocated on the GPU
    */
    std::shared_ptr<OctreeHandle> createOctree(const RelearnGPUTypes::number_neurons_type number_neurons, const RelearnGPUTypes::number_neurons_type number_virtual_neurons) {
        return std::make_shared<OctreeHandleImpl>(number_neurons, number_virtual_neurons);
    }
};