#pragma once

#include "Commons.cuh"
#include "utils/GpuTypes.h"
#include "utils/Interface.h"

#include "CudaArray.cuh"
#include "CudaVector.cuh"

//#include <numeric>

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
        OctreeHandleImpl(const RelearnGPUTypes::number_neurons_type number_neurons, const RelearnGPUTypes::number_neurons_type number_virtual_neurons)
            : number_neurons(number_neurons), number_virtual_neurons(number_virtual_neurons) 
        {
            octree_dev_ptrs = new Octree(number_neurons, number_virtual_neurons);
        }

        ~OctreeHandleImpl() {
            delete octree_dev_ptrs;
        }

        /**
        * @brief Returns the number of virtual neuronss on the octree on the GPU
        * @return The number of virtual neurons on the tree
        */
        [[nodiscard]] RelearnGPUTypes::number_neurons_type get_number_virtual_neurons() const {
            return number_virtual_neurons;
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

        void copy_to_CPU(OctreeCPUCopy& octreeCPUCopy) {
            
            // this would have been easier if the octreeCPUCopy was made out of arrays and not vectors

            uint64_t neuron_ids[number_neurons];
            cuda_memcpy_to_host((void*)octree_dev_ptrs->neuron_ids, (void*)(&neuron_ids[0]), sizeof(uint64_t), number_neurons);
            octreeCPUCopy.neuron_ids.insert(octreeCPUCopy.neuron_ids.end(), &neuron_ids[0], &neuron_ids[number_neurons]);

            //std::array<std::vector<uint64_t>, 8> child_indices;
            uint64_t child_indices[8 * number_virtual_neurons];
            cuda_memcpy_to_host((void*)octree_dev_ptrs->child_indices, (void*)(&child_indices[0]), sizeof(uint64_t), 8 * number_virtual_neurons);
            octreeCPUCopy.child_indices[0].insert(octreeCPUCopy.child_indices[0].end(), &child_indices[0], &child_indices[number_virtual_neurons]);
            octreeCPUCopy.child_indices[1].insert(octreeCPUCopy.child_indices[1].end(), &child_indices[number_virtual_neurons], &child_indices[number_virtual_neurons * 2]);
            octreeCPUCopy.child_indices[2].insert(octreeCPUCopy.child_indices[2].end(), &child_indices[number_virtual_neurons * 2], &child_indices[number_virtual_neurons * 3]);
            octreeCPUCopy.child_indices[3].insert(octreeCPUCopy.child_indices[3].end(), &child_indices[number_virtual_neurons * 3], &child_indices[number_virtual_neurons * 4]);
            octreeCPUCopy.child_indices[4].insert(octreeCPUCopy.child_indices[4].end(), &child_indices[number_virtual_neurons * 4], &child_indices[number_virtual_neurons * 5]);
            octreeCPUCopy.child_indices[5].insert(octreeCPUCopy.child_indices[5].end(), &child_indices[number_virtual_neurons * 5], &child_indices[number_virtual_neurons * 6]);
            octreeCPUCopy.child_indices[6].insert(octreeCPUCopy.child_indices[6].end(), &child_indices[number_virtual_neurons * 6], &child_indices[number_virtual_neurons * 7]);
            octreeCPUCopy.child_indices[7].insert(octreeCPUCopy.child_indices[7].end(), &child_indices[number_virtual_neurons * 7], &child_indices[number_virtual_neurons * 8]);

            unsigned int num_children[number_virtual_neurons];
            cuda_memcpy_to_host((void*)octree_dev_ptrs->num_children, (void*)(&num_children[0]), sizeof(unsigned int), number_virtual_neurons);
            octreeCPUCopy.num_children.insert(octreeCPUCopy.num_children.end(), &num_children[0], &num_children[number_virtual_neurons]);

            gpu::Vec3d minimum_cell_position[number_neurons];
            cuda_memcpy_to_host((void*)octree_dev_ptrs->minimum_cell_position, (void*)(&minimum_cell_position[0]), sizeof(gpu::Vec3d), number_neurons);
            octreeCPUCopy.minimum_cell_position.insert(octreeCPUCopy.minimum_cell_position.end(), &minimum_cell_position[0], &minimum_cell_position[number_neurons]);

            gpu::Vec3d minimum_cell_position_virtual[number_virtual_neurons];
            cuda_memcpy_to_host((void*)(octree_dev_ptrs->minimum_cell_position + number_neurons), (void*)(&minimum_cell_position_virtual[0]), sizeof(gpu::Vec3d), number_virtual_neurons);
            octreeCPUCopy.minimum_cell_position_virtual.insert(octreeCPUCopy.minimum_cell_position_virtual.end(), &minimum_cell_position_virtual[0], &minimum_cell_position_virtual[number_virtual_neurons]);

            gpu::Vec3d maximum_cell_position[number_neurons];
            cuda_memcpy_to_host((void*)octree_dev_ptrs->maximum_cell_position, (void*)(&maximum_cell_position[0]), sizeof(gpu::Vec3d), number_neurons);
            octreeCPUCopy.maximum_cell_position.insert(octreeCPUCopy.maximum_cell_position.end(), &maximum_cell_position[0], &maximum_cell_position[number_neurons]);

            gpu::Vec3d maximum_cell_position_virtual[number_virtual_neurons];
            cuda_memcpy_to_host((void*)(octree_dev_ptrs->maximum_cell_position + number_neurons), (void*)(&maximum_cell_position_virtual[0]), sizeof(gpu::Vec3d), number_virtual_neurons);
            octreeCPUCopy.maximum_cell_position_virtual.insert(octreeCPUCopy.maximum_cell_position_virtual.end(), &maximum_cell_position_virtual[0], &maximum_cell_position_virtual[number_virtual_neurons]);

            gpu::Vec3d position_excitatory_element[number_neurons];
            cuda_memcpy_to_host((void*)octree_dev_ptrs->position_excitatory_element, (void*)(&position_excitatory_element[0]), sizeof(gpu::Vec3d), number_neurons);
            octreeCPUCopy.position_excitatory_element.insert(octreeCPUCopy.position_excitatory_element.end(), &position_excitatory_element[0], &position_excitatory_element[number_neurons]);

            gpu::Vec3d position_excitatory_element_virtual[number_virtual_neurons];
            cuda_memcpy_to_host((void*)(octree_dev_ptrs->position_excitatory_element + number_neurons), (void*)(&position_excitatory_element_virtual[0]), sizeof(gpu::Vec3d), number_virtual_neurons);
            octreeCPUCopy.position_excitatory_element_virtual.insert(octreeCPUCopy.position_excitatory_element_virtual.end(), &position_excitatory_element_virtual[0], &position_excitatory_element_virtual[number_virtual_neurons]);

            gpu::Vec3d position_inhibitory_element[number_neurons];
            cuda_memcpy_to_host((void*)octree_dev_ptrs->position_inhibitory_element, (void*)(&position_inhibitory_element[0]), sizeof(gpu::Vec3d), number_neurons);
            octreeCPUCopy.position_inhibitory_element.insert(octreeCPUCopy.position_inhibitory_element.end(), &position_inhibitory_element[0], &position_inhibitory_element[number_neurons]);

            gpu::Vec3d position_inhibitory_element_virtual[number_virtual_neurons];
            cuda_memcpy_to_host((void*)(octree_dev_ptrs->position_inhibitory_element + number_neurons), (void*)(&position_inhibitory_element_virtual[0]), sizeof(gpu::Vec3d), number_virtual_neurons);
            octreeCPUCopy.position_inhibitory_element_virtual.insert(octreeCPUCopy.position_inhibitory_element_virtual.end(), &position_inhibitory_element_virtual[0], &position_inhibitory_element_virtual[number_virtual_neurons]);

            unsigned int num_free_elements_excitatory[number_neurons];
            cuda_memcpy_to_host((void*)octree_dev_ptrs->num_free_elements_excitatory, (void*)(&num_free_elements_excitatory[0]), sizeof(unsigned int), number_neurons);
            octreeCPUCopy.num_free_elements_excitatory.insert(octreeCPUCopy.num_free_elements_excitatory.end(), &num_free_elements_excitatory[0], &num_free_elements_excitatory[number_neurons]);

            unsigned int num_free_elements_excitatory_virtual[number_virtual_neurons];
            cuda_memcpy_to_host((void*)(octree_dev_ptrs->num_free_elements_excitatory + number_neurons), (void*)(&num_free_elements_excitatory_virtual[0]), sizeof(unsigned int), number_virtual_neurons);
            octreeCPUCopy.num_free_elements_excitatory_virtual.insert(octreeCPUCopy.num_free_elements_excitatory_virtual.end(), &num_free_elements_excitatory_virtual[0], &num_free_elements_excitatory_virtual[number_virtual_neurons]);

            unsigned int num_free_elements_inhibitory[number_neurons];
            cuda_memcpy_to_host((void*)octree_dev_ptrs->num_free_elements_inhibitory, (void*)(&num_free_elements_inhibitory[0]), sizeof(unsigned int), number_neurons);
            octreeCPUCopy.num_free_elements_inhibitory.insert(octreeCPUCopy.num_free_elements_inhibitory.end(), &num_free_elements_inhibitory[0], &num_free_elements_inhibitory[number_neurons]);

            unsigned int num_free_elements_inhibitory_virtual[number_virtual_neurons];
            cuda_memcpy_to_host((void*)(octree_dev_ptrs->num_free_elements_inhibitory + number_neurons), (void*)(&num_free_elements_inhibitory_virtual[0]), sizeof(unsigned int), number_virtual_neurons);
            octreeCPUCopy.num_free_elements_inhibitory_virtual.insert(octreeCPUCopy.num_free_elements_inhibitory_virtual.end(), &num_free_elements_inhibitory_virtual[0], &num_free_elements_inhibitory_virtual[number_virtual_neurons]);
        }

    private:
        Octree* octree_dev_ptrs;

        RelearnGPUTypes::number_neurons_type number_neurons;
        RelearnGPUTypes::number_neurons_type number_virtual_neurons;
    };

    /**
    * @brief Returns a shared pointer to a newly created handle to the Octree on the GPU
    * @param number_neurons Number of neurons, influences how much memory will be allocated on the GPU
    * @param number_virtual_neurons Number of virtual neurons, influences how much memory will be allocated on the GPU
    */
    std::shared_ptr<OctreeHandle> createOctree(RelearnGPUTypes::number_neurons_type number_neurons, RelearnGPUTypes::number_neurons_type number_virtual_neurons) {
        return std::make_shared<OctreeHandleImpl>(number_neurons, number_virtual_neurons);
    }
};
