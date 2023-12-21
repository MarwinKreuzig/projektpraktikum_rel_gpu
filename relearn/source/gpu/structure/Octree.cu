#include "Octree.cuh"

namespace gpu::algorithm {

    /**
    * @brief Allocates the necessary memory for the octree on the GPU and saves the device pointers to this memory
    * @param number_neurons Number of neurons, influences how much memory will be allocated on the GPU
    * @param number_virtual_neurons Number of virtual neurons, influences how much memory will be allocated on the GPU
    */
    OctreeHandleImpl::OctreeHandleImpl(const RelearnGPUTypes::number_neurons_type number_neurons, const RelearnGPUTypes::number_neurons_type number_virtual_neurons)
        : number_neurons(number_neurons), number_virtual_neurons(number_virtual_neurons) 
    {
        octree_dev_ptrs = new Octree(number_neurons, number_virtual_neurons);
    }

    OctreeHandleImpl::~OctreeHandleImpl() {
        delete octree_dev_ptrs;
    }

    /**
    * @brief Returns the number of virtual neuronss on the octree on the GPU
    * @return The number of virtual neurons on the tree
    */
    [[nodiscard]] RelearnGPUTypes::number_neurons_type OctreeHandleImpl::get_number_virtual_neurons() const {
        return number_virtual_neurons;
    }

    /**
    * @brief Copies the GPU data structure version of the octree, which was constructed on the CPU, to the GPU
    * @param octree_cpu_copy Struct which holds the octree data to be copied on to the GPU
    */
    void OctreeHandleImpl::copy_to_gpu(OctreeCPUCopy&& octree_cpu_copy) {

        cuda_memcpy_to_device((void*)octree_dev_ptrs->neuron_ids, (void*)octree_cpu_copy.neuron_ids.data(), sizeof(uint64_t), octree_cpu_copy.neuron_ids.size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->child_indices, (void*)octree_cpu_copy.child_indices[0].data(), sizeof(uint64_t), octree_cpu_copy.child_indices[0].size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octree_cpu_copy.child_indices[0].size()), (void*)octree_cpu_copy.child_indices[1].data(), sizeof(uint64_t), octree_cpu_copy.child_indices[1].size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octree_cpu_copy.child_indices[1].size()), (void*)octree_cpu_copy.child_indices[2].data(), sizeof(uint64_t), octree_cpu_copy.child_indices[2].size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octree_cpu_copy.child_indices[2].size()), (void*)octree_cpu_copy.child_indices[3].data(), sizeof(uint64_t), octree_cpu_copy.child_indices[3].size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octree_cpu_copy.child_indices[3].size()), (void*)octree_cpu_copy.child_indices[4].data(), sizeof(uint64_t), octree_cpu_copy.child_indices[4].size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octree_cpu_copy.child_indices[4].size()), (void*)octree_cpu_copy.child_indices[5].data(), sizeof(uint64_t), octree_cpu_copy.child_indices[5].size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octree_cpu_copy.child_indices[5].size()), (void*)octree_cpu_copy.child_indices[6].data(), sizeof(uint64_t), octree_cpu_copy.child_indices[6].size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->child_indices + octree_cpu_copy.child_indices[6].size()), (void*)octree_cpu_copy.child_indices[7].data(), sizeof(uint64_t), octree_cpu_copy.child_indices[7].size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->num_children, (void*)octree_cpu_copy.num_children.data(), sizeof(unsigned int), octree_cpu_copy.num_children.size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->minimum_cell_position, (void*)octree_cpu_copy.minimum_cell_position.data(), sizeof(double3), octree_cpu_copy.minimum_cell_position.size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->minimum_cell_position + octree_cpu_copy.minimum_cell_position.size()), (void*)octree_cpu_copy.minimum_cell_position_virtual.data(), sizeof(double3), octree_cpu_copy.minimum_cell_position_virtual.size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->maximum_cell_position, (void*)octree_cpu_copy.maximum_cell_position.data(), sizeof(double3), octree_cpu_copy.maximum_cell_position.size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->maximum_cell_position + octree_cpu_copy.maximum_cell_position.size()), (void*)octree_cpu_copy.maximum_cell_position_virtual.data(), sizeof(double3), octree_cpu_copy.maximum_cell_position_virtual.size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->position_excitatory_element, (void*)octree_cpu_copy.position_excitatory_element.data(), sizeof(double3), octree_cpu_copy.position_excitatory_element.size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->position_excitatory_element + octree_cpu_copy.position_excitatory_element.size()), (void*)octree_cpu_copy.position_excitatory_element_virtual.data(), sizeof(double3), octree_cpu_copy.position_excitatory_element_virtual.size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->position_inhibitory_element, (void*)octree_cpu_copy.position_inhibitory_element.data(), sizeof(double3), octree_cpu_copy.position_inhibitory_element.size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->position_inhibitory_element + octree_cpu_copy.position_inhibitory_element.size()), (void*)octree_cpu_copy.position_inhibitory_element_virtual.data(), sizeof(double3), octree_cpu_copy.position_inhibitory_element_virtual.size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->num_free_elements_excitatory, (void*)octree_cpu_copy.num_free_elements_excitatory.data(), sizeof(unsigned int), octree_cpu_copy.num_free_elements_excitatory.size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->num_free_elements_excitatory + octree_cpu_copy.num_free_elements_excitatory.size()), (void*)octree_cpu_copy.num_free_elements_excitatory_virtual.data(), sizeof(unsigned int), octree_cpu_copy.num_free_elements_excitatory_virtual.size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->num_free_elements_inhibitory, (void*)octree_cpu_copy.num_free_elements_inhibitory.data(), sizeof(unsigned int), octree_cpu_copy.num_free_elements_inhibitory.size());
        cuda_memcpy_to_device((void*)(octree_dev_ptrs->num_free_elements_inhibitory + octree_cpu_copy.num_free_elements_inhibitory.size()), (void*)octree_cpu_copy.num_free_elements_inhibitory_virtual.data(), sizeof(unsigned int), octree_cpu_copy.num_free_elements_inhibitory_virtual.size());
    }

    void OctreeHandleImpl::copy_to_cpu(OctreeCPUCopy& octree_cpu_copy) {
                
        // this would have been easier if the octree_cpu_copy was made out of arrays and not vectors

        uint64_t neuron_ids[number_neurons];
        cuda_memcpy_to_host((void*)octree_dev_ptrs->neuron_ids, (void*)(&neuron_ids[0]), sizeof(uint64_t), number_neurons);
        octree_cpu_copy.neuron_ids.insert(octree_cpu_copy.neuron_ids.end(), &neuron_ids[0], &neuron_ids[number_neurons]);

        //std::array<std::vector<uint64_t>, 8> child_indices;
        uint64_t child_indices[8 * number_virtual_neurons];
        cuda_memcpy_to_host((void*)octree_dev_ptrs->child_indices, (void*)(&child_indices[0]), sizeof(uint64_t), 8 * number_virtual_neurons);
        octree_cpu_copy.child_indices[0].insert(octree_cpu_copy.child_indices[0].end(), &child_indices[0], &child_indices[number_virtual_neurons]);
        octree_cpu_copy.child_indices[1].insert(octree_cpu_copy.child_indices[1].end(), &child_indices[number_virtual_neurons], &child_indices[number_virtual_neurons * 2]);
        octree_cpu_copy.child_indices[2].insert(octree_cpu_copy.child_indices[2].end(), &child_indices[number_virtual_neurons * 2], &child_indices[number_virtual_neurons * 3]);
        octree_cpu_copy.child_indices[3].insert(octree_cpu_copy.child_indices[3].end(), &child_indices[number_virtual_neurons * 3], &child_indices[number_virtual_neurons * 4]);
        octree_cpu_copy.child_indices[4].insert(octree_cpu_copy.child_indices[4].end(), &child_indices[number_virtual_neurons * 4], &child_indices[number_virtual_neurons * 5]);
        octree_cpu_copy.child_indices[5].insert(octree_cpu_copy.child_indices[5].end(), &child_indices[number_virtual_neurons * 5], &child_indices[number_virtual_neurons * 6]);
        octree_cpu_copy.child_indices[6].insert(octree_cpu_copy.child_indices[6].end(), &child_indices[number_virtual_neurons * 6], &child_indices[number_virtual_neurons * 7]);
        octree_cpu_copy.child_indices[7].insert(octree_cpu_copy.child_indices[7].end(), &child_indices[number_virtual_neurons * 7], &child_indices[number_virtual_neurons * 8]);

        unsigned int num_children[number_virtual_neurons];
        cuda_memcpy_to_host((void*)octree_dev_ptrs->num_children, (void*)(&num_children[0]), sizeof(unsigned int), number_virtual_neurons);
        octree_cpu_copy.num_children.insert(octree_cpu_copy.num_children.end(), &num_children[0], &num_children[number_virtual_neurons]);

        gpu::Vec3d minimum_cell_position[number_neurons];
        cuda_memcpy_to_host((void*)octree_dev_ptrs->minimum_cell_position, (void*)(&minimum_cell_position[0]), sizeof(gpu::Vec3d), number_neurons);
        octree_cpu_copy.minimum_cell_position.insert(octree_cpu_copy.minimum_cell_position.end(), &minimum_cell_position[0], &minimum_cell_position[number_neurons]);

        gpu::Vec3d minimum_cell_position_virtual[number_virtual_neurons];
        cuda_memcpy_to_host((void*)(octree_dev_ptrs->minimum_cell_position + number_neurons), (void*)(&minimum_cell_position_virtual[0]), sizeof(gpu::Vec3d), number_virtual_neurons);
        octree_cpu_copy.minimum_cell_position_virtual.insert(octree_cpu_copy.minimum_cell_position_virtual.end(), &minimum_cell_position_virtual[0], &minimum_cell_position_virtual[number_virtual_neurons]);

        gpu::Vec3d maximum_cell_position[number_neurons];
        cuda_memcpy_to_host((void*)octree_dev_ptrs->maximum_cell_position, (void*)(&maximum_cell_position[0]), sizeof(gpu::Vec3d), number_neurons);
        octree_cpu_copy.maximum_cell_position.insert(octree_cpu_copy.maximum_cell_position.end(), &maximum_cell_position[0], &maximum_cell_position[number_neurons]);

        gpu::Vec3d maximum_cell_position_virtual[number_virtual_neurons];
        cuda_memcpy_to_host((void*)(octree_dev_ptrs->maximum_cell_position + number_neurons), (void*)(&maximum_cell_position_virtual[0]), sizeof(gpu::Vec3d), number_virtual_neurons);
        octree_cpu_copy.maximum_cell_position_virtual.insert(octree_cpu_copy.maximum_cell_position_virtual.end(), &maximum_cell_position_virtual[0], &maximum_cell_position_virtual[number_virtual_neurons]);

        gpu::Vec3d position_excitatory_element[number_neurons];
        cuda_memcpy_to_host((void*)octree_dev_ptrs->position_excitatory_element, (void*)(&position_excitatory_element[0]), sizeof(gpu::Vec3d), number_neurons);
        octree_cpu_copy.position_excitatory_element.insert(octree_cpu_copy.position_excitatory_element.end(), &position_excitatory_element[0], &position_excitatory_element[number_neurons]);

        gpu::Vec3d position_excitatory_element_virtual[number_virtual_neurons];
        cuda_memcpy_to_host((void*)(octree_dev_ptrs->position_excitatory_element + number_neurons), (void*)(&position_excitatory_element_virtual[0]), sizeof(gpu::Vec3d), number_virtual_neurons);
        octree_cpu_copy.position_excitatory_element_virtual.insert(octree_cpu_copy.position_excitatory_element_virtual.end(), &position_excitatory_element_virtual[0], &position_excitatory_element_virtual[number_virtual_neurons]);

        gpu::Vec3d position_inhibitory_element[number_neurons];
        cuda_memcpy_to_host((void*)octree_dev_ptrs->position_inhibitory_element, (void*)(&position_inhibitory_element[0]), sizeof(gpu::Vec3d), number_neurons);
        octree_cpu_copy.position_inhibitory_element.insert(octree_cpu_copy.position_inhibitory_element.end(), &position_inhibitory_element[0], &position_inhibitory_element[number_neurons]);

        gpu::Vec3d position_inhibitory_element_virtual[number_virtual_neurons];
        cuda_memcpy_to_host((void*)(octree_dev_ptrs->position_inhibitory_element + number_neurons), (void*)(&position_inhibitory_element_virtual[0]), sizeof(gpu::Vec3d), number_virtual_neurons);
        octree_cpu_copy.position_inhibitory_element_virtual.insert(octree_cpu_copy.position_inhibitory_element_virtual.end(), &position_inhibitory_element_virtual[0], &position_inhibitory_element_virtual[number_virtual_neurons]);

        unsigned int num_free_elements_excitatory[number_neurons];
        cuda_memcpy_to_host((void*)octree_dev_ptrs->num_free_elements_excitatory, (void*)(&num_free_elements_excitatory[0]), sizeof(unsigned int), number_neurons);
        octree_cpu_copy.num_free_elements_excitatory.insert(octree_cpu_copy.num_free_elements_excitatory.end(), &num_free_elements_excitatory[0], &num_free_elements_excitatory[number_neurons]);

        unsigned int num_free_elements_excitatory_virtual[number_virtual_neurons];
        cuda_memcpy_to_host((void*)(octree_dev_ptrs->num_free_elements_excitatory + number_neurons), (void*)(&num_free_elements_excitatory_virtual[0]), sizeof(unsigned int), number_virtual_neurons);
        octree_cpu_copy.num_free_elements_excitatory_virtual.insert(octree_cpu_copy.num_free_elements_excitatory_virtual.end(), &num_free_elements_excitatory_virtual[0], &num_free_elements_excitatory_virtual[number_virtual_neurons]);

        unsigned int num_free_elements_inhibitory[number_neurons];
        cuda_memcpy_to_host((void*)octree_dev_ptrs->num_free_elements_inhibitory, (void*)(&num_free_elements_inhibitory[0]), sizeof(unsigned int), number_neurons);
        octree_cpu_copy.num_free_elements_inhibitory.insert(octree_cpu_copy.num_free_elements_inhibitory.end(), &num_free_elements_inhibitory[0], &num_free_elements_inhibitory[number_neurons]);

        unsigned int num_free_elements_inhibitory_virtual[number_virtual_neurons];
        cuda_memcpy_to_host((void*)(octree_dev_ptrs->num_free_elements_inhibitory + number_neurons), (void*)(&num_free_elements_inhibitory_virtual[0]), sizeof(unsigned int), number_virtual_neurons);
        octree_cpu_copy.num_free_elements_inhibitory_virtual.insert(octree_cpu_copy.num_free_elements_inhibitory_virtual.end(), &num_free_elements_inhibitory_virtual[0], &num_free_elements_inhibitory_virtual[number_virtual_neurons]);
    }
}