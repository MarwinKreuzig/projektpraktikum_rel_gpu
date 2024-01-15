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

    // TODO BRIEF
    [[nodiscard]] RelearnGPUTypes::number_neurons_type OctreeHandleImpl::get_number_neurons() const {
        return number_neurons;
    }

    /**
    * @brief Copies the GPU data structure version of the octree, which was constructed on the CPU, to the GPU
    * @param octree_cpu_copy Struct which holds the octree data to be copied on to the GPU
    */
    void OctreeHandleImpl::copy_to_gpu(OctreeCPUCopy&& octree_cpu_copy) {

        cuda_memcpy_to_device((void*)octree_dev_ptrs->neuron_ids, (void*)octree_cpu_copy.neuron_ids.data(), sizeof(uint64_t), octree_cpu_copy.neuron_ids.size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->child_indices, (void*)octree_cpu_copy.child_indices.data(), sizeof(uint64_t), octree_cpu_copy.child_indices.size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->num_children, (void*)octree_cpu_copy.num_children.data(), sizeof(unsigned int), octree_cpu_copy.num_children.size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->minimum_cell_position, (void*)octree_cpu_copy.minimum_cell_position.data(), sizeof(double3), octree_cpu_copy.minimum_cell_position.size());
        cuda_memcpy_to_device((void*)octree_dev_ptrs->maximum_cell_position, (void*)octree_cpu_copy.maximum_cell_position.data(), sizeof(double3), octree_cpu_copy.maximum_cell_position.size());
    
        cuda_memcpy_to_device((void*)octree_dev_ptrs->position_excitatory_element, (void*)octree_cpu_copy.position_excitatory_element.data(), sizeof(double3), octree_cpu_copy.position_excitatory_element.size());
        cuda_memcpy_to_device((void*)octree_dev_ptrs->position_inhibitory_element, (void*)octree_cpu_copy.position_inhibitory_element.data(), sizeof(double3), octree_cpu_copy.position_inhibitory_element.size());

        cuda_memcpy_to_device((void*)octree_dev_ptrs->num_free_elements_excitatory, (void*)octree_cpu_copy.num_free_elements_excitatory.data(), sizeof(unsigned int), octree_cpu_copy.num_free_elements_excitatory.size());
        cuda_memcpy_to_device((void*)octree_dev_ptrs->num_free_elements_inhibitory, (void*)octree_cpu_copy.num_free_elements_inhibitory.data(), sizeof(unsigned int), octree_cpu_copy.num_free_elements_inhibitory.size());
    }

    // TODO BRIEF
    void OctreeHandleImpl::copy_to_cpu(OctreeCPUCopy& octree_cpu_copy) {

        cuda_memcpy_to_host((void*)octree_dev_ptrs->neuron_ids, (void*)(octree_cpu_copy.neuron_ids.data()), sizeof(uint64_t), number_neurons);

        cuda_memcpy_to_host((void*)octree_dev_ptrs->child_indices, (void*)(octree_cpu_copy.child_indices.data()), sizeof(uint64_t), 8 * number_virtual_neurons);

        cuda_memcpy_to_host((void*)octree_dev_ptrs->num_children, (void*)(octree_cpu_copy.num_children.data()), sizeof(unsigned int), number_virtual_neurons);

        cuda_memcpy_to_host((void*)octree_dev_ptrs->minimum_cell_position, (void*)(octree_cpu_copy.minimum_cell_position.data()), sizeof(gpu::Vec3d), number_neurons + number_virtual_neurons);
        cuda_memcpy_to_host((void*)octree_dev_ptrs->maximum_cell_position, (void*)(octree_cpu_copy.maximum_cell_position.data()), sizeof(gpu::Vec3d), number_neurons + number_virtual_neurons);

        cuda_memcpy_to_host((void*)octree_dev_ptrs->position_excitatory_element, (void*)(octree_cpu_copy.position_excitatory_element.data()), sizeof(gpu::Vec3d), number_neurons + number_virtual_neurons);
        cuda_memcpy_to_host((void*)octree_dev_ptrs->position_inhibitory_element, (void*)(octree_cpu_copy.position_inhibitory_element.data()), sizeof(gpu::Vec3d), number_neurons + number_virtual_neurons);

        cuda_memcpy_to_host((void*)octree_dev_ptrs->num_free_elements_excitatory, (void*)(octree_cpu_copy.num_free_elements_excitatory.data()), sizeof(unsigned int), number_neurons + number_virtual_neurons);
        cuda_memcpy_to_host((void*)octree_dev_ptrs->num_free_elements_inhibitory, (void*)(octree_cpu_copy.num_free_elements_inhibitory.data()), sizeof(unsigned int), number_neurons + number_virtual_neurons);
    }

    /**
    * @brief Returns a shared pointer to a newly created handle to the Octree on the GPU
    * @param number_neurons Number of neurons, influences how much memory will be allocated on the GPU
    * @param number_virtual_neurons Number of virtual neurons, influences how much memory will be allocated on the GPU
    */
    std::shared_ptr<OctreeHandle> create_octree(RelearnGPUTypes::number_neurons_type number_neurons, RelearnGPUTypes::number_neurons_type number_virtual_neurons) {
        return std::make_shared<OctreeHandleImpl>(number_neurons, number_virtual_neurons);
    }
}