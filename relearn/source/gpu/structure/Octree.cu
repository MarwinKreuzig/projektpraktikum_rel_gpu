#include "Octree.cuh"

namespace gpu::algorithm {
    /**
    * @brief Allocates the necessary memory for the octree on the GPU and saves the device pointers to this memory
    * @param number_neurons Number of neurons, influences how much memory will be allocated on the GPU
    * @param number_virtual_neurons Number of virtual neurons, influences how much memory will be allocated on the GPU
    */
    OctreeHandleImpl::OctreeHandleImpl(Octree* dev_ptr, const RelearnGPUTypes::number_neurons_type number_neurons, const RelearnGPUTypes::number_neurons_type number_virtual_neurons, ElementType stored_element_type)
        : number_neurons(number_neurons), number_virtual_neurons(number_virtual_neurons), octree_dev_ptr(dev_ptr)
    {
        _init(stored_element_type);
    }

    void OctreeHandleImpl::_init(ElementType stored_element_type) {
        void* neuron_ids_ptr = execute_and_copy<void*>([=] __device__(Octree* octree) { return (void*)&octree->neuron_ids; }, octree_dev_ptr);
        handle_neuron_ids = gpu::Vector::CudaArrayDeviceHandle<uint64_t>(neuron_ids_ptr);
        handle_neuron_ids.resize(number_neurons);

        void* child_indices_ptr = execute_and_copy<void*>([=] __device__(Octree* octree) { return (void*)&octree->child_indices; }, octree_dev_ptr);
        handle_child_indices = gpu::Vector::CudaArrayDeviceHandle<uint64_t>(child_indices_ptr);
        handle_child_indices.resize(number_virtual_neurons * 8);

        void* num_children_ptr = execute_and_copy<void*>([=] __device__(Octree* octree) { return (void*)&octree->num_children; }, octree_dev_ptr);
        handle_num_children = gpu::Vector::CudaArrayDeviceHandle<unsigned int>(num_children_ptr);
        handle_num_children.resize(number_virtual_neurons);

        void* minimum_cell_position_ptr = execute_and_copy<void*>([=] __device__(Octree* octree) { return (void*)&octree->minimum_cell_position; }, octree_dev_ptr);
        handle_minimum_cell_position = gpu::Vector::CudaArrayDeviceHandle<double3>(minimum_cell_position_ptr);
        handle_minimum_cell_position.resize(number_virtual_neurons + number_neurons);

        void* maximum_cell_position_ptr = execute_and_copy<void*>([=] __device__(Octree* octree) { return (void*)&octree->maximum_cell_position; }, octree_dev_ptr);
        handle_maximum_cell_position = gpu::Vector::CudaArrayDeviceHandle<double3>(maximum_cell_position_ptr);
        handle_maximum_cell_position.resize(number_virtual_neurons + number_neurons);

        void* position_excitatory_element_ptr = execute_and_copy<void*>([=] __device__(Octree* octree) { return (void*)&octree->position_excitatory_element; }, octree_dev_ptr);
        handle_position_excitatory_element = gpu::Vector::CudaArrayDeviceHandle<double3>(position_excitatory_element_ptr);
        handle_position_excitatory_element.resize(number_virtual_neurons + number_neurons);

        void* position_inhibitory_element_ptr = execute_and_copy<void*>([=] __device__(Octree* octree) { return (void*)&octree->position_inhibitory_element; }, octree_dev_ptr);
        handle_position_inhibitory_element = gpu::Vector::CudaArrayDeviceHandle<double3>(position_inhibitory_element_ptr);
        handle_position_inhibitory_element.resize(number_virtual_neurons + number_neurons);

        void* num_free_elements_excitatory_ptr = execute_and_copy<void*>([=] __device__(Octree* octree) { return (void*)&octree->num_free_elements_excitatory; }, octree_dev_ptr);
        handle_num_free_elements_excitatory = gpu::Vector::CudaArrayDeviceHandle<unsigned int>(num_free_elements_excitatory_ptr);
        handle_num_free_elements_excitatory.resize(number_virtual_neurons + number_neurons);

        void* num_free_elements_inhibitory_ptr = execute_and_copy<void*>([=] __device__(Octree* octree) { return (void*)&octree->num_free_elements_inhibitory; }, octree_dev_ptr);
        handle_num_free_elements_inhibitory = gpu::Vector::CudaArrayDeviceHandle<unsigned int>(num_free_elements_inhibitory_ptr);
        handle_num_free_elements_inhibitory.resize(number_virtual_neurons + number_neurons);

        size_t* number_neurons_ptr = execute_and_copy<size_t*>([=] __device__(Octree* octree) { return &octree->number_neurons; }, octree_dev_ptr);
        handle_number_neurons = number_neurons_ptr;
        cuda_memcpy_to_device((void*)handle_number_neurons, (void*)&number_neurons, sizeof(size_t), 1);

        size_t* number_virtual_neurons_ptr = execute_and_copy<size_t*>([=] __device__(Octree* octree) { return &octree->number_virtual_neurons; }, octree_dev_ptr);
        handle_number_virtual_neurons = number_virtual_neurons_ptr;
        cuda_memcpy_to_device((void*)handle_number_virtual_neurons, (void*)&number_virtual_neurons, sizeof(size_t), 1);

        ElementType* stored_element_type_ptr = execute_and_copy<ElementType*>([=] __device__(Octree* octree) { return &octree->stored_element_type; }, octree_dev_ptr);
        handle_stored_element_type = stored_element_type_ptr;
        cuda_memcpy_to_device((void*)handle_stored_element_type, (void*)&stored_element_type, sizeof(ElementType), 1);
    }

    OctreeHandleImpl::~OctreeHandleImpl() {}

    /**
    * @brief Returns the number of virtual neuronss on the octree on the GPU
    * @return The number of virtual neurons on the tree
    */
    [[nodiscard]] RelearnGPUTypes::number_neurons_type OctreeHandleImpl::get_number_virtual_neurons() const {
        return number_virtual_neurons;
    }

    /**
    * @brief Returns the number of neurons in the octree on the GPU
    * @return The number of neurons on the tree
    */
    [[nodiscard]] RelearnGPUTypes::number_neurons_type OctreeHandleImpl::get_number_neurons() const {
        return number_neurons;
    }

    /**
    * @brief Copies the GPU data structure version of the octree, which was constructed on the CPU, to the GPU
    * @param octree_cpu_copy Struct which holds the octree data to be copied on to the GPU
    */
    void OctreeHandleImpl::copy_to_gpu(OctreeCPUCopy&& octree_cpu_copy) {
        auto convert = [](const gpu::Vec3d& vec) -> double3 {
            return make_double3(vec.x, vec.y, vec.z);
        };

        std::vector<double3> pos_gpu(octree_cpu_copy.minimum_cell_position.size());

        handle_neuron_ids.copy_to_device(octree_cpu_copy.neuron_ids);

        handle_child_indices.copy_to_device(octree_cpu_copy.child_indices);

        handle_num_children.copy_to_device(octree_cpu_copy.num_children);

        std::transform(octree_cpu_copy.minimum_cell_position.begin(), octree_cpu_copy.minimum_cell_position.end(), pos_gpu.begin(), convert);
        handle_minimum_cell_position.copy_to_device(pos_gpu);
        std::transform(octree_cpu_copy.maximum_cell_position.begin(), octree_cpu_copy.maximum_cell_position.end(), pos_gpu.begin(), convert);
        handle_maximum_cell_position.copy_to_device(pos_gpu);
    
        std::transform(octree_cpu_copy.position_excitatory_element.begin(), octree_cpu_copy.position_excitatory_element.end(), pos_gpu.begin(), convert);
        handle_position_excitatory_element.copy_to_device(pos_gpu);
        std::transform(octree_cpu_copy.position_inhibitory_element.begin(), octree_cpu_copy.position_inhibitory_element.end(), pos_gpu.begin(), convert);
        handle_position_inhibitory_element.copy_to_device(pos_gpu);

        handle_num_free_elements_excitatory.copy_to_device(octree_cpu_copy.num_free_elements_excitatory);
        handle_num_free_elements_inhibitory.copy_to_device(octree_cpu_copy.num_free_elements_inhibitory);

        number_neurons = octree_cpu_copy.neuron_ids.size();
        cuda_memcpy_to_device((void*)handle_number_neurons, (void*)&number_neurons, sizeof(size_t), 1);

        number_virtual_neurons = octree_cpu_copy.num_children.size();
        cuda_memcpy_to_device((void*)handle_number_virtual_neurons, (void*)&number_virtual_neurons, sizeof(size_t), 1);
    }

    /**
    * @brief Copies the GPU data structure version of the octree to the CPU
    * @param octree_cpu_copy Struct which holds the octree data to be copied on to the CPU
    */
    void OctreeHandleImpl::copy_to_cpu(OctreeCPUCopy& octree_cpu_copy) {
        auto convert = [](const double3& vec) -> gpu::Vec3d {
            return gpu::Vec3d(vec.x, vec.y, vec.z);
        };

        std::vector<double3> pos_gpu(octree_cpu_copy.minimum_cell_position.size());

        handle_neuron_ids.copy_to_host(octree_cpu_copy.neuron_ids);

        handle_child_indices.copy_to_host(octree_cpu_copy.child_indices);

        handle_num_children.copy_to_host(octree_cpu_copy.num_children);

        handle_minimum_cell_position.copy_to_host(pos_gpu);
        std::transform(pos_gpu.begin(), pos_gpu.end(), octree_cpu_copy.minimum_cell_position.begin(), convert);
        handle_maximum_cell_position.copy_to_host(pos_gpu);
        std::transform(pos_gpu.begin(), pos_gpu.end(), octree_cpu_copy.maximum_cell_position.begin(), convert);

        handle_position_excitatory_element.copy_to_host(pos_gpu);
        std::transform(pos_gpu.begin(), pos_gpu.end(), octree_cpu_copy.position_excitatory_element.begin(), convert);
        handle_position_inhibitory_element.copy_to_host(pos_gpu);
        std::transform(pos_gpu.begin(), pos_gpu.end(), octree_cpu_copy.position_inhibitory_element.begin(), convert);

        handle_num_free_elements_excitatory.copy_to_host(octree_cpu_copy.num_free_elements_excitatory);
        handle_num_free_elements_inhibitory.copy_to_host(octree_cpu_copy.num_free_elements_inhibitory);
    }

    [[nodiscard]] void* OctreeHandleImpl::get_device_pointer() {
        return octree_dev_ptr;
    }

    void OctreeHandleImpl::update_tree() {
        update_tree_kernel<<<1,1>>>(octree_dev_ptr);
    }

    void OctreeHandleImpl::update_leaf_nodes(std::vector<gpu::Vec3d> position_excitatory_element,
                               std::vector<gpu::Vec3d> position_inhibitory_element,
                               std::vector<unsigned int> num_free_elements_excitatory,
                               std::vector<unsigned int> num_free_elements_inhibitory) {
        
        auto convert = [](const gpu::Vec3d& vec) -> double3 {
            return make_double3(vec.x, vec.y, vec.z);
        };

        std::vector<double3> pos_gpu(position_excitatory_element.size());

        std::transform(position_excitatory_element.begin(), position_excitatory_element.end(), pos_gpu.begin(), convert);
        handle_position_excitatory_element.copy_to_device_at(pos_gpu, 0);
        std::transform(position_inhibitory_element.begin(), position_inhibitory_element.end(), pos_gpu.begin(), convert);
        handle_position_inhibitory_element.copy_to_device_at(pos_gpu, 0);

        handle_num_free_elements_excitatory.copy_to_device_at(num_free_elements_excitatory, 0);
        handle_num_free_elements_inhibitory.copy_to_device_at(num_free_elements_inhibitory, 0);
    }

    [[nodiscard]] std::vector<uint64_t> OctreeHandleImpl::get_neuron_ids() {
        std::vector<uint64_t> host_neuron_ids;
        handle_neuron_ids.copy_to_host(host_neuron_ids);
        return host_neuron_ids;
    }

    /**
    * @brief Returns a shared pointer to a newly created handle to the Octree on the GPU
    * @param number_neurons Number of neurons, influences how much memory will be allocated on the GPU
    * @param number_virtual_neurons Number of virtual neurons, influences how much memory will be allocated on the GPU
    */
    std::shared_ptr<OctreeHandle> create_octree(RelearnGPUTypes::number_neurons_type number_neurons, RelearnGPUTypes::number_neurons_type number_virtual_neurons, ElementType stored_element_type) {
        Octree* octree_dev_ptr = init_class_on_device<Octree>();

        cudaDeviceSynchronize();
        gpu_check_last_error();

        auto a = std::make_unique<OctreeHandleImpl>(octree_dev_ptr, number_neurons, number_virtual_neurons, stored_element_type);
        return std::move(a);
    }

    __global__ void update_tree_kernel(Octree* octree) {
        octree->update_tree();
    }
}