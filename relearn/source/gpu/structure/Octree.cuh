#pragma once

#include <vector>

#include "cuda.h"
#include "utils/GpuTypes.h"
#include "utils/Interface.h"
#include "../../shared/enums/ElementType.h"

#include "CudaArray.cuh"

namespace gpu::algorithm {
struct Octree {
    /**
     * This struct represents a GPU-based Octree data structure storing attributes of neurons. It utilizes
     * gpu::Vector::CudaArray to efficiently manage these attributes on the GPU using the struct-of-arrays pattern.
     * That means that each attribute of a neuron is stored in a seperate array. A single neuron is therefore
     * defined by an index into each array.
     * Unique to this implementation, indexing combines virtual and actual neurons, placing actual neurons at the
     * beginning of the range. When accessing child indices, remember to subtract the total number of actual neurons
     * to account for this ordering.
     */
    gpu::Vector::CudaArray<uint64_t> neuron_ids;

    gpu::Vector::CudaArray<uint64_t> child_indices;

    // we need this, since we can't use -1 to indicate an invalid child_indices entry
    gpu::Vector::CudaArray<uint8_t> num_children;

    gpu::Vector::CudaArray<double3> minimum_cell_position;
    gpu::Vector::CudaArray<double3> maximum_cell_position;

    gpu::Vector::CudaArray<double3> position_excitatory_element;
    gpu::Vector::CudaArray<double3> position_inhibitory_element;

    // depending on if barnes hut or inverse barnes hut will be done, these are either dendrites or axons
    gpu::Vector::CudaArray<unsigned int> num_free_elements_excitatory;
    gpu::Vector::CudaArray<unsigned int> num_free_elements_inhibitory;

    uint64_t number_neurons;
    uint64_t number_virtual_neurons;

    ElementType stored_element_type;

    // TODO implement for Barnes Hut US
    __device__ void update_tree() {
    }
};

class OctreeHandleImpl : public OctreeHandle {
public:
    /**
     * @brief Allocates the necessary memory for the octree on the GPU and saves the device pointers to this memory
     * @param dev_ptr A pointer to an allocated Octree instance. Its members do not need to be allocated.
     * @param number_neurons Number of neurons, determines how much memory will be allocated on the GPU
     * @param number_virtual_neurons Number of virtual neurons, determines how much memory will be allocated on the GPU
     * @paramm stored_element_type Type of elements that will be stored (Axon or Dendrite)
     */
    OctreeHandleImpl(Octree* dev_ptr,
        const RelearnGPUTypes::number_neurons_type number_neurons,
        const RelearnGPUTypes::number_neurons_type number_virtual_neurons,
        ElementType stored_element_type);

    /**
     * @brief This is an empty Deconstructor for OctreeHandleImpl
     */
    ~OctreeHandleImpl();

    /**
     * @brief Allocates the necessary memory for the octree on the GPU and saves the device pointers to this memory
     * @param stored_element_type Type of elements that will be stored (Axon or Dendrite)
     */
    void _init(ElementType stored_element_type);

    /**
     * @brief Returns the number of virtual neurons in the octree on the GPU
     * @return The number of virtual neurons in the tree
     */
    [[nodiscard]] RelearnGPUTypes::number_neurons_type get_number_virtual_neurons() const override;

    /**
     * @brief Returns the number of neurons in the octree on the GPU
     * @return The number of neurons in the tree
     */
    [[nodiscard]] RelearnGPUTypes::number_neurons_type get_number_neurons() const override;

    /**
     * @brief Copies the GPU data structure version of the octree, which was constructed on the CPU, to the GPU
     * @param octree_cpu_copy Struct which holds the octree data to be copied to the GPU
     */
    void copy_to_device(OctreeCPUCopy&& octree_cpu_copy) override;

    /**
     * @brief Copies the GPU data structure version of the octree to the CPU
     * @param octree_cpu_copy Struct which holds the octree data to be copied to the CPU
     */
    OctreeCPUCopy copy_to_host(
        const RelearnGPUTypes::number_neurons_type num_neurons,
        const RelearnGPUTypes::number_neurons_type num_virtual_neurons) override;

    /**
     * @brief Calls the update_tree_kernel (WILL BE CHANGED IN BARNES HUT US)
     */
    void update_tree() override;

    /**
     * @brief Updates the octree leaf nodes (TODO WILL BE CHANGED IN BARNES HUT US)
     */
    void update_leaf_nodes(std::vector<gpu::Vec3d> position_excitatory_element,
        std::vector<gpu::Vec3d> position_inhibitory_element,
        std::vector<unsigned int> num_free_elements_excitatory,
        std::vector<unsigned int> num_free_elements_inhibitory) override;

    /**
     * @brief Getter for octree_dev_ptr
     * @return octree_dev_ptr
     */
    [[nodiscard]] void* get_device_pointer() override;

    /**
     * @brief Getter for Neuron IDs
     * @return Neuron IDs
     */
    // TODO once plasticity is on the GPU, this serves no purpose and can be deleted
    [[nodiscard]] std::vector<uint64_t> get_neuron_ids() override;

private:
    Octree* octree_dev_ptr;

    gpu::Vector::CudaArrayDeviceHandle<uint64_t> handle_neuron_ids;

    gpu::Vector::CudaArrayDeviceHandle<uint64_t> handle_child_indices;

    gpu::Vector::CudaArrayDeviceHandle<uint8_t> handle_num_children;

    gpu::Vector::CudaArrayDeviceHandle<double3> handle_minimum_cell_position;
    gpu::Vector::CudaArrayDeviceHandle<double3> handle_maximum_cell_position;

    gpu::Vector::CudaArrayDeviceHandle<double3> handle_position_excitatory_element;
    gpu::Vector::CudaArrayDeviceHandle<double3> handle_position_inhibitory_element;

    gpu::Vector::CudaArrayDeviceHandle<unsigned int> handle_num_free_elements_excitatory;
    gpu::Vector::CudaArrayDeviceHandle<unsigned int> handle_num_free_elements_inhibitory;

    ElementType* handle_stored_element_type;

    size_t number_neurons;
    size_t* handle_number_neurons;
    size_t number_virtual_neurons;
    size_t* handle_number_virtual_neurons;
};

/**
 * @brief Updates the virtual neurons of the octree (TODO WILL BE CHANGED IN BARNES HUT US)
 */
__global__ void update_tree_kernel(Octree* octree);
};
