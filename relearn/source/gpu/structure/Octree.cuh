#pragma once

#include "Commons.cuh"
#include "utils/GpuTypes.h"
#include "utils/Interface.h"
#include "../../shared/enums/ElementType.h"

#include "CudaArray.cuh"
#include "CudaVector.cuh"

namespace gpu::algorithm {
    struct Octree {
        /**
        * Represents a GPU Octree data structure. Indexes are always in the combined range with neurons and virtual neurons with neurons in the front
        * when trying to access child_indices, num_neurons has to be subtracted from the index.
        */

        gpu::Vector::CudaArray<uint64_t> neuron_ids;

        gpu::Vector::CudaArray<uint64_t> child_indices;

        // we need this, since we can't use -1 to indicate an invalid child_indices entry
        gpu::Vector::CudaArray<unsigned int> num_children;

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

        // TODO implement
        __device__ void update_tree() {

        }
    };

    class OctreeHandleImpl : public OctreeHandle {
    public:

        OctreeHandleImpl(Octree* dev_ptr, 
        const RelearnGPUTypes::number_neurons_type number_neurons, 
        const RelearnGPUTypes::number_neurons_type number_virtual_neurons,
        ElementType stored_element_type);

        ~OctreeHandleImpl();

        void _init(ElementType stored_element_type);

        [[nodiscard]] RelearnGPUTypes::number_neurons_type get_number_virtual_neurons() const override;
        [[nodiscard]] RelearnGPUTypes::number_neurons_type get_number_neurons() const override;

        void copy_to_gpu(OctreeCPUCopy&& octree_cpu_copy) override;
        void copy_to_cpu(OctreeCPUCopy& octree_cpu_copy) override;

        void update_tree() override;

        void update_leaf_nodes(std::vector<gpu::Vec3d> position_excitatory_element,
                               std::vector<gpu::Vec3d> position_inhibitory_element,
                               std::vector<unsigned int> num_free_elements_excitatory,
                               std::vector<unsigned int> num_free_elements_inhibitory) override;

        [[nodiscard]] void* get_device_pointer() override;

        // TODO once plasticity is on the GPU, this serves no purpose and can be deleted
        [[nodiscard]] std::vector<uint64_t> get_neuron_ids() override;

    private:
        Octree* octree_dev_ptr;

        gpu::Vector::CudaArrayDeviceHandle<uint64_t> handle_neuron_ids;

        gpu::Vector::CudaArrayDeviceHandle<uint64_t> handle_child_indices;

        gpu::Vector::CudaArrayDeviceHandle<unsigned int> handle_num_children;

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

    __global__ void update_tree_kernel(Octree* octree);
};
