#pragma once

#include "Commons.cuh"
#include "utils/GpuTypes.h"
#include "utils/Interface.h"
#include "../../shared/enums/ElementType.h"
#include "../../shared/enums/SignalType.h"

#include "CudaArray.cuh"
#include "CudaDouble3.cuh"

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

        gpu::Vector::CudaArray<RelearnGPUTypes::neuron_id_type> neuron_ids;

        gpu::Vector::CudaArray<RelearnGPUTypes::neuron_index_type> child_indices;

        // we need this, since we can't use -1 to indicate an invalid child_indices entry
        gpu::Vector::CudaArray<unsigned int> num_children;

        gpu::Vector::CudaArray<gpu::Vector::CudaDouble3> minimum_cell_position;
        gpu::Vector::CudaArray<gpu::Vector::CudaDouble3> maximum_cell_position;

        gpu::Vector::CudaArray<gpu::Vector::CudaDouble3> position_excitatory_element;
        gpu::Vector::CudaArray<gpu::Vector::CudaDouble3> position_inhibitory_element;

        // depending on if barnes hut or inverse barnes hut will be done, these are either dendrites or axons
        gpu::Vector::CudaArray<RelearnGPUTypes::number_elements_type> num_free_elements_excitatory;
        gpu::Vector::CudaArray<RelearnGPUTypes::number_elements_type> num_free_elements_inhibitory;

        RelearnGPUTypes::number_neurons_type number_neurons;
        RelearnGPUTypes::number_neurons_type number_virtual_neurons;

        ElementType stored_element_type;

        /**
        * @brief Returns the number of free elements for the given node for the given signal type 
        * @param signal_type The signal type
        * @param node_index The index to the node
        * @return The number of free elements for the given node for the given signal type 
        */
        __device__ int get_num_free_elements_for_signal(SignalType signal_type, RelearnGPUTypes::neuron_index_type node_index) {
            if (signal_type == SignalType::Excitatory) {
                // Since this number can go below 0 in the synapse creation process, we need to give back the max of 0 here
                return max(0, num_free_elements_excitatory[node_index]);
            } else {
                return max(0, num_free_elements_inhibitory[node_index]);
            }
        }

        /**
        * @brief Returns the position for the given node for the given signal type 
        * @param signal_type The signal type
        * @param node_index The index to the node
        * @return The position for the given node for the given signal type 
        */
        __device__ gpu::Vector::CudaDouble3 get_position_for_signal(SignalType signal_type, RelearnGPUTypes::neuron_index_type node_index) {
            if (signal_type == SignalType::Excitatory) {
                return position_excitatory_element[node_index];
            } else {
                return position_inhibitory_element[node_index];
            }
        }

        /**
        * @brief Returns the maximal dimensional difference of the cell in which the node is located
        * @param node_index The index to the node in the cell to get the maximal dimensional difference from
        * @return The maximal dimensional difference of the cell in which the node is located
        */
        __device__ double get_maximal_dimension_difference(RelearnGPUTypes::neuron_index_type node_index) {
            gpu::Vector::CudaDouble3 diff_vector = maximum_cell_position[node_index] - minimum_cell_position[node_index];

            return diff_vector.max();
        }

        /**
        * @brief Updates the octree bottom up after the leaf nodes had their number of elements updated
        */
        __device__ void update_virtual_neurons() {

            int thread_id = threadIdx.x + blockIdx.x * blockDim.x
                + threadIdx.y * gridDim.x * blockDim.x
                + threadIdx.z * gridDim.x * blockDim.x * blockDim.y
                + blockIdx.y * blockDim.x * blockDim.y * gridDim.x
                + blockIdx.z * blockDim.x * blockDim.y * gridDim.x * gridDim.y;

            int num_threads = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;


            int i = thread_id;

            int missing_children[8];
            int missing = 0;

            //Position of elements
            gpu::Vector::CudaDouble3 position_excitatory(0., 0., 0.);
            gpu::Vector::CudaDouble3 position_inhibitory(0., 0., 0.);

            //Number of free elements
            unsigned int free_elements_excitatory = 0;
            unsigned int free_elements_inhibitory = 0;


            while (i < number_virtual_neurons)    {

                bool changed = false;

                if (missing == 0)   {

                    for (int j = 0; j < num_children[i]; j++)    {
                        auto child = child_indices[i + j * number_virtual_neurons];

                        if (num_free_elements_excitatory[child] >= 0)   {
                            free_elements_excitatory += num_free_elements_excitatory[child];
                            free_elements_inhibitory += num_free_elements_inhibitory[child];
                            position_excitatory = position_excitatory + (position_excitatory_element[child] * num_free_elements_excitatory[child]);
                            position_inhibitory = position_inhibitory + (position_inhibitory_element[child] * num_free_elements_inhibitory[child]);
                        }else {
                            missing_children[missing] = child;
                            missing++;
                        }
                    }
                }


                if (missing != 0)   {
                    do {
                        auto child = missing_children[missing - 1];
                        if (num_free_elements_excitatory[missing_children[missing - 1]] >= 0)   {

                            free_elements_excitatory += num_free_elements_excitatory[child];
                            free_elements_inhibitory += num_free_elements_inhibitory[child];
                            position_excitatory = position_excitatory + (position_excitatory_element[child] * num_free_elements_excitatory[child]);
                            position_inhibitory = position_inhibitory + (position_inhibitory_element[child] * num_free_elements_inhibitory[child]);

                            missing--;
                            changed = true;
                        } else {
                            changed = false;
                        }
                    } while (changed && (missing != 0));
                }
                __threadfence();

                if (missing == 0)   {
                    if (0 == free_elements_excitatory) {
                        gpu::Vector::CudaDouble3 tmp(0., 0., 0.);
                        position_excitatory_element[i + number_neurons] = tmp;
                    } else {
                        const auto scaled_position = position_excitatory / free_elements_excitatory;
                        position_excitatory_element[i + number_neurons] = scaled_position;
                    }

                    if (0 == free_elements_inhibitory) {
                        gpu::Vector::CudaDouble3 tmp(0., 0., 0.);
                        position_inhibitory_element[i + number_neurons] = tmp;
                    } else {
                        const auto scaled_position = position_inhibitory / free_elements_inhibitory;
                        position_inhibitory_element[i + number_neurons] = scaled_position;
                    }

                    __threadfence();

                    num_free_elements_inhibitory[i + number_neurons] = free_elements_inhibitory;
                    num_free_elements_excitatory[i + number_neurons] = free_elements_excitatory;

                    i += num_threads;

                    position_excitatory = gpu::Vector::CudaDouble3(0., 0., 0.);
                    position_inhibitory = gpu::Vector::CudaDouble3(0., 0., 0.);

                    //Number of free elements
                    free_elements_excitatory = 0;
                    free_elements_inhibitory = 0;
                }
                __threadfence();
            }
        }
    };

    class OctreeHandleImpl : public OctreeHandle {
    public:

        /**
         * @brief Allocates the necessary memory for the octree on the GPU and saves the device pointers to this memory
         * @param dev_ptr A pointer to an allocated Octree instance. Its members do not need to be allocated.
         * @param number_neurons Number of neurons, determines how much memory will be allocated on the GPU
         * @param number_virtual_neurons Number of virtual neurons, determines how much memory will be allocated on the GPU
         * @param stored_element_type Type of elements that will be stored (Axon or Dendrite)
         */
        OctreeHandleImpl(Octree* dev_ptr, 
                         const RelearnGPUTypes::number_neurons_type number_neurons, 
                         const RelearnGPUTypes::number_neurons_type number_virtual_neurons,
                         ElementType stored_element_type);

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
         * @param number_neurons The number of leaf nodes
         * @param number_virtual_neurons The number of virtual neurons
         */
        OctreeCPUCopy copy_to_host(
            const RelearnGPUTypes::number_neurons_type num_neurons,
            const RelearnGPUTypes::number_neurons_type num_virtual_neurons) override;

        /**
         * @brief Calls the kernel that updates the octree
         */
        void update_virtual_neurons() override;

        /**
         * @brief Updates the octree leaf nodes
         */
        void update_leaf_nodes(std::vector<gpu::Vec3d> position_excitatory_element,
                               std::vector<gpu::Vec3d> position_inhibitory_element,
                               std::vector<RelearnGPUTypes::number_elements_type> num_free_elements_excitatory,
                               std::vector<RelearnGPUTypes::number_elements_type> num_free_elements_inhibitory) override;

        /**
         * @brief Getter for octree_dev_ptr
         * @return octree_dev_ptr
         */
        [[nodiscard]] void* get_device_pointer() override;

        /**
         * @brief Getter for Neuron IDs
         * @return Neuron IDs
         */
        [[nodiscard]] std::vector<RelearnGPUTypes::neuron_id_type> get_neuron_ids() override;

        /**
         * @brief Returns the total excitatory elements in the tree through the root node
         * @return The total excitatory elements in the tree
         */
        [[nodiscard]] RelearnGPUTypes::number_elements_type get_total_excitatory_elements() override; 

        /**
         * @brief Returns the total inhibitory elements in the tree through the root node
         * @return The total inhibitory elements in the tree
         */
        [[nodiscard]] RelearnGPUTypes::number_elements_type get_total_inhibitory_elements() override;

        /**
         * @brief Returns the position of a node for a given signal type
         * @param node_index The index of the node
         * @param signal_type The signal type
         * @return The position of the node
         */
        [[nodiscard]] gpu::Vec3d get_node_position(RelearnGPUTypes::neuron_index_type node_index, SignalType signal_type) override;

        /**
         * @brief Returns the bounding box of the given cell of the node index given
         * @param node_index The index of the node
         * @return The bounding box of the given cell
         */
        [[nodiscard]] std::pair<gpu::Vec3d, gpu::Vec3d> get_bounding_box(RelearnGPUTypes::neuron_index_type node_index) override;

    private:
        Octree* octree_dev_ptr;

        gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::neuron_id_type> handle_neuron_ids;

        gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::neuron_index_type> handle_child_indices;

        gpu::Vector::CudaArrayDeviceHandle<unsigned int> handle_num_children;

        gpu::Vector::CudaArrayDeviceHandle<gpu::Vector::CudaDouble3> handle_minimum_cell_position;
        gpu::Vector::CudaArrayDeviceHandle<gpu::Vector::CudaDouble3> handle_maximum_cell_position;

        gpu::Vector::CudaArrayDeviceHandle<gpu::Vector::CudaDouble3> handle_position_excitatory_element;
        gpu::Vector::CudaArrayDeviceHandle<gpu::Vector::CudaDouble3> handle_position_inhibitory_element;

        gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::number_elements_type> handle_num_free_elements_excitatory;
        gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::number_elements_type> handle_num_free_elements_inhibitory;

        ElementType* handle_stored_element_type;

        RelearnGPUTypes::number_neurons_type number_neurons;
        RelearnGPUTypes::number_neurons_type* handle_number_neurons;
        RelearnGPUTypes::number_neurons_type number_virtual_neurons;
        RelearnGPUTypes::number_neurons_type* handle_number_virtual_neurons;
    };

    __global__ void update_virtual_neurons_kernel(Octree* octree);
};
