/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Octree.cuh"

#include <algorithm>
#include "CudaDouble3.cuh"

namespace gpu::algorithm {
/*converts a gpu::Vec3 to an util::Vec3*/
double3 convert_to_cpu(const gpu::Vec3d& vec) {
    return make_double3(vec.x, vec.y, vec.z);
}

OctreeHandleImpl::OctreeHandleImpl(Octree* dev_ptr, const RelearnGPUTypes::number_neurons_type number_neurons, const RelearnGPUTypes::number_neurons_type number_virtual_neurons, ElementType stored_element_type)
    : number_neurons(number_neurons)
    , number_virtual_neurons(number_virtual_neurons)
    , octree_dev_ptr(dev_ptr) {
    _init(stored_element_type);
}

void OctreeHandleImpl::_init(ElementType stored_element_type) {
    void* neuron_ids_ptr = execute_and_copy<void*>([=] __device__(Octree * octree) { return (void*)&octree->neuron_ids; }, octree_dev_ptr);
    handle_neuron_ids = gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::neuron_id_type>(neuron_ids_ptr);
    handle_neuron_ids.resize(number_neurons);

    void* child_indices_ptr = execute_and_copy<void*>([=] __device__(Octree * octree) { return (void*)&octree->child_indices; }, octree_dev_ptr);
    handle_child_indices = gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::neuron_index_type>(child_indices_ptr);
    handle_child_indices.resize(number_virtual_neurons * 8);

    void* num_children_ptr = execute_and_copy<void*>([=] __device__(Octree * octree) { return (void*)&octree->num_children; }, octree_dev_ptr);
    handle_num_children = gpu::Vector::CudaArrayDeviceHandle<unsigned int>(num_children_ptr);
    handle_num_children.resize(number_virtual_neurons);

    void* minimum_cell_position_ptr = execute_and_copy<void*>([=] __device__(Octree * octree) { return (void*)&octree->minimum_cell_position; }, octree_dev_ptr);
    handle_minimum_cell_position = gpu::Vector::CudaArrayDeviceHandle<gpu::Vector::CudaDouble3>(minimum_cell_position_ptr);
    handle_minimum_cell_position.resize(number_virtual_neurons + number_neurons);

    void* maximum_cell_position_ptr = execute_and_copy<void*>([=] __device__(Octree * octree) { return (void*)&octree->maximum_cell_position; }, octree_dev_ptr);
    handle_maximum_cell_position = gpu::Vector::CudaArrayDeviceHandle<gpu::Vector::CudaDouble3>(maximum_cell_position_ptr);
    handle_maximum_cell_position.resize(number_virtual_neurons + number_neurons);

    void* position_excitatory_element_ptr = execute_and_copy<void*>([=] __device__(Octree * octree) { return (void*)&octree->position_excitatory_element; }, octree_dev_ptr);
    handle_position_excitatory_element = gpu::Vector::CudaArrayDeviceHandle<gpu::Vector::CudaDouble3>(position_excitatory_element_ptr);
    handle_position_excitatory_element.resize(number_virtual_neurons + number_neurons);

    void* position_inhibitory_element_ptr = execute_and_copy<void*>([=] __device__(Octree * octree) { return (void*)&octree->position_inhibitory_element; }, octree_dev_ptr);
    handle_position_inhibitory_element = gpu::Vector::CudaArrayDeviceHandle<gpu::Vector::CudaDouble3>(position_inhibitory_element_ptr);
    handle_position_inhibitory_element.resize(number_virtual_neurons + number_neurons);

    void* num_free_elements_excitatory_ptr = execute_and_copy<void*>([=] __device__(Octree * octree) { return (void*)&octree->num_free_elements_excitatory; }, octree_dev_ptr);
    handle_num_free_elements_excitatory = gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::number_elements_type>(num_free_elements_excitatory_ptr);
    handle_num_free_elements_excitatory.resize(number_virtual_neurons + number_neurons);

    void* num_free_elements_inhibitory_ptr = execute_and_copy<void*>([=] __device__(Octree * octree) { return (void*)&octree->num_free_elements_inhibitory; }, octree_dev_ptr);
    handle_num_free_elements_inhibitory = gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::number_elements_type>(num_free_elements_inhibitory_ptr);
    handle_num_free_elements_inhibitory.resize(number_virtual_neurons + number_neurons);

    RelearnGPUTypes::number_neurons_type* number_neurons_ptr = execute_and_copy<RelearnGPUTypes::number_neurons_type*>([=] __device__(Octree * octree) { return &octree->number_neurons; }, octree_dev_ptr);
    handle_number_neurons = number_neurons_ptr;
    cuda_memcpy_to_device((void*)handle_number_neurons, (void*)&number_neurons, sizeof(RelearnGPUTypes::number_neurons_type), 1);

    RelearnGPUTypes::number_neurons_type* number_virtual_neurons_ptr = execute_and_copy<RelearnGPUTypes::number_neurons_type*>([=] __device__(Octree * octree) { return &octree->number_virtual_neurons; }, octree_dev_ptr);
    handle_number_virtual_neurons = number_virtual_neurons_ptr;
    cuda_memcpy_to_device((void*)handle_number_virtual_neurons, (void*)&number_virtual_neurons, sizeof(RelearnGPUTypes::number_neurons_type), 1);

    ElementType* stored_element_type_ptr = execute_and_copy<ElementType*>([=] __device__(Octree * octree) { return &octree->stored_element_type; }, octree_dev_ptr);
    handle_stored_element_type = stored_element_type_ptr;
    cuda_memcpy_to_device((void*)handle_stored_element_type, (void*)&stored_element_type, sizeof(ElementType), 1);
}

[[nodiscard]] RelearnGPUTypes::number_neurons_type OctreeHandleImpl::get_number_virtual_neurons() const {
    return number_virtual_neurons;
}

[[nodiscard]] RelearnGPUTypes::number_neurons_type OctreeHandleImpl::get_number_neurons() const {
    return number_neurons;
}

void OctreeHandleImpl::copy_to_device(OctreeCPUCopy&& octree_cpu_copy) {
    auto convert = [](const gpu::Vec3d& vec) -> gpu::Vector::CudaDouble3 {
        return gpu::Vector::CudaDouble3(vec.x, vec.y, vec.z);
    };

    std::vector<gpu::Vector::CudaDouble3> pos_gpu(octree_cpu_copy.minimum_cell_position.size());

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
    cuda_memcpy_to_device((void*)handle_number_neurons, (void*)&number_neurons, sizeof(RelearnGPUTypes::number_neurons_type), 1);

    number_virtual_neurons = octree_cpu_copy.num_children.size();
    cuda_memcpy_to_device((void*)handle_number_virtual_neurons, (void*)&number_virtual_neurons, sizeof(RelearnGPUTypes::number_neurons_type), 1);
}

OctreeCPUCopy OctreeHandleImpl::copy_to_host(const RelearnGPUTypes::number_neurons_type num_neurons, const RelearnGPUTypes::number_neurons_type num_virtual_neurons) {
    auto convert = [](const gpu::Vector::CudaDouble3& vec) -> gpu::Vec3d {
        return gpu::Vec3d(vec.get_x(), vec.get_y(), vec.get_z());
    };

    OctreeCPUCopy octree_cpu_copy(num_neurons, num_virtual_neurons);

    std::vector<gpu::Vector::CudaDouble3> pos_gpu(octree_cpu_copy.minimum_cell_position.size());

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

    return octree_cpu_copy;
}

[[nodiscard]] void* OctreeHandleImpl::get_device_pointer() {
    return octree_dev_ptr;
}

void OctreeHandleImpl::update_virtual_neurons() {
    const auto num_threads = get_number_threads(update_virtual_neurons_kernel, number_virtual_neurons);
    const auto num_blocks = get_number_blocks(num_threads, number_virtual_neurons);

    update_virtual_neurons_kernel<<<num_blocks, num_threads>>>(octree_dev_ptr);

    cudaDeviceSynchronize();
    gpu_check_last_error();
}

void OctreeHandleImpl::update_leaf_nodes(std::vector<gpu::Vec3d> position_excitatory_element,
    std::vector<gpu::Vec3d> position_inhibitory_element,
    std::vector<RelearnGPUTypes::number_elements_type> num_free_elements_excitatory,
    std::vector<RelearnGPUTypes::number_elements_type> num_free_elements_inhibitory) {

    auto convert = [](const gpu::Vec3d& vec) -> gpu::Vector::CudaDouble3 {
        return gpu::Vector::CudaDouble3(vec.x, vec.y, vec.z);
    };

    std::vector<gpu::Vector::CudaDouble3> pos_gpu(position_excitatory_element.size());

    std::transform(position_excitatory_element.begin(), position_excitatory_element.end(), pos_gpu.begin(), convert);
    handle_position_excitatory_element.copy_to_device_at(pos_gpu, 0);
    std::transform(position_inhibitory_element.begin(), position_inhibitory_element.end(), pos_gpu.begin(), convert);
    handle_position_inhibitory_element.copy_to_device_at(pos_gpu, 0);

    handle_num_free_elements_excitatory.copy_to_device_at(num_free_elements_excitatory, 0);
    handle_num_free_elements_inhibitory.copy_to_device_at(num_free_elements_inhibitory, 0);
}

[[nodiscard]] std::vector<RelearnGPUTypes::neuron_id_type> OctreeHandleImpl::get_neuron_ids() {
    std::vector<RelearnGPUTypes::neuron_id_type> host_neuron_ids;
    handle_neuron_ids.copy_to_host(host_neuron_ids);
    return host_neuron_ids;
}

[[nodiscard]] RelearnGPUTypes::number_elements_type OctreeHandleImpl::get_total_excitatory_elements() {
    RelearnGPUTypes::number_elements_type total_ex_elements;
    handle_num_free_elements_excitatory.copy_to_host_from_to(&total_ex_elements, number_neurons + number_virtual_neurons - 1, number_neurons + number_virtual_neurons);
    return total_ex_elements;
}

[[nodiscard]] RelearnGPUTypes::number_elements_type OctreeHandleImpl::get_total_inhibitory_elements() {
    RelearnGPUTypes::number_elements_type total_in_elements;
    handle_num_free_elements_inhibitory.copy_to_host_from_to(&total_in_elements, number_neurons + number_virtual_neurons - 1, number_neurons + number_virtual_neurons);
    return total_in_elements;
}

[[nodiscard]] gpu::Vec3d OctreeHandleImpl::get_node_position(RelearnGPUTypes::neuron_index_type node_index, SignalType signal_type) {
    RelearnGPUException::check(node_index < number_neurons + number_virtual_neurons,
        "gpu::algorithm::OctreeHandleImpl::get_node_position: node_index was invalid");

    gpu::Vector::CudaDouble3 pos;
    if (signal_type == SignalType::Excitatory) {
        handle_position_excitatory_element.copy_to_host_from_to(&pos, node_index, node_index + 1);
    } else {
        handle_position_inhibitory_element.copy_to_host_from_to(&pos, node_index, node_index + 1);
    }

    return gpu::Vec3d(pos.get_x(), pos.get_y(), pos.get_z());
}

[[nodiscard]] std::pair<gpu::Vec3d, gpu::Vec3d> OctreeHandleImpl::get_bounding_box(RelearnGPUTypes::neuron_index_type node_index) {
    gpu::Vector::CudaDouble3 min;
    gpu::Vector::CudaDouble3 max;
    handle_minimum_cell_position.copy_to_host_from_to(&min, node_index, node_index + 1);
    handle_maximum_cell_position.copy_to_host_from_to(&max, node_index, node_index + 1);

    return std::make_pair(gpu::Vec3d(min.get_x(), min.get_y(), min.get_z()), gpu::Vec3d(max.get_x(), max.get_y(), max.get_z()));
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

    auto a = std::make_shared<OctreeHandleImpl>(octree_dev_ptr, number_neurons, number_virtual_neurons, stored_element_type);
    return std::move(a);
}

__global__ void update_virtual_neurons_kernel(Octree* octree) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x
        + threadIdx.y * gridDim.x * blockDim.x
        + threadIdx.z * gridDim.x * blockDim.x * blockDim.y
        + blockIdx.y * blockDim.x * blockDim.y * gridDim.x
        + blockIdx.z * blockDim.x * blockDim.y * gridDim.x * gridDim.y;

    uint64_t number_neurons = octree->number_neurons;
    uint64_t number_virtual_neurons = octree->number_virtual_neurons;

    int num_threads = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;

    for (int i = thread_id; i < number_virtual_neurons - 1; i += num_threads) {
        octree->num_free_elements_excitatory[i + number_neurons] = -1;
    }

    __threadfence();

    octree->update_virtual_neurons();
}
}