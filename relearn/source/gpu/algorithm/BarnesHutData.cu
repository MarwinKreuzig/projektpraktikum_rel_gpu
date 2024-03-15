/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "../Commons.cuh"
#include "BarnesHutData.cuh"
#include "BarnesHutKernel.cuh"

namespace gpu::algorithm {

BarnesHutDataHandleImpl::BarnesHutDataHandleImpl(BarnesHutData* _dev_ptr, unsigned int synapse_space, RelearnGPUTypes::number_neurons_type _neurons_per_thread, uint64_t _num_nodes_gathered_before_pick)
    : device_ptr(_dev_ptr)
    , neurons_per_thread(_neurons_per_thread)
    , num_nodes_gathered_before_pick(_num_nodes_gathered_before_pick) {
    _init(synapse_space);

    RelearnGPUException::check(num_nodes_gathered_before_pick >= 2, "BarnesHutDataHandleImpl::BarnesHutDataHandleImpl: num_nodes_gathered_before_pick was smaller than 2");
}

void BarnesHutDataHandleImpl::_init(unsigned int synapse_space) {
    void* source_ids_ptr = execute_and_copy<void*>([=] __device__(BarnesHutData * barnes_hut_data) { return (void*)&barnes_hut_data->source_ids; }, device_ptr);
    handle_source_ids = gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::neuron_id_type>(source_ids_ptr);
    handle_source_ids.resize(synapse_space);

    void* target_ids_ptr = execute_and_copy<void*>([=] __device__(BarnesHutData * barnes_hut_data) { return (void*)&barnes_hut_data->target_ids; }, device_ptr);
    handle_target_ids = gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::neuron_id_type>(target_ids_ptr);
    handle_target_ids.resize(synapse_space);

    void* weights_ptr = execute_and_copy<void*>([=] __device__(BarnesHutData * barnes_hut_data) { return (void*)&barnes_hut_data->weights; }, device_ptr);
    handle_weights = gpu::Vector::CudaArrayDeviceHandle<int>(weights_ptr);
    handle_weights.resize(synapse_space);

    unsigned int* number_synapses_ptr = execute_and_copy<unsigned int*>([=] __device__(BarnesHutData * barnes_hut_data) { return &barnes_hut_data->number_synapses; }, device_ptr);
    handle_number_synapses = number_synapses_ptr;

    RelearnGPUTypes::number_neurons_type* neurons_per_thread_ptr = execute_and_copy<RelearnGPUTypes::number_neurons_type*>([=] __device__(BarnesHutData * barnes_hut_data) { return &barnes_hut_data->neurons_per_thread; }, device_ptr);
    handle_neurons_per_thread = neurons_per_thread_ptr;
    cuda_memcpy_to_device((void*)handle_neurons_per_thread, (void*)&neurons_per_thread, sizeof(RelearnGPUTypes::number_neurons_type), 1);

    uint64_t* num_nodes_gathered_before_pick_ptr = execute_and_copy<uint64_t*>([=] __device__(BarnesHutData * barnes_hut_data) { return &barnes_hut_data->num_nodes_gathered_before_pick; }, device_ptr);
    handle_num_nodes_gathered_before_pick = num_nodes_gathered_before_pick_ptr;
    cuda_memcpy_to_device((void*)handle_num_nodes_gathered_before_pick, (void*)&num_nodes_gathered_before_pick, sizeof(uint64_t), 1);

    double* acceptance_criterion_ptr = execute_and_copy<double*>([=] __device__(BarnesHutData * barnes_hut_data) { return &barnes_hut_data->acceptance_criterion; }, device_ptr);
    handle_acceptance_criterion = acceptance_criterion_ptr;

    void* probability_intervals_ptr = execute_and_copy<void*>([=] __device__(BarnesHutData * barnes_hut_data) { return (void*)&barnes_hut_data->probability_intervals; }, device_ptr);
    handle_probability_intervals = gpu::Vector::CudaArrayDeviceHandle<double>(probability_intervals_ptr);

    void* nodes_to_consider_ptr = execute_and_copy<void*>([=] __device__(BarnesHutData * barnes_hut_data) { return (void*)&barnes_hut_data->nodes_to_consider; }, device_ptr);
    handle_nodes_to_consider = gpu::Vector::CudaArrayDeviceHandle<RelearnGPUTypes::neuron_index_type>(nodes_to_consider_ptr);

    void* coalesced_prefix_traversal_stack_ptr = execute_and_copy<void*>([=] __device__(BarnesHutData * barnes_hut_data) { return (void*)&barnes_hut_data->coalesced_prefix_traversal_stack; }, device_ptr);
    handle_coalesced_prefix_traversal_stack = gpu::Vector::CudaCoalescedStackDeviceHandle<RelearnGPUTypes::neuron_index_type>(coalesced_prefix_traversal_stack_ptr);
}

[[nodiscard]] void* BarnesHutDataHandleImpl::get_device_pointer() {
    return (void*)device_ptr;
}

[[nodiscard]] unsigned int BarnesHutDataHandleImpl::get_number_synapses() {
    unsigned int number_synapses;
    cuda_memcpy_to_host((void*)handle_number_synapses, (void*)(&number_synapses), sizeof(unsigned int), 1);
    return number_synapses;
}

[[nodiscard]] RelearnGPUTypes::number_neurons_type BarnesHutDataHandleImpl::get_neurons_per_thread() {
    return neurons_per_thread;
}

[[nodiscard]] std::vector<RelearnGPUTypes::neuron_id_type> BarnesHutDataHandleImpl::get_source_ids() {
    std::vector<RelearnGPUTypes::neuron_id_type> source_ids;
    handle_source_ids.copy_to_host(source_ids);
    return source_ids;
}

[[nodiscard]] std::vector<RelearnGPUTypes::neuron_id_type> BarnesHutDataHandleImpl::get_target_ids() {
    std::vector<RelearnGPUTypes::neuron_id_type> target_ids;
    handle_target_ids.copy_to_host(target_ids);
    return target_ids;
}

[[nodiscard]] std::vector<int> BarnesHutDataHandleImpl::get_weights() {
    std::vector<int> weights;
    handle_weights.copy_to_host(weights);
    return weights;
}

[[nodiscard]] unsigned int BarnesHutDataHandleImpl::get_block_size() {
    return block_size;
}

[[nodiscard]] unsigned int BarnesHutDataHandleImpl::get_grid_size() {
    return grid_size;
}

void BarnesHutDataHandleImpl::update_synapse_allocation(unsigned int synapse_space) {
    handle_source_ids.resize(synapse_space);
    handle_target_ids.resize(synapse_space);
    handle_weights.resize(synapse_space);
}

void BarnesHutDataHandleImpl::update_kernel_allocation_sizes(RelearnGPUTypes::number_neurons_type number_neurons, uint16_t max_level) {
    auto [_grid_size, _block_size] = get_number_blocks_and_threads(gpu::kernel::update_connectivity_kernel, (uint64_t)std::ceil((double)number_neurons / (double)neurons_per_thread));

    block_size = _block_size;
    grid_size = _grid_size;

    handle_probability_intervals.resize(block_size * grid_size * num_nodes_gathered_before_pick);
    handle_nodes_to_consider.resize(block_size * grid_size * num_nodes_gathered_before_pick);
    handle_coalesced_prefix_traversal_stack.resize(8 * max_level, block_size * grid_size);

    gpu::random::RandomHolder::get_instance().init_allocation(gpu::random::RandomKeyHolder::BARNES_HUT, block_size, grid_size);
}

void BarnesHutDataHandleImpl::set_acceptance_criterion(double acceptance_criterion) {
    cuda_memcpy_to_device((void*)handle_acceptance_criterion, (void*)&acceptance_criterion, sizeof(double), 1);
}

double BarnesHutDataHandleImpl::get_acceptance_criterion() {
    double acceptance_criterion;
    cuda_memcpy_to_host((void*)handle_acceptance_criterion, (void*)&acceptance_criterion, sizeof(double), 1);
    return acceptance_criterion;
}

std::unique_ptr<BarnesHutDataHandle> create_barnes_hut_data(unsigned int synapse_space, uint64_t num_nodes_gathered_before_pick, RelearnGPUTypes::number_neurons_type neurons_per_thread) {
    BarnesHutData* barnes_hut_data_dev_ptr = init_class_on_device<BarnesHutData>();

    cudaDeviceSynchronize();
    gpu_check_last_error();

    auto a = std::make_unique<BarnesHutDataHandleImpl>(barnes_hut_data_dev_ptr, synapse_space, neurons_per_thread, num_nodes_gathered_before_pick);
    return std::move(a);
}
};