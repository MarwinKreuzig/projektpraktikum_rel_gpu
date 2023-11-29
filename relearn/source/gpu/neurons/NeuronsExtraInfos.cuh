#pragma once

#include "../Commons.cuh"
#include "enums/UpdateStatus.h"
#include "gpu/GpuTypes.h"
#include "gpu/Interface.h"
#include "../structure/CudaArray.cuh"
#include "../structure/CudaVector.cuh"

#include <iostream>
#include <cuda.h>

namespace gpu::neurons {

class NeuronsExtraInfos {
    /**
     * Class representing NeuronsExtraInfos on the utils. Contains the disable flags and number of local neurons
     */

public:
    size_t number_local_neurons_device = 0;

    gpu::Vector::CudaArray<UpdateStatus> disable_flags;

public:
    /**
     * @return Return the number of local neurons
     */
    inline __device__ size_t get_number_local_neurons() {
        return number_local_neurons_device;
    }
};

class NeuronsExtraInfosHandleImpl : public NeuronsExtraInfosHandle {
    /**
     * Implementation of the handle for the cpu that controls the utils object
     */
public:
    NeuronsExtraInfosHandleImpl(void* _dev_ptr)
        : device_ptr(_dev_ptr) {
        _init();
    }

    void _init() {
        void* disable_flags_ptr = execute_and_copy<void*>([=] __device__(NeuronsExtraInfos * extra_infos) { return (void*)&extra_infos->disable_flags; }, (neurons::NeuronsExtraInfos*)device_ptr);
        handle_disable_flags = gpu::Vector::CudaArrayDeviceHandle<UpdateStatus>(disable_flags_ptr);
    }

    void* get_device_pointer() {
        return device_ptr;
    }

    void disable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) override {
        const auto num_disabled_neurons = neuron_ids.size();
        if (num_disabled_neurons == 0) {
            return;
        }
        handle_disable_flags.set(neuron_ids.data(), num_disabled_neurons, UpdateStatus::Disabled);
    }

    void enable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) override {
        const auto num_enabled_neurons = neuron_ids.size();
        if (num_enabled_neurons == 0) {
            return;
        }
        handle_disable_flags.set(neuron_ids.data(), num_enabled_neurons, UpdateStatus::Enabled);
    }

    void init(const RelearnGPUTypes::number_neurons_type _num_neurons) override {
        handle_disable_flags.resize(_num_neurons, UpdateStatus::Enabled);
        num_neurons = _num_neurons;
        set_num_neurons(_num_neurons);
    }

    void set_num_neurons(size_t _num_neurons) {
        num_neurons = _num_neurons;
        void* ptr = execute_and_copy<void*>([=] __device__(NeuronsExtraInfos * extra_infos) { return (void*)&extra_infos->number_local_neurons_device; }, (neurons::NeuronsExtraInfos*)device_ptr);
        cuda_memcpy_to_device(ptr, &num_neurons, sizeof(size_t), 1);
    }

    void create_neurons(size_t creation_count) {
        const auto old_size = num_neurons;
        const auto new_size = old_size + creation_count;
        num_neurons = new_size;
        handle_disable_flags.resize(new_size, UpdateStatus::Enabled);
        set_num_neurons(num_neurons);
    }

private:
    /**
     * Pointer to the NeuronsExtraInfos instance on the utils
     */
    void* device_ptr;

    size_t num_neurons;

    gpu::Vector::CudaArrayDeviceHandle<UpdateStatus> handle_disable_flags;
};

std::unique_ptr<NeuronsExtraInfosHandle> create() {
    void* extra_infos_dev_ptr = init_class_on_device<NeuronsExtraInfos>();

    cudaDeviceSynchronize();
    gpu_check_last_error();

    auto a = std::make_unique<NeuronsExtraInfosHandleImpl>(extra_infos_dev_ptr);
    return std::move(a);
}

};