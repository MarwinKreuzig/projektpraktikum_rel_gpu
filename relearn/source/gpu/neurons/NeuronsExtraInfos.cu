#include "NeuronsExtraInfos.cuh"

#include "../Commons.cuh"
#include <iostream>

namespace gpu::neurons {
    NeuronsExtraInfosHandleImpl::NeuronsExtraInfosHandleImpl(void* _dev_ptr)
            : device_ptr(_dev_ptr) {
        _init();
    }

    void NeuronsExtraInfosHandleImpl::_init() {
        void* disable_flags_ptr = execute_and_copy<void*>([=] __device__(NeuronsExtraInfos * extra_infos) { return (void*)&extra_infos->disable_flags; }, (neurons::NeuronsExtraInfos*)device_ptr);
        handle_disable_flags = gpu::Vector::CudaArrayDeviceHandle<UpdateStatus>(disable_flags_ptr);
    }

    void* NeuronsExtraInfosHandleImpl::get_device_pointer() {
        return device_ptr;
    }

    void NeuronsExtraInfosHandleImpl::disable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) {
        const auto num_disabled_neurons = neuron_ids.size();
        if (num_disabled_neurons == 0) {
            return;
        }
        handle_disable_flags.set(neuron_ids.data(), num_disabled_neurons, UpdateStatus::Disabled);
    }

    void NeuronsExtraInfosHandleImpl::enable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) {
        const auto num_enabled_neurons = neuron_ids.size();
        if (num_enabled_neurons == 0) {
            return;
        }
        handle_disable_flags.set(neuron_ids.data(), num_enabled_neurons, UpdateStatus::Enabled);
    }

    void NeuronsExtraInfosHandleImpl::init(const RelearnGPUTypes::number_neurons_type _num_neurons) {
        handle_disable_flags.resize(_num_neurons, UpdateStatus::Enabled);
        num_neurons = _num_neurons;
        set_num_neurons(_num_neurons);
    }

    void NeuronsExtraInfosHandleImpl::set_num_neurons(size_t _num_neurons) {
        num_neurons = _num_neurons;
        void* ptr = execute_and_copy<void*>([=] __device__(NeuronsExtraInfos * extra_infos) { return (void*)&extra_infos->number_local_neurons_device; }, (neurons::NeuronsExtraInfos*)device_ptr);
        cuda_memcpy_to_device(ptr, &num_neurons, sizeof(size_t), 1);
    }

    void NeuronsExtraInfosHandleImpl::create_neurons(size_t creation_count) {
        const auto old_size = num_neurons;
        const auto new_size = old_size + creation_count;
        num_neurons = new_size;
        handle_disable_flags.resize(new_size, UpdateStatus::Enabled);
        set_num_neurons(num_neurons);
    }

    std::unique_ptr<NeuronsExtraInfosHandle> create() {
        void* extra_infos_dev_ptr = init_class_on_device<NeuronsExtraInfos>();

        cudaDeviceSynchronize();
        gpu_check_last_error();

        auto a = std::make_unique<NeuronsExtraInfosHandleImpl>(extra_infos_dev_ptr);
        return std::move(a);
    }

};