#include "NeuronsExtraInfos.cuh"

#include "../Commons.cuh"
#include <cuda.h>

namespace gpu::neurons {

NeuronsExtraInfosHandleImpl::NeuronsExtraInfosHandleImpl(NeuronsExtraInfos* _dev_ptr)
    : device_ptr(_dev_ptr) {
    _init();
}

void NeuronsExtraInfosHandleImpl::_init() {
    void* disable_flags_ptr = execute_and_copy<void*>([=] __device__(NeuronsExtraInfos * extra_infos) { return (void*)&extra_infos->disable_flags; }, /*(neurons::NeuronsExtraInfos*)*/device_ptr);
    handle_disable_flags = gpu::Vector::CudaArrayDeviceHandle<UpdateStatus>(disable_flags_ptr);

    void* positions_ptr = execute_and_copy<void*>([=] __device__(NeuronsExtraInfos * extra_infos) { return (void*)&extra_infos->positions; }, /*(neurons::NeuronsExtraInfos*)*/device_ptr);
    handle_positions = gpu::Vector::CudaArrayDeviceHandle<double3>(positions_ptr);

    RelearnGPUTypes::number_neurons_type* num_neurons_ptr = execute_and_copy<RelearnGPUTypes::number_neurons_type*>([=] __device__(NeuronsExtraInfos* extra_infos) { return &extra_infos->num_neurons; }, /*(neurons::NeuronsExtraInfos*)*/device_ptr);
    handle_num_neurons = num_neurons_ptr;
}

void* NeuronsExtraInfosHandleImpl::get_device_pointer() {
    return device_ptr;
}

// Generally, functionality like this can be done on the cpu here, but somtimes we will need in on a kernel on the GPU
// In this case, outsource it as a device function into the above struct and call it indirictly over a global function here, if it is also needed on the GPU
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
    set_num_neurons(_num_neurons);
}

// currently only updates disable_flags, size and positions. Add more parameters as needed
void NeuronsExtraInfosHandleImpl::create_neurons(RelearnGPUTypes::number_neurons_type new_size, const std::vector<gpu::Vec3d>& positions) {
    set_num_neurons(new_size);
    handle_disable_flags.resize(new_size, UpdateStatus::Enabled);

    set_positions(positions);
}

void NeuronsExtraInfosHandleImpl::set_positions(const std::vector<gpu::Vec3d>& pos) {
    auto convert = [](const gpu::Vec3d& vec) -> double3 {
        return make_double3(vec.x, vec.y, vec.z);
    };

    std::vector<double3> pos_gpu(pos.size());
    std::transform(pos.begin(), pos.end(), pos_gpu.begin(), convert);
        
    handle_positions.copy_to_device(pos_gpu);
}

void NeuronsExtraInfosHandleImpl::set_num_neurons(RelearnGPUTypes::number_neurons_type _num_neurons) {
    cuda_memcpy_to_device((void*)handle_num_neurons, (void*)&_num_neurons, sizeof(RelearnGPUTypes::number_neurons_type), 1);
}

std::unique_ptr<NeuronsExtraInfosHandle> create() {
    NeuronsExtraInfos* extra_infos_dev_ptr = init_class_on_device<NeuronsExtraInfos>();

    auto a = std::make_unique<NeuronsExtraInfosHandleImpl>(extra_infos_dev_ptr);
    return std::move(a);
}

};