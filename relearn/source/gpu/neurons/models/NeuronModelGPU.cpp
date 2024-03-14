#include "NeuronModelGPU.h"

namespace gpu {
  // This looks sketchy, but the CPU neuron models don't do anything either in these functions
void NeuronModelGPU::enable_neurons(const std::span<const NeuronID> neuron_ids) {}

void NeuronModelGPU::disable_neurons(const std::span<const NeuronID> neuron_ids) {}

void NeuronModelGPU::set_fired(const NeuronID neuron_id, const FiredStatus new_value) {
    gpu_handle->set_fired(neuron_id.get_neuron_id(), new_value);
}

bool NeuronModelGPU::get_fired(const NeuronID neuron_id) const {
    return gpu_handle->get_fired(neuron_id.get_neuron_id());
}
std::span<const FiredStatus> NeuronModelGPU::get_fired() const noexcept {
    return gpu_handle->get_fired();
}

void NeuronModelGPU::set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {  
    gpu_handle->set_extra_infos(new_extra_info->get_gpu_handle());
    NeuronModel::set_extra_infos(new_extra_info);
}
}
