#include "NeuronModelGPU.h"

#include <iostream>

namespace gpu {
NeuronModelGPU::NeuronModelGPU(std::unique_ptr<models::ModelDataHandle> model_data_handle_, const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator)
    : model_data_handle{ std::move(model_data_handle_) }
    , NeuronModel{ h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator) } {
    std::vector<double> x_{};
    std::vector<double> stimulus_{};
    std::vector<double> syn_input_{};
    std::vector<FiredStatus> fired_{};
    
    gpu_handle = create_neuron_model_data(&x_, nullptr, h, scale, 0, get_background_activity_calculator()->get_gpu_handle().get(), &stimulus_, &syn_input_, &fired_);
}

std::unique_ptr<NeuronModelDataHandle> gpu_handle;
RelearnGPUTypes::number_neurons_type number_neurons;
std::unique_ptr<models::ModelDataHandle> model_data_handle;

unsigned int h;
double scale;

size_t cur_step;

// This looks sketchy, but the CPU neuron models don't do anything either in these functions
void NeuronModelGPU::enable_neurons(const std::span<const NeuronID> neuron_ids) { }

void NeuronModelGPU::disable_neurons(const std::span<const NeuronID> neuron_ids) { }

void NeuronModelGPU::set_fired(const NeuronID neuron_id, const FiredStatus new_value) {
    NeuronModel::set_fired(neuron_id, new_value);
    gpu_handle->set_fired(neuron_id.get_neuron_id(), new_value);
}

void NeuronModelGPU::set_fired(std::vector<FiredStatus> new_values) {
    gpu_handle->set_fired(&new_values);
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
