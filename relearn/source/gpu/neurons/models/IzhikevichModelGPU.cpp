#include "IzhikevichModelGPU.h"
#include "ModelKernels.h"

namespace gpu::models {
IzhikevichModelGPU::IzhikevichModelGPU(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator, double a, double b, double c, double d, double V_spike, double k1, double k2, double k3) { }

void IzhikevichModelGPU::update_activity(RelearnGPUTypes::step_type step) {
    init_izhikevich_kernel(step, gpu_handle.get(), number_neurons, model_data_handle.get(), V_spike, a, b, c, d, k1, k2, k3, scale, h);
}

std::string IzhikevichModelGPU::name() {
    return "IzhikevichModel";
}

std::unique_ptr<NeuronModel> IzhikevichModelGPU::clone() const {
    return std::make_unique<IzhikevichModelGPU>(get_h(), get_synaptic_input_calculator()->clone(), get_background_activity_calculator()->clone(), get_stimulus_calculator()->clone(), a, b, c, d, V_spike, k1, k2, k3);
}
}
