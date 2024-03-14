#include "IzhikevichModelGPU.h"
#include "ModelKernels.h"

namespace gpu::models {
IzhikevichModelGPU::IzhikevichModelGPU(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator, double _a, double _b, double _c, double _d, double _V_spike, double _k1, double _k2, double _k3)
    : NeuronModelGPU(create_poisson_model_data(), h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator))
    , a{ _a }
    , b{ _b }
    , c{ _c }
    , d{ _d }
    , V_spike{ _V_spike }
    , k1{ _k1 }
    , k2{ _k2 }
    , k3{ _k3 } { }

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
