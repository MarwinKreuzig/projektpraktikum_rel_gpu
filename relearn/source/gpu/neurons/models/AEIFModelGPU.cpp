#include "AEIFModelGPU.h"
#include "ModelKernels.h"

namespace gpu::models {
AEIFModelGPU::AEIFModelGPU(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator, double _C, double _g_L, double _E_L, double _V_T, double _d_T, double _tau_w, double _a, double _b, double _V_spike)
    : NeuronModelGPU(create_aeif_model_data(E_L), h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator))
    , C{ _C }
    , g_L{ _g_L }
    , E_L{ _E_L }
    , V_T{ _V_T }
    , d_T{ _d_T }
    , tau_w{ _tau_w }
    , a{ _a }
    , b{ _b }
    , V_spike{ _V_spike } {
}

void AEIFModelGPU::update_activity(RelearnGPUTypes::step_type step) {
    init_aeif_kernel(step, gpu_handle.get(), get_number_neurons(), model_data_handle.get(), C, g_L, E_L, V_T, d_T, tau_w, a, b, V_spike, d_T_inverse, tau_w_inverse, C_inverse, scale, h);
}

std::string AEIFModelGPU::name() {
    return "AEIFModel";
}

std::unique_ptr<NeuronModel> AEIFModelGPU::clone() const {
    return std::make_unique<AEIFModelGPU>(get_h(), get_synaptic_input_calculator()->clone(), get_background_activity_calculator()->clone(), get_stimulus_calculator()->clone(), C, g_L, E_L, V_T, d_T, tau_w, a, b, V_spike);
}
}
