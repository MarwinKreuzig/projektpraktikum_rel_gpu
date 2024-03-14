#include "AEIFModelGPU.h"
#include "ModelKernels.h"

namespace gpu::models {
AEIFModelGPU::AEIFModelGPU(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator, double C, double g_L, double E_L, double V_T, double d_T, double tau_w, double a, double b, double V_spike) { }

void AEIFModelGPU::update_activity(RelearnGPUTypes::step_type step) {
    init_aeif_kernel(step, gpu_handle.get(), number_neurons, model_data_handle.get(), C, g_L, E_L, V_T, d_T, tau_w, a, b, V_spike, d_T_inverse, tau_w_inverse, C_inverse, scale, h);
}

std::string AEIFModelGPU::name() {
    return "AEIFModel";
}

std::unique_ptr<NeuronModel> AEIFModelGPU::clone() const {
    return std::make_unique<AEIFModelGPU>(get_h(), get_synaptic_input_calculator()->clone(), get_background_activity_calculator()->clone(), get_stimulus_calculator()->clone(), C, g_L, E_L, V_T, d_T, tau_w, a, b, V_spike);
}
}
