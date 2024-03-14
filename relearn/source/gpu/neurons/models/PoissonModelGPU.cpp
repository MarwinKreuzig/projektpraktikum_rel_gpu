#include "PoissonModelGPU.h"
#include "ModelKernels.h"

namespace gpu::models {
PoissonModelGPU::PoissonModelGPU(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator, double x_0, double tau_x, unsigned int refractory_period) { 
}

void PoissonModelGPU::update_activity(RelearnGPUTypes::step_type step) {
    init_poisson_kernel(step, gpu_handle.get(), number_neurons, model_data_handle.get(), x_0, tau_x, refractory_period);
}

std::string PoissonModelGPU::name() {
    return "PoissonModel";
}

std::unique_ptr<NeuronModel> PoissonModelGPU::clone() const {
    return std::make_unique<PoissonModelGPU>(get_h(), get_synaptic_input_calculator()->clone(), get_background_activity_calculator()->clone(), get_stimulus_calculator()->clone(), x_0, tau_x, refractory_period);
}
}
