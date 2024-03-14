#include "FitzHughNagumoModelGPU.h"
#include "ModelKernels.h"

namespace gpu::models {
FitzHughNagumoModelGPU::FitzHughNagumoModelGPU(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
    std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator, double _a, double _b, double _phi)
    : NeuronModelGPU(create_poisson_model_data(), h, std::move(synaptic_input_calculator), std::move(background_activity_calculator), std::move(stimulus_calculator)), a {_a}, b{_b}, phi{_phi} {
    }

void FitzHughNagumoModelGPU::update_activity(RelearnGPUTypes::step_type step) {
    init_fitzhughnagumo_kernel(step, gpu_handle.get(), number_neurons, model_data_handle.get(), a, b, phi, init_w, init_x, scale, h);
}

std::string FitzHughNagumoModelGPU::name() {
    return "FitzHughNagumoModel";
}

std::unique_ptr<NeuronModel> FitzHughNagumoModelGPU::clone() const {
    return std::make_unique<FitzHughNagumoModelGPU>(get_h(), get_synaptic_input_calculator()->clone(), get_background_activity_calculator()->clone(), get_stimulus_calculator()->clone(), a, b, phi);
}
}
