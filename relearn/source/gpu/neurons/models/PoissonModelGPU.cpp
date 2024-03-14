#include "PoissonModelGPU.h"
#include "ModelKernels.h"

namespace gpu::models {
void PoissonModelGPU::update_activity(RelearnGPUTypes::step_type step) {
    init_poisson_kernel(step, gpu_handle.get(), number_neurons, model_data_handle, x_0, tau_x, refractory_period);
}
}
