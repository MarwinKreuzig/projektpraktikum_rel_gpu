#include "IzhikevichModelGPU.h"
#include "ModelKernels.h"

namespace gpu::models {
void IzhikevichModelGPU::update_activity(RelearnGPUTypes::step_type step) {
    init_izhikevich_kernel(step, gpu_handle.get(), number_neurons, model_data_handle, V_spike, a, b, c, d, k1, k2, k3);
}
}
