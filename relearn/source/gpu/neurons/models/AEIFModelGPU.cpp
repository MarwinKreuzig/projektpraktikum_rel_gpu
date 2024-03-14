#include "AEIFModelGPU.h"
#include "ModelKernels.h"

namespace gpu::models {
    void AEIFModelGPU::update_activity(RelearnGPUTypes::step_type step) {
        init_aeif_kernel(step, gpu_handle.get(), number_neurons, model_data_handle, C, g_L, E_L, V_T, d_T, tau_w, a, b, V_spike, d_T_inverse, tau_w_inverse, C_inverse, scale, h);
    }
}
