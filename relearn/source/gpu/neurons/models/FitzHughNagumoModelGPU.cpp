#include "FitzHughNagumoModelGPU.h"
#include "ModelKernels.h"

namespace gpu::models {
    void FitzHughNagumoModelGPU::update_activity(RelearnGPUTypes::step_type step) {
        init_fitzhughnagumo_kernel(step, gpu_handle.get(), number_neurons, model_data_handle, a, b, phi, init_w, init_x);
    }
}
