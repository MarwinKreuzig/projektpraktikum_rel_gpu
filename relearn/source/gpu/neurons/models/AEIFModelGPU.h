#pragma once

#include "NeuronModelGPU.h"

namespace gpu::models {
class AEIFModelGPU : public NeuronModelGPU {
public:
    virtual void update_activity(RelearnGPUTypes::step_type step) override;

private:
    double C;
    double g_L;
    double E_L;
    double V_T;
    double d_T;
    double tau_w;
    double a;
    double b;
    double V_spike;

    double d_T_inverse;
    double tau_w_inverse;
    double C_inverse;
};
}
