#pragma once

#include "NeuronModelGPU.h"

namespace gpu::models {
class PoissonModelGPU : public NeuronModelGPU {
public:
    virtual void update_activity(RelearnGPUTypes::step_type step) override;

private:
    double x_0;
    double tau_x;
    unsigned int refractory_period;
};
}
