#pragma once

#include "NeuronModelGPU.h"

namespace gpu::models {
class FitzHughNagumoModelGPU : public NeuronModelGPU {
public:
    virtual void update_activity(RelearnGPUTypes::step_type step) override;

private:
    double a;
    double b;
    double phi;
    double init_w;
    double init_x;
};
}
