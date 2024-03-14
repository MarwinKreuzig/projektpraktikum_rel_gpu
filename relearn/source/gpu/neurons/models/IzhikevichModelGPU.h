#pragma once

#include "NeuronModelGPU.h"

namespace gpu::models {
class IzhikevichModelGPU : public NeuronModelGPU {
public:
    IzhikevichModelGPU(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
        std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator, double a, double b, double c, double d, double V_spike, double k1, double k2, double k3);
    virtual void update_activity(RelearnGPUTypes::step_type step) override;

    virtual std::string name() override;
    virtual std::unique_ptr<NeuronModel> clone() const override;

private:
    double V_spike;
    double a;
    double b;
    double c;
    double d;
    double k1;
    double k2;
    double k3;
    double host_c;
};
}
