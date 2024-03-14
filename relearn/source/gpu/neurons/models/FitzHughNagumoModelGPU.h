#pragma once

#include "NeuronModelGPU.h"

namespace gpu::models {
class FitzHughNagumoModelGPU : public NeuronModelGPU {
public:
    FitzHughNagumoModelGPU(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
        std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator, double a, double b, double phi);
    virtual void update_activity(RelearnGPUTypes::step_type step) override;

    virtual std::string name() override;
    virtual std::unique_ptr<NeuronModel> clone() const override;

private:
    double a;
    double b;
    double phi;
    double init_w;
    double init_x;
};
}
