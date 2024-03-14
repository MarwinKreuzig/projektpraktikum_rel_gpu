#pragma once

#include "NeuronModelGPU.h"

namespace gpu::models {
class PoissonModelGPU : public NeuronModelGPU {
public:
    PoissonModelGPU(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
        std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator, double x_0, double tau_x, unsigned int refractory_period);

    virtual void update_activity(RelearnGPUTypes::step_type step) override;

    virtual std::string name() override;
    virtual std::unique_ptr<NeuronModel> clone() const override;

private:
    double x_0;
    double tau_x;
    unsigned int refractory_period;
};
}
