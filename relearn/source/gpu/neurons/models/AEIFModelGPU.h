#pragma once

#include "NeuronModelGPU.h"

namespace gpu::models {
class AEIFModelGPU : public NeuronModelGPU {
public:
    AEIFModelGPU() = default;
    AEIFModelGPU(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
        std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator, double C, double g_L, double E_L, double V_T, double d_T, double tau_w, double a, double b, double V_spike);
    virtual void update_activity(RelearnGPUTypes::step_type step) override;

    virtual std::string name() override;
    virtual std::unique_ptr<NeuronModel> clone() const override;

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
