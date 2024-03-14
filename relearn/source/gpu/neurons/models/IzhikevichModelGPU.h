#pragma once

#include "NeuronModelGPU.h"

namespace gpu::models {
  class IzhikevichModelGPU : public NeuronModelGPU {
    public:
      virtual void update_activity(RelearnGPUTypes::step_type step) override;

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
