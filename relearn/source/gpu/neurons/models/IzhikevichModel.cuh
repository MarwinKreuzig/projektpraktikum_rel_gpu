#pragma once

#include "NeuronModel.cuh"

#include "../../utils/Interface.h"

#include "../NeuronsExtraInfos.cuh"
#include "../../Commons.cuh"
#include "NeuronModel.cuh"
#include "../../utils/Random.cuh"

#include "../../../shared/calculations/NeuronModelCalculations.h"

#include "../../../shared/enums/FiredStatus.h"

#include "../input/BackgroundActivity.cuh"

#include <iostream>

namespace gpu::models {

class Izhikevich : public NeuronModel {
private:
    double V_spike;
    double a;
    double b;
    double c;
    double d;
    double k1;
    double k2;
    double k3;

    gpu::Vector::CudaVector<double> u;

    double host_c;

public:
    __device__ Izhikevich(const unsigned int _h, gpu::background::BackgroundActivity* bgc, double _V_spike, double _a, double _b, double _c, double _d, double _k1, double _k2, double _k3);

    __device__ void init(RelearnGPUTypes::number_neurons_type number_neurons) override;

    __device__ void update_activity(RelearnGPUTypes::step_type step) override;

    __device__ void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) override;

    __device__ void init_neurons(const RelearnGPUTypes::number_neurons_type start_id, const RelearnGPUTypes::number_neurons_type end_id) override;
};

namespace izhikevich {

    std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int _h, double V_spike, double a, double b, double c, double d, double k1, double k2, double k3);
};

};