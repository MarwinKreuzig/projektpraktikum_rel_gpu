#pragma once

#include "NeuronModel.cuh"

#include "../../utils/Interface.h"

#include "../NeuronsExtraInfos.cuh"
#include "../../Commons.cuh"
#include "NeuronModel.cuh"
#include "../../utils/Random.cuh"

#include "../../../shared/calculations/NeuronModelCalculations.h"

#include "../../../shared/enums/FiredStatus.h"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

namespace gpu::models {

class PoissonModel : public NeuronModel {
public:
    __device__ PoissonModel(const unsigned int _h, gpu::background::BackgroundActivity* bgc, const double _x_0,
        const double _tau_x,
        const unsigned int _refractory_period);

    gpu::Vector::CudaVector<double> refractory_time;

    double x_0;
    double tau_x;
    unsigned int refractory_period;

    __device__ void init(const RelearnGPUTypes::number_neurons_type number_neurons) override;

    __device__ void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) override;

    __device__ void update_activity(RelearnGPUTypes::step_type step) override;
};

namespace poisson {

    std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int _h, double x_0, double _tau_x, const unsigned int _refractory_period);
};
};