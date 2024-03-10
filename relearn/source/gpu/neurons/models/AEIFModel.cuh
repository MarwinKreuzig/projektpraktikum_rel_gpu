#pragma once

#include "NeuronModel.cuh"

#include "../../utils/Interface.h"

#include "../NeuronsExtraInfos.cuh"
#include "../../Commons.cuh"
#include "NeuronModel.cuh"
#include "../../utils/Random.cuh"

#include "../../../shared/calculations/NeuronModelCalculations.h"

#include "../../../shared/enums/FiredStatus.h"

namespace gpu::models {

    class AEIF : public NeuronModel {
    public:
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

        gpu::Vector::CudaVector<double> w;

        __device__ AEIF(unsigned int h, gpu::background::BackgroundActivity* bgc, double _C, double _g_L, double _E_L, double _V_T, double _d_T, double _tau_w, double _a, double _b, double _V_spike);

        __device__ void init(RelearnGPUTypes::number_neurons_type number_neurons) override;

        __device__ void init_neurons(const RelearnGPUTypes::number_neurons_type start_id, const RelearnGPUTypes::number_neurons_type end_id) override;

        __device__ void update_activity(RelearnGPUTypes::step_type step) override;

        __device__ void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) override;
    };

    namespace aeif {
        std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int _h, double _C, double _g_L, double _E_L, double _V_T, double _d_T, double _tau_w, double _a, double _b, double _V_spike);
    };
};