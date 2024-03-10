#pragma once

#include "NeuronModel.cuh"

namespace gpu::models {

class FitzHughNagumo : public NeuronModel {

public:
    __device__ FitzHughNagumo(const unsigned int _h, gpu::background::BackgroundActivity* bgc, double _a, double _b, double _phi, double _init_w, double _init_x);

    double a;
    double b;
    double phi;
    double init_w;
    double init_x;

    gpu::Vector::CudaVector<double> w;

    __device__ void init(RelearnGPUTypes::number_neurons_type number_neurons) override;

    __device__ void init_neurons(const RelearnGPUTypes::number_neurons_type start_id, const RelearnGPUTypes::number_neurons_type end_id) override;

    __device__ void update_activity(RelearnGPUTypes::step_type step) override;

    __device__ void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) override;
};

namespace fitz_hugh_nagumo {

    std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int _h, double _a, double _b, double _phi, double _init_w, double _init_x);
};
};