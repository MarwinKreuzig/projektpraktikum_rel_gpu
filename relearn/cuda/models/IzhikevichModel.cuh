#pragma once

#include "models/NeuronModel.cuh"

#include "gpu/Interface.h"

#include "NeuronsExtraInfos.cuh"
#include "Commons.cuh"
#include "models/NeuronModel.cuh"
#include "Random.cuh"

#include "calculations/NeuronModelCalculations.h"

#include "enums/FiredStatus.h"

#include "background/BackgroundActivity.cuh"

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
    __device__ Izhikevich(const unsigned int _h, gpu::background::BackgroundActivity* bgc, double _V_spike, double _a, double _b, double _c, double _d, double _k1, double _k2, double _k3)
        : NeuronModel(_h, bgc)
        , V_spike(_V_spike)
        , a(_a)
        , b(_b)
        , c(_c)
        , d(_d)
        , k1(_k1)
        , k2(_k2)
        , k3(_k3) { }

    __device__ void init(RelearnGPUTypes::number_neurons_type number_neurons) override {
        printf("Neuron model init2\n");
        NeuronModel::init(number_neurons);

        u.resize(number_neurons, 0);
    }

    __device__ void update_activity(RelearnGPUTypes::step_type step) override {

        const auto neuron_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);

        if (neuron_id >= extra_infos->get_number_local_neurons()) {
            return;
        }

        if (extra_infos->disable_flags[neuron_id] == UpdateStatus::Disabled) {
            return;
        }
        const auto synaptic_input = get_synaptic_input(step, neuron_id);
        const auto background_activity = get_background_activity(step, neuron_id);
        const auto stimulus = get_stimulus(step, neuron_id);
        // bg__[neuron_id] = background_activity;

        const auto _x = x[neuron_id];

        const auto _u = u[neuron_id];

        const auto& [x_val, this_fired, u_val] = Calculations::izhikevich(_x, synaptic_input, background_activity, stimulus, _u, h, scale, V_spike, a, b, c, d, k1, k2, k3);

        u[neuron_id] = u_val;
        set_x(neuron_id, x_val);
        set_fired(neuron_id, this_fired);
    }

    __device__ void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) override {
        NeuronModel::create_neurons(creation_count);
        const auto new_size = extra_infos->get_number_local_neurons();
        u.resize(new_size);
    }

    __device__ void init_neurons(const RelearnGPUTypes::number_neurons_type start_id, const RelearnGPUTypes::number_neurons_type end_id) override {
        x.fill(start_id, end_id, c);
    }
};

namespace izhikevich {

    std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int _h, double V_spike, double a, double b, double c, double d, double k1, double k2, double k3) {
        return gpu::models::construct<gpu::models::Izhikevich>(background_handle, _h, V_spike, a, b, c, d, k1, k2, k3);
    }
};

};