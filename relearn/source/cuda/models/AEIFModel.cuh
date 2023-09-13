#pragma once

#include "models/NeuronModel.cuh"

#include "gpu/Interface.h"

#include "NeuronsExtraInfos.cuh"
#include "Commons.cuh"
#include "models/NeuronModel.cuh"
#include "Random.cuh"

#include "calculations/NeuronModelCalculations.h"

#include "enums/FiredStatus.h"

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

    __device__ AEIF(unsigned int h, gpu::background::BackgroundActivity* bgc, double _C, double _g_L, double _E_L, double _V_T, double _d_T, double _tau_w, double _a, double _b, double _V_spike)
        : NeuronModel(h, bgc)
        , C(_C)
        , g_L(_g_L)
        , E_L(_E_L)
        , V_T(_V_T)
        , d_T(_d_T)
        , tau_w(_tau_w)
        , a(_a)
        , b(_b)
        , V_spike(_V_spike) {

        d_T_inverse = 1.0 / _d_T;
        tau_w_inverse = 1.0 / _tau_w;
        C_inverse = 1.0 / _C;
    }

    __device__ void init(RelearnGPUTypes::number_neurons_type number_neurons) override {
        NeuronModel::init(number_neurons);

        w.resize(number_neurons, 0);
    }

    __device__ void init_neurons(const RelearnGPUTypes::number_neurons_type start_id, const RelearnGPUTypes::number_neurons_type end_id) override {
        NeuronModel::init_neurons(start_id, end_id);
        x.fill(start_id, end_id, E_L);
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

        const auto _x = get_x(neuron_id);

        const auto _w = w[neuron_id];

        const auto& [x_val, this_fired, w_val] = Calculations::aeif(_x, synaptic_input, background_activity, stimulus, _w, gpu::models::NeuronModel::h, gpu::models::NeuronModel::scale, V_spike, g_L, E_L, V_T, d_T, d_T_inverse, a, b, C_inverse, tau_w_inverse);

        w[neuron_id] = w_val;
        set_x(neuron_id, x_val);
        set_fired(neuron_id, this_fired);
    }

    __device__ void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) override {
        NeuronModel::create_neurons(creation_count);
        const auto new_size = extra_infos->get_number_local_neurons();
        w.resize(new_size);
    }
};

namespace aeif {
    std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int _h, double _C, double _g_L, double _E_L, double _V_T, double _d_T, double _tau_w, double _a, double _b, double _V_spike) {
        return gpu::models::construct<gpu::models::AEIF>(background_handle, _h, _C, _g_L, _E_L, _V_T, _d_T, _tau_w, _a, _b, _V_spike);
    }
};
};