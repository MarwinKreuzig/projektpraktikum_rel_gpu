#include "AEIFModel.cuh"
#include "../../utils/Interface.h"

#include "../NeuronsExtraInfos.cuh"
#include "../../Commons.cuh"
#include "NeuronModel.cuh"
#include "../../utils/Random.cuh"

#include "../../../shared/calculations/NeuronModelCalculations.h"

#include "../../../shared/enums/FiredStatus.h"

namespace gpu::models {
__device__ AEIF::AEIF(unsigned int h, gpu::background::BackgroundActivity* bgc, double _C, double _g_L, double _E_L, double _V_T, double _d_T, double _tau_w, double _a, double _b, double _V_spike)
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

__device__ void AEIF::init(RelearnGPUTypes::number_neurons_type number_neurons) {
    NeuronModel::init(number_neurons);

    w.resize(number_neurons, 0);
}

__device__ void AEIF::init_neurons(const RelearnGPUTypes::number_neurons_type start_id, const RelearnGPUTypes::number_neurons_type end_id) {
    NeuronModel::init_neurons(start_id, end_id);
    x.fill(start_id, end_id, E_L);
}

__device__ void AEIF::update_activity(RelearnGPUTypes::step_type step) {

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

__device__ void AEIF::create_neurons(RelearnGPUTypes::number_neurons_type creation_count) {
    NeuronModel::create_neurons(creation_count);
    const auto new_size = extra_infos->get_number_local_neurons();
    w.resize(new_size);
}

namespace aeif {
    std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int _h, double _C, double _g_L, double _E_L, double _V_T, double _d_T, double _tau_w, double _a, double _b, double _V_spike) {
        return gpu::models::construct<gpu::models::AEIF>(background_handle, _h, _C, _g_L, _E_L, _V_T, _d_T, _tau_w, _a, _b, _V_spike);
    }
};
};