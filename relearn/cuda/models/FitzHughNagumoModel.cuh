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

    class FitzHughNagumo : public NeuronModel{
    

    public:

    __device__ FitzHughNagumo(const unsigned int _h, gpu::background::BackgroundActivity* bgc,double _a, double _b, double _phi, double _init_w, double _init_x) : NeuronModel(_h,bgc), a(_a), b(_b), phi(_phi), init_w(_init_w), init_x(_init_x) {}

     double a;
     double b;
     double phi;
     double init_w;
     double init_x;
   

    gpu::Vector::CudaVector<double> w;


__device__ void init(RelearnTypes::number_neurons_type number_neurons) override {
    NeuronModel::init(number_neurons);

   w.resize(number_neurons, 0);
}

__device__ void init_neurons(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) override {
    NeuronModel::init_neurons(start_id,end_id);
    x.fill(start_id,end_id,init_x);
    w.fill(start_id,end_id,init_w);
}

__device__ void update_activity(size_t step) override{


    const auto neuron_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);

    if (neuron_id >= extra_infos->get_number_local_neurons()) {
        return;
    }

    if (extra_infos->disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }
        const auto synaptic_input = get_synaptic_input(step,neuron_id);
        const auto background_activity = get_background_activity(step,neuron_id);
        const auto stimulus =get_stimulus(step,neuron_id);

        const auto _x = get_x(neuron_id);

        const auto _w = w[neuron_id];

        const auto& [x_val, this_fired, w_val] = Calculations::fitz_hugh_nagumo(_x,  synaptic_input,  background_activity,  stimulus,  _w,  gpu::models::NeuronModel::h,  gpu::models::NeuronModel::scale,phi, a, b);

        w[neuron_id] = w_val;
        set_x(neuron_id, x_val);
        set_fired(neuron_id, this_fired);
}


__device__ void create_neurons(size_t creation_count) override {
    NeuronModel::create_neurons(creation_count);
    const auto new_size = extra_infos->get_number_local_neurons();
    w.resize(new_size);

}
    };

    namespace fitz_hugh_nagumo {

std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle,const unsigned int _h,  double _a, double _b, double _phi, double _init_w, double _init_x) {
    return gpu::models::construct<gpu::models::FitzHughNagumo>(background_handle, _h,  _a,  _b,  _phi,  _init_w,  _init_x);
}
};
};