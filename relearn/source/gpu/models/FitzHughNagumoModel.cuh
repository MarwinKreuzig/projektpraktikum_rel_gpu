#pragma once

#include "gpu/models/NeuronModel.cuh"

#include "gpu/Interface.h"

#include "gpu/NeuronsExtraInfos.cuh"
#include "gpu/Commons.cuh"
#include "gpu/models/NeuronModel.cuh"
#include "gpu/Random.cuh"
#include "gpu/RelearnGPUException.h"

#include "calculations/NeuronModelCalculations.h"

#include "neurons/enums/FiredStatus.h"

namespace gpu::models::fitz_hugh_nagumo {

    __device__ __constant__ double a;
    __device__ __constant__ double b;
    __device__ __constant__ double phi;
    __device__ __constant__ double init_w;
    __device__ __constant__ double init_x;
   

    __device__ gpu::Vector::CudaArray<double> w;
    gpu::Vector::CudaArrayDeviceHandle<double> handle_w{w};

    double host_init_w;
    double host_init_x;

void construct_gpu(const unsigned int _h, double _a, double _b, double _phi, double _init_w, double _init_x) {
    gpu::models::NeuronModel::construct_gpu(_h);

    cuda_copy_to_device(a, _a);
    cuda_copy_to_device(b, _b);
    cuda_copy_to_device(phi, _phi);
    cuda_copy_to_device(init_w, _init_w);
    cuda_copy_to_device(init_x, _init_x);
    host_init_w = _init_w;
    host_init_x = _init_x;
}

void init_gpu(RelearnTypes::number_neurons_type number_neurons) {
    gpu::models::NeuronModel::init_neuron_model(number_neurons);

    handle_w.resize(number_neurons, 0);
}

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) {
    gpu::models::NeuronModel::handle_x.fill(start_id,end_id,host_init_x);
    handle_w.fill(start_id,end_id,host_init_w);
}

__global__ void update_activity_kernel(size_t step) {


    const auto neuron_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);

    if (neuron_id >= gpu::neurons::NeuronsExtraInfos::number_local_neurons_device) {
        return;
    }

    if (gpu::neurons::NeuronsExtraInfos::disable_flags[neuron_id] == 0) {
            return;
    }
        const auto synaptic_input = gpu::models::NeuronModel::get_synaptic_input(neuron_id);
        const auto background_activity = gpu::models::NeuronModel::get_background_activity(neuron_id);
        const auto stimulus = gpu::models::NeuronModel::get_stimulus(neuron_id);

        const auto _x = gpu::models::NeuronModel::get_x(neuron_id);

        const auto _w = w[neuron_id];

        const auto& [x_val, this_fired, w_val] = Calculations::fitz_hugh_nagumo(_x,  synaptic_input,  background_activity,  stimulus,  _w,  gpu::models::NeuronModel::h,  gpu::models::NeuronModel::scale,phi, a, b);

        w[neuron_id] = w_val;
        gpu::models::NeuronModel::set_x(neuron_id, x_val);
        gpu::models::NeuronModel::set_fired(neuron_id, this_fired);
}

void update_activity_gpu(const size_t step,  const double* stimulus, const double* background, const double* syn_input, size_t num_neurons) {
    gpu::models::NeuronModel::prepare_update(step, stimulus, background, syn_input, num_neurons);

        const auto number_local_neurons = gpu::neurons::NeuronsExtraInfos::number_local_neurons_host;
        const auto num_threads = get_number_threads(gpu::models::poisson::update_activity_kernel, number_local_neurons);
        const auto num_blocks = get_number_blocks(num_threads, number_local_neurons);

        cudaDeviceSynchronize();
        gpu_check_last_error();
        update_activity_kernel<<<num_blocks, num_threads>>>(step);

        cudaDeviceSynchronize();
        gpu_check_last_error();

        gpu::models::NeuronModel::finish_update();
}

void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count) {
    RelearnGPUException::fail("No gpu support");
}
};