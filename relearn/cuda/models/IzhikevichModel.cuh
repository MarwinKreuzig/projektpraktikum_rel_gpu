#pragma once

#include "models/NeuronModel.cuh"

#include "gpu/Interface.h"

#include "NeuronsExtraInfos.cuh"
#include "Commons.cuh"
#include "models/NeuronModel.cuh"
#include "Random.cuh"

#include "calculations/NeuronModelCalculations.h"

#include "enums/FiredStatus.h"

namespace gpu::models::izhekevich {

    __device__ __constant__ double V_spike;
    __device__ __constant__ double a;
    __device__ __constant__ double b;
    __device__ __constant__ double c;
    __device__ __constant__ double d;
    __device__ __constant__ double k1;
    __device__ __constant__ double k2;
    __device__ __constant__ double k3;

    __device__ gpu::Vector::CudaArray<double> u;
    gpu::Vector::CudaArrayDeviceHandle<double> handle_u{u};

    double host_c;

void construct_gpu(const unsigned int _h, double _V_spike, double _a, double _b, double _c, double _d, double _k1, double _k2, double _k3) {
    gpu::models::NeuronModel::construct_gpu(_h);

    cuda_copy_to_device(V_spike, _V_spike);
    cuda_copy_to_device(a, _a);
    cuda_copy_to_device(b, _b);
    cuda_copy_to_device(c, _c);
    cuda_copy_to_device(d, _d);
    cuda_copy_to_device(k1, _k1);
    cuda_copy_to_device(k2, _k2);
    cuda_copy_to_device(k3, _k3);
    host_c = _c;
}

void init_gpu(RelearnTypes::number_neurons_type number_neurons) {
    gpu::models::NeuronModel::init_neuron_model(number_neurons);

    handle_u.resize(number_neurons, 0);
    
}

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) {
    gpu::models::NeuronModel::handle_x.fill(start_id,end_id,host_c);
    gpu::models::NeuronModel::handle_x.print_content();
    handle_u.print_content();
}

__global__ void update_activity_kernel(size_t step) {


    const auto neuron_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);

    if (neuron_id >= gpu::neurons::NeuronsExtraInfos::number_local_neurons_device) {
        return;
    }

    if (gpu::neurons::NeuronsExtraInfos::disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }
        const auto synaptic_input = gpu::models::NeuronModel::get_synaptic_input(neuron_id);
        const auto background_activity = gpu::models::NeuronModel::get_background_activity(neuron_id);
        const auto stimulus = gpu::models::NeuronModel::get_stimulus(neuron_id);

        const auto _x = gpu::models::NeuronModel::x[neuron_id];

        const auto _u = u[neuron_id];

        const auto& [x_val, this_fired, u_val] = Calculations::izhikevich(_x, synaptic_input, background_activity, stimulus, _u, gpu::models::NeuronModel::h, gpu::models::NeuronModel::scale, V_spike, a, b,c, d, k1, k2, k3);

        u[neuron_id] = u_val;
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
        gpu::models::izhekevich::update_activity_kernel<<<num_blocks, num_threads>>>(step);

        cudaDeviceSynchronize();
        gpu_check_last_error();

        gpu::models::NeuronModel::finish_update();
}

void create_neurons_gpu(size_t creation_count) {
    const auto old_size = gpu::neurons::NeuronsExtraInfos::number_local_neurons_host;
    const auto new_size = old_size + creation_count;
    handle_u.resize(new_size);

    gpu::neurons::NeuronsExtraInfos::create_neurons(creation_count);
}
};