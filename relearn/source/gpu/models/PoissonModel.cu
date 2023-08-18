#include "gpu/Interface.h"

#include "gpu/NeuronsExtraInfos.cuh"
#include "gpu/Commons.cuh"
#include "gpu/models/NeuronModel.cuh"
#include "gpu/Random.cuh"
#include "gpu/RelearnGPUException.h"

#include "neurons/enums/FiredStatus.h"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>

#include <iostream>

namespace gpu::models::izhekevich {

void construct_gpu(const unsigned int _h) {
    gpu::models::NeuronModel::construct_gpu(_h);
}

void init_gpu(RelearnTypes::number_neurons_type number_neurons) {
    RelearnGPUException::fail("No gpu support");
}

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) {
    RelearnGPUException::fail("No gpu support");
}

void update_activity_gpu(const size_t step) {
    RelearnGPUException::fail("No gpu support");
}

void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count) {
    RelearnGPUException::fail("No gpu support");
}
};

namespace gpu::models::poisson {

    __device__ gpu::Vector::CudaArray<double> refractory_time;
    gpu::Vector::CudaArrayDeviceHandle<double> handle_refractory_time{refractory_time};

    __device__ __constant__ double x_0;
    __device__ __constant__ double tau_x;
    __device__ __constant__ unsigned int refractory_period;

        void
        construct_gpu(const unsigned int _h, const double _x_0,
            const double _tau_x,
            const unsigned int _refractory_period) {
    gpu::models::NeuronModel::construct_gpu(_h);

    cuda_copy_to_device(x_0,_x_0);
    cuda_copy_to_device(tau_x, _tau_x);
    cuda_copy_to_device(refractory_period, _refractory_period);
    }

void init_gpu(const RelearnTypes::number_neurons_type number_neurons) {
    std::cout << "INIT GPU " << std::endl;
    gpu::models::NeuronModel::init_neuron_model(number_neurons);

    handle_refractory_time.resize(number_neurons);
}

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) {

}

void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count) {
    RelearnGPUException::fail("No gpu support");
}

__global__ void update_activity_kernel(size_t step) {

    const auto scale = 1.0 / gpu::models::NeuronModel::h;
    const auto tau_x_inverse = 1.0 / tau_x;

    const auto neuron_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);

    if (neuron_id >= gpu::neurons::NeuronsExtraInfos::number_local_neurons_device) {
        return;
    }

    //Init
    auto curand_state = gpu::RandomHolder::init(step, gpu::RandomHolder::POISSON, neuron_id);

    if (gpu::neurons::NeuronsExtraInfos::disable_flags[neuron_id] == 0) {
            return;
    }
        const auto synaptic_input = gpu::models::NeuronModel::get_synaptic_input(neuron_id);
        const auto background = gpu::models::NeuronModel::get_background_activity(neuron_id);
        const auto stimulus = gpu::models::NeuronModel::get_stimulus(neuron_id);
        const auto input = synaptic_input + background + stimulus;

        auto x_val = gpu::models::NeuronModel::x[neuron_id];

        for (unsigned int integration_steps = 0; integration_steps < gpu::models::NeuronModel::h; integration_steps++) {
            x_val += ((x_0 - x_val) * tau_x_inverse + input) * scale;
        }

        if (refractory_time[neuron_id] == 0) {
            const auto threshold = gpu::RandomHolder::get_percentage(&curand_state);
            const auto f = x_val >= threshold;
            if (f) {
                gpu::models::NeuronModel::set_fired(neuron_id, FiredStatus::Fired);
                refractory_time[neuron_id] = refractory_period;
            } else {
                gpu::models::NeuronModel::set_fired(neuron_id, FiredStatus::Inactive);
            }
        } else {
            gpu::models::NeuronModel::set_fired(neuron_id, FiredStatus::Inactive);
            --refractory_time[neuron_id];
        }

        gpu::models::NeuronModel::set_x(0, 0.0);
}

void update_activity_gpu(size_t step, const double* stimulus, const double* background, const double* syn_input, size_t num_neurons) {
    
    
    gpu::models::NeuronModel::prepare_update(step, stimulus, background, syn_input, num_neurons);

        const auto number_local_neurons = gpu::neurons::NeuronsExtraInfos::number_local_neurons_host;
        const auto num_threads = get_number_threads(gpu::models::poisson::update_activity_kernel, number_local_neurons);
        const auto num_blocks = get_number_blocks(num_threads, number_local_neurons);

        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, 0);
        //std::cout << "using " << properties.multiProcessorCount << " multiprocessors" << std::endl;
        //std::cout << "max threads per processor: " << properties.maxThreadsPerMultiProcessor << std::endl;

        //std::cout << "starting with " << num_blocks << " blocks and " << num_threads << " threads" << std::endl;

        cudaDeviceSynchronize();
        gpu_check_last_error();
        gpu::models::poisson::update_activity_kernel<<<num_blocks, num_threads>>>(step);

        cudaDeviceSynchronize();
        gpu_check_last_error();

        gpu::models::NeuronModel::finish_update();
}
};