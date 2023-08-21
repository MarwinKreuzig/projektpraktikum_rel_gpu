#pragma once

#include "neurons/enums/FiredStatus.h"

#include "gpu/Commons.cuh"
#include "gpu/GpuTypes.h"
#include "gpu/NeuronsExtraInfos.cuh"

#include "gpu/CudaVector.cuh"

#include <numeric>

namespace gpu::models::NeuronModel {
__device__ gpu::Vector::CudaArray<double> x;
gpu::Vector::CudaArrayDeviceHandle<double> handle_x(x);

__device__ gpu::Vector::CudaArray<FiredStatus> fired;
gpu::Vector::CudaArrayDeviceHandle<FiredStatus> handle_fired(fired);
std::vector<FiredStatus> fired_host;

__device__ gpu::Vector::CudaArray<double> syn_input;
gpu::Vector::CudaArrayDeviceHandle<double> handle_syn_input(syn_input);

__device__ gpu::Vector::CudaArray<double> background;
gpu::Vector::CudaArrayDeviceHandle<double> handle_background(background);

__device__ gpu::Vector::CudaArray<double> stimulus;
gpu::Vector::CudaArrayDeviceHandle<double> handle_stimulus(stimulus);

__device__ __constant__ unsigned int h;
__device__ __constant__ double scale;

void construct_gpu(const unsigned int _h) {
    cuda_copy_to_device(h, _h);
    const auto _scale = 1.0/_h;
    cuda_copy_to_device(scale, _scale);
}

void init_neuron_model(const RelearnTypes::number_neurons_type number_neurons) {
    handle_x.resize(number_neurons);
    handle_fired.resize(number_neurons);
    fired_host.resize(number_neurons);
}

__device__ inline double get_x(size_t neuron_id) {
    return x[neuron_id];
}

__device__ inline void set_x(const size_t neuron_id, double _x) {
    x[neuron_id] = _x;
}

__device__ inline void set_fired(const size_t neuron_id, FiredStatus _fired) {
    fired[neuron_id] = _fired;
}

FiredStatus* get_fired() {
    return fired_host.data();
}

void finish_update() {
    fired_host.resize(handle_fired.get_size());
    handle_fired.copy_to_host(fired_host);

}

void prepare_update(const size_t step, const double* _stimulus, const double* _background, const double* _syn_input, size_t num_neurons) {
    handle_stimulus.copy_to_device(_stimulus, num_neurons);
    handle_background.copy_to_device(_background, num_neurons);
    handle_syn_input.copy_to_device(_syn_input, num_neurons);
}

    __device__ inline double get_stimulus(const size_t neuron_id) {
    return stimulus[neuron_id];
}

__device__ inline double get_background_activity(const size_t neuron_id) {
    return background[neuron_id];

}
__device__ inline double get_synaptic_input(const size_t neuron_id) {
    return syn_input[neuron_id];
}
};