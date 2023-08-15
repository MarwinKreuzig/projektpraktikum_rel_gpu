#pragma once

#include "neurons/enums/FiredStatus.h"

#include "gpu/Commons.cuh"
#include "gpu/GpuTypes.h"
#include "gpu/NeuronsExtraInfos.cuh"

//#include "gpu/neurons/NetworkGraph.cuh"

#include "gpu/CudaVector.cuh"

#include <numeric>

namespace gpu::models::NeuronModel {
__device__ gpu::Vector::CudaArray<double> x;
gpu::Vector::CudaArrayDeviceHandle<double> handle_x(x);

__device__ gpu::Vector::CudaArray<FiredStatus> fired;
gpu::Vector::CudaArrayDeviceHandle<FiredStatus> handle_fired(fired);
std::vector<FiredStatus> fired_host;

__device__ __constant__ unsigned int h;

void construct_gpu(const unsigned int _h) {
    cuda_copy_to_device(h, _h);
}

void init_neuron_model(const RelearnTypes::number_neurons_type number_neurons) {
    handle_x.resize(number_neurons);
    handle_fired.resize(number_neurons);
    fired_host.resize(number_neurons);
}

__device__ void set_x(const size_t neuron_id, double _x) {
    x[neuron_id] = _x;
}

__device__ void set_fired(const size_t neuron_id, FiredStatus _fired) {
    fired[neuron_id] = _fired;
}

FiredStatus* get_fired() {
    return fired_host.data();
}

void finish_update() {
    handle_fired.copy_to_host(fired_host);
}

__device__ double get_stimulus(const size_t neuron_id) {
    return 0.0;
}

__device__ double get_background_activity(const size_t neuron_id) {
    return 0.0;

}
__device__ double get_synaptic_input(const size_t neuron_id) {
    return 0.0;
}
};