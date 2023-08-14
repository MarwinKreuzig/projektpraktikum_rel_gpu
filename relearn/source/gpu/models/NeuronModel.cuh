#pragma once

#include "neurons/enums/FiredStatus.h"

#include "gpu/Commons.cuh"
#include "gpu/GpuTypes.h"
#include "gpu/NeuronsExtraInfos.cuh"

#include <numeric>

namespace gpu::models::NeuronModel {
__device__ double* x;
__device__ FiredStatus* fired;

FiredStatus* fired_host;

__device__ __constant__ unsigned int h;

//__device__ UpdateStatus* disable_flags;

void construct_gpu(const unsigned int _h) {
    cuda_copy_to_device(h, _h);
}

void init_neuron_model(const RelearnTypes::number_neurons_type number_neurons) {

    cuda_calloc_symbol(x, sizeof(double) * number_neurons, 0);
    cuda_calloc_symbol( fired, sizeof(double) * number_neurons, 0);
    fired_host = (FiredStatus*) malloc( sizeof(char) * number_neurons);

    cuda_memcpy_to_host_symbol(fired, fired_host, sizeof(char), number_neurons);
}

__device__ void set_x(const size_t neuron_id, double _x) {
    x[neuron_id] = _x;
}

__device__ void set_fired(const size_t neuron_id, FiredStatus _fired) {
    fired[neuron_id] = _fired;
}

FiredStatus* get_fired() {
    return fired_host;
}

void finish_update() {
    cuda_memcpy_to_host_symbol(fired, fired_host, sizeof(char), gpu::neurons::NeuronsExtraInfos::number_local_neurons_host);
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