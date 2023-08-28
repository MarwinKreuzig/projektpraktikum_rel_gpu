#pragma once

#include "cuda.h"
#include "curand.h"
#include "curand_kernel.h"

namespace gpu::RandomHolder {

    __device__ int seed = 42;

    using random_state_type = curandStateXORWOW_t;

        enum RandomKeyHolder {
            POISSON
        };

    __device__ random_state_type init(const size_t step, const RandomKeyHolder key, const size_t neuron_id) {
        random_state_type state;
        curand_init(seed+step, neuron_id, key, &state);
        return state;
    }

    __device__ void skip_to_next_item(curandState* state) {
        const auto& num_neurons = gpu::neurons::NeuronsExtraInfos::number_local_neurons_device;
        skipahead(num_neurons, state);
    }

    __device__ double get_percentage(curandState* state) {
        const auto value = curand_uniform_double(state);
        skip_to_next_item(state);
        return value;      
    }
};