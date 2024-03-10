#include "Random.cuh"

#include "../Commons.cuh"
#include "../neurons/NeuronsExtraInfos.cuh"

#include "cuda.h"
#include "curand.h"


namespace gpu::RandomHolder {

    __device__ int seed = 42;

    __device__ size_t number_neurons = 0;

    __device__ random_state_type init(const size_t step, size_t _number_neurons, const RandomKeyHolder key, const size_t neuron_id) {
        number_neurons = _number_neurons;
        random_state_type state;
        curand_init(seed + step, neuron_id, key, &state);
        return state;
    }

    __device__ void skip_to_next_item(curandState* state) {
        skipahead(number_neurons, state);
    }

    __device__ double get_percentage(curandState* state) {
        const auto value = curand_uniform_double(state);
        skip_to_next_item(state);
        return value;
    }

    __device__ double get_normal(curandState* state) {
        const auto value = curand_normal_double(state);
        skip_to_next_item(state);
        return value;
    }

    __device__ double get_normal(curandState* state, double mean, double stddev) {
        const auto value = get_normal(state) * stddev + mean;
        return value;
    }
};