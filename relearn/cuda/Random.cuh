#pragma once

#include "Commons.cuh"
#include "NeuronsExtraInfos.cuh"

#include "cuda.h"
#include "curand.h"
#include "curand_kernel.h"

namespace gpu::RandomHolder {

    __device__ int seed = 42;

    using random_state_type = curandStateXORWOW_t;

        //Use different key for each kernel
        enum RandomKeyHolder {
            POISSON,
            BACKGROUND,
        };

        __device__ size_t number_neurons;

    /**
     * Call this method at the beginnong of every kernel call if you want to use random values
     * @param step Current step
     * @param _number_neurons Number of local neurons
     * @param key The random key. Use separate for every kernel
     * @param neuron_id The neuron id that is currently processed. Alternativly this can be a id that identifies the thread
     * @return A random state. Use it for every call to this namespace in this kernel
    */
    __device__ random_state_type init(const size_t step, size_t _number_neurons, const RandomKeyHolder key, const size_t neuron_id) {
        number_neurons = _number_neurons;
        random_state_type state;
        curand_init(seed+step, neuron_id, key, &state);
        return state;
    }

    /**
     * @param state The random state returned by the init(..) method
     * Skip to next random value. No need to call from the outside
    */
    __device__ void skip_to_next_item(curandState* state) {
        skipahead(number_neurons, state);
    }

    /**
     * @param state The random state returned by the init(..) method
     * @return Returns random value between 0 and 1
    */
    __device__ double get_percentage(curandState* state) {
        const auto value = curand_uniform_double(state);
        skip_to_next_item(state);
        return value;      
    }

    /**
     * @param state The random state returned by the init(..) method
     * @return Returns random value from normal distribution N(0,1)
    */
    __device__ double get_normal(curandState* state) {
        const auto value = curand_normal_double(state);
        skip_to_next_item(state);
        return value;      
    }

    /**
     * @param mean The mean of the normal distribution
     * @param stddev The standard deviation of the normal distribution
     * @param state The random state returned by the init(..) method 
     * @return Returns random value from normal distribution 
    */
    __device__ double get_normal(curandState* state, double mean, double stddev) {
        const auto value = get_normal(state)*stddev + mean;
        return value;
    }
};