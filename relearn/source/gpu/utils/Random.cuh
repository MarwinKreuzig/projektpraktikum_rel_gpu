#pragma once

#include "curand_kernel.h"

namespace gpu::RandomHolder {

    __device__ extern int seed;

    __device__ extern size_t number_neurons;

using random_state_type = curandStateXORWOW_t;

// Use different key for each kernel
enum RandomKeyHolder {
    POISSON,
    BACKGROUND,
};

/**
 * Call this method at the beginnong of every kernel call if you want to use random values
 * @param step Current step
 * @param _number_neurons Number of local neurons
 * @param key The random key. Use separate for every kernel
 * @param neuron_id The neuron id that is currently processed. Alternativly this can be a id that identifies the thread
 * @return A random state. Use it for every call to this namespace in this kernel
 */
__device__ random_state_type init(const size_t step, size_t _number_neurons, const RandomKeyHolder key, const size_t neuron_id);

/**
 * @param state The random state returned by the init(..) method
 * Skip to next random value. No need to call from the outside
 */
__device__ void skip_to_next_item(curandState* state);

/**
 * @param state The random state returned by the init(..) method
 * @return Returns random value between 0 and 1
 */
__device__ double get_percentage(curandState* state);

/**
 * @param state The random state returned by the init(..) method
 * @return Returns random value from normal distribution N(0,1)
 */
__device__ double get_normal(curandState* state);

/**
 * @param mean The mean of the normal distribution
 * @param stddev The standard deviation of the normal distribution
 * @param state The random state returned by the init(..) method
 * @return Returns random value from normal distribution
 */
__device__ double get_normal(curandState* state, double mean, double stddev);
};