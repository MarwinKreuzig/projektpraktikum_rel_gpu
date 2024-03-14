#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "../Commons.cuh"
#include "../structure/CudaArray.cuh"

#include "cuda.h"
#include "curand.h"
#include "curand_kernel.h"

namespace gpu::random {

using random_state_type = curandStateXORWOW_t;

// Use different key for each kernel, currently only has barnes hut here, but other kernels using random should be included as well
enum RandomKeyHolder : uint16_t{
    BARNES_HUT = 0,
    COUNT = 1
};

struct RandomStateData {
    /**
    * Stores the data needed for the generation of random numbers on the GPU
    */


    // A state for every thread for every kernel using random
    gpu::Vector::CudaArray<random_state_type> random_states[RandomKeyHolder::COUNT];
    unsigned long long int current_sequence{0};
    // for each experiment run, a new seed should be useds
    int seed{42};

    /**
    * @brief Initializes the random state for a given thread on a given kernel, should be called initially and on resize of the number of threads 
    * @param kernel The kernel to which the thread belongs
    * @param thread_id The id of the thread
    */
    __device__ void init_state(RandomKeyHolder kernel, size_t thread_id) {
        // need to atomic sum this
        unsigned long long int sequence = atomicAdd(&current_sequence, 1);
        curand_init(seed, sequence, 0, &(random_states[kernel][thread_id]));
    }

    /**
    * @brief Returns a random double from 0..1 for the given kernel for the given thread
    * @param kernel The kernel to which the thread belongs
    * @param thread_id The id of the thread
    * @return The random double from 0..1 for the given kernel for the given thread
    */
    __device__ double get_percentage(RandomKeyHolder kernel, uint64_t thread_id) {
        const auto value = curand_uniform_double(&(random_states[kernel][thread_id]));
        return value;
    }
};

// This is a singleton
class RandomHolder {
    /**
    * Holds the RandomStateData on the GPU
    */

public:
    /**
    * @brief Returns the RandomHolder singleton
    * @return The RandomHolder singleton
    */
    static RandomHolder& get_instance() {
        static RandomHolder instance; 
        return instance;
    }

private:
    RandomHolder();

public:
    RandomHolder(RandomHolder const&) = delete;
    void operator=(RandomHolder const&) = delete;

    /**
    * @brief Returns a pointer to the RandomStateData on the GPU
    * @return A pointer to the RandomStateData on the GPU
    */
    [[nodiscard]] RandomStateData* get_device_pointer() {
        return device_ptr;
    }

    /**
    * @brief Creates the RandomStateData on the GPU and initializes the class, do not call from the outside, needs to be public for device lamdas to work
    */
    void init();

    /**
    * @brief Initializes the random states for the given kernel, given the block and grid size
    * Should be called before the kernel is called for the first time when the number of threads is first known and should be called when the number changes
    * @param kernel The kernel to initialize the radnom states for
    * @param block_size The block size for the kernel
    * @param grid_size The grid size for the kernel
    */
    void init_allocation(RandomKeyHolder kernel, size_t block_size, size_t grid_size);

private:
    RandomStateData* device_ptr;
    gpu::Vector::CudaArrayDeviceHandle<random_state_type> handle_random_states[RandomKeyHolder::COUNT];
};
};