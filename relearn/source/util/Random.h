/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include "../Config.h"
#include "../util/RelearnException.h"

#include <algorithm>
#include <map>
#include <random>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_thread_num() { return 0; }
#endif

/**
 * This enum allows a type safe differentiation between the types that require access to random numbers.
 * Can be extended without requiring an extension at a different place.
 */
enum class RandomHolderKey : char {
    Algorithm = 0,
    Partition = 1,
    SubdomainFromNeuronDensity = 2,
    PoissonModel = 3,
    Neurons = 4,
    NeuronModel = 5,
    SynapticElements = 6,
    NeuronsExtraInformation = 7,
};

/**
 * This type provides a static thread-safe interface for generating random numbers.
 * Each instance of RandomHolderKey and each thread in an OMP parallel region has its own random number generator.
 */
class RandomHolder {
    RandomHolder() = default;

    thread_local static inline std::map<RandomHolderKey, std::mt19937> random_number_generators{};

public:
    /**
     * @brief Generates a random double (uniformly distributed in [lower_inclusive, upper_exclusive)).
     *      Uses the RNG that is associated with the key.
     * @param key The type which's RNG shall be used
     * @param lower_inclusive The lower inclusive bound for the random double
     * @param upper_exclusive The upper exclusive bound for the random double
     * @exception Throws a RelearnException if lower_inclusive >= upper_exclusive
     * @return A uniformly distributed double in [lower_inclusive, upper_exclusive)
     */
    static double get_random_uniform_double(RandomHolderKey key, double lower_inclusive, double upper_exclusive) {
        RelearnException::check(lower_inclusive < upper_exclusive, "Random number from invalid interval");
        std::uniform_real_distribution<double> urd(lower_inclusive, upper_exclusive);

        return urd(random_number_generators[key]);
    }

    /**
     * @brief Generates a random double (normally distributed in with specified mean and standard deviation).
     *      Uses the RNG that is associated with the key.
     * @param key The type which's RNG shall be used
     * @param mean The mean of the normal distribution
     * @param stddev The standard deviation of the normal distribution
     * @exception Throws a RelearnException if stddev <= 0.0
     * @return A normally distributed double with specified mean and standard deviation
     */
    static double get_random_normal_double(RandomHolderKey key, double mean, double stddev) {
        RelearnException::check(0.0 < stddev, "Random number from invalid deviation");
        std::normal_distribution<double> nd(mean, stddev);

        return nd(random_number_generators[key]);
    }

    /**
     * @brief Shuffles all values in [begin, end) such that all permutations have equal probability.
     *      Uses the RNG that is associated with the key. There should be a natural number n st. begin + n = end.
     * @param key The type which's RNG shall be used
     * @param begin The iterator that marks the inclusive begin
     * @param end the iterator that marks the exclusive end
     * @tparam IteratorType The iterator type that is used to iterate the elements. Should be 'nice'.
     */
    template <typename IteratorType>
    static void shuffle(RandomHolderKey key, IteratorType begin, IteratorType end) {
        std::shuffle(begin, end, random_number_generators[key]);
    }

    /**
     * @brief Fills all values in [begin, end) with uniformly distributed doubles from [lower_inclusive, upper_exclusive).
     *      Uses the RNG that is associated with the key. There should be a natural number n st. begin + n = end.
     * @param key The type which's RNG shall be used 
     * @param begin The iterator that marks the inclusive begin
     * @param end the iterator that marks the exclusive end
     * @param lower_inclusive The lower inclusive bound for the random doubles
     * @param upper_exclusive The upper exclusive bound for the random doubles
     * @tparam IteratorType The iterator type that is used to iterate the elements. Should be 'nice'.
     * @exception Throws a RelearnException if lower_inclusive >= upper_exclusive.
     */
    template <typename IteratorType>
    static void fill(RandomHolderKey key, IteratorType begin, IteratorType end, double lower_inclusive, double upper_exclusive) {
        RelearnException::check(lower_inclusive < upper_exclusive, "Random number from invalid interval");
        std::uniform_real_distribution<double> urd(lower_inclusive, upper_exclusive);
        auto& gen = random_number_generators[key];

        for (; begin != end; begin++) {
            *begin = urd(gen);
        }
    }

    /**
     * @brief Seeds the random number generators associated with the key.
     *      The seed used is seed + omp_get_thread_num()
     * @param key The type which's RNG shall be seeded 
     * @param seed The base seed that should be used
     */
    static void seed(RandomHolderKey key, unsigned int seed) {
        // NOLINTNEXTLINE
#pragma omp parallel shared(key, seed)
        {
            const auto thread_id = omp_get_thread_num();
            random_number_generators[key].seed(seed + thread_id);
        }
    }
};
