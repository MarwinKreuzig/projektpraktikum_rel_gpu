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

#include "Config.h"
#include "util/RelearnException.h"
#include "util/shuffle/shuffle.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <array>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_thread_num() { return 0; }
#endif

template <typename T>
using uniform_int_distribution = boost::random::uniform_int_distribution<T>;
template <typename T>
using uniform_real_distribution = boost::random::uniform_real_distribution<T>;
template <typename T>
using normal_distribution = boost::random::normal_distribution<T>;

using mt19937 = std::mt19937;

/**
 * This enum allows a type safe differentiation between the types that require access to random numbers.
 */
enum class RandomHolderKey : char {
    Algorithm = 0,
    Partition = 1,
    Subdomain = 2,
    PoissonModel = 3,
    Neurons = 4,
    NeuronModel = 5,
    SynapticElements = 6,
    NeuronsExtraInformation = 7,
    Connector = 8,
    BackgroundActivity = 9,
};

constexpr size_t NUMBER_RANDOM_HOLDER_KEYS = 10;

/**
 * This type provides a static thread-safe interface for generating random numbers.
 * Each instance of RandomHolderKey and each thread in an OMP parallel region has its own random number generator.
 */
class RandomHolder {
public:
    /**
     * @brief Generates a random double (uniformly distributed in [lower_inclusive, upper_exclusive)).
     *      Uses the RNG that is associated with the key.
     * @param key The type whose RNG shall be used
     * @param lower_inclusive The lower inclusive bound for the random double
     * @param upper_exclusive The upper exclusive bound for the random double
     * @exception Throws a RelearnException if lower_inclusive >= upper_exclusive
     * @return A uniformly distributed double in [lower_inclusive, upper_exclusive)
     */
    static double get_random_uniform_double(const RandomHolderKey key, const double lower_inclusive, const double upper_exclusive) {
        RelearnException::check(lower_inclusive < upper_exclusive,
            "RandomHolder::get_random_uniform_double: Random number from invalid interval [{}, {}) for key {}", lower_inclusive, upper_exclusive, static_cast<int>(key));
        uniform_real_distribution<double> dist(lower_inclusive, upper_exclusive);

        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        auto& generator = random_number_generators[static_cast<int>(key)];
        return dist(generator);
    }

    /**
     * @brief Generates a random integer (uniformly distributed in [lower_inclusive, upper_inclusive]).
     *      Uses the RNG that is associated with the key.
     * @param key The type whose RNG shall be used
     * @param lower_inclusive The lower inclusive bound for the random integer
     * @param upper_inclusive The upper inclusive bound for the random integer
     * @exception Throws a RelearnException if lower_inclusive > upper_inclusive
     * @return A uniformly integer double in [lower_inclusive, upper_inclusive)
     */
    template <typename integer_type>
    static integer_type get_random_uniform_integer(const RandomHolderKey key, const integer_type lower_inclusive, const integer_type upper_inclusive) {
        RelearnException::check(lower_inclusive <= upper_inclusive,
            "RandomHolder::get_random_uniform_integer: Random number from invalid interval [{}, {}] for key {}", lower_inclusive, upper_inclusive, static_cast<int>(key));
        uniform_int_distribution<integer_type> uid(lower_inclusive, upper_inclusive);

        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        auto& generator = random_number_generators[static_cast<int>(key)];
        return uid(generator);
    }

    /**
     * @brief Generates a random double (normally distributed in with specified mean and standard deviation).
     *      Uses the RNG that is associated with the key.
     * @param key The type whose RNG shall be used
     * @param mean The mean of the normal distribution
     * @param stddev The standard deviation of the normal distribution
     * @exception Throws a RelearnException if stddev <= 0.0
     * @return A normally distributed double with specified mean and standard deviation
     */
    static double get_random_normal_double(const RandomHolderKey key, const double mean, const double stddev) {
        RelearnException::check(0.0 < stddev, "RandomHolder::get_random_normal_double: Random number with invalid standard deviation {} for key {}", stddev, static_cast<int>(key));
        normal_distribution<double> nd(mean, stddev);

        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        auto& generator = random_number_generators[static_cast<int>(key)];
        return nd(generator);
    }

    /**
     * @brief Shuffles all values in [begin, end) such that all permutations have equal probability.
     *      Uses the RNG that is associated with the key. There should be a natural number n st. begin + n = end.
     * @param key The type whose RNG shall be used
     * @param begin The iterator that marks the inclusive begin
     * @param end the iterator that marks the exclusive end
     * @tparam IteratorType The iterator type that is used to iterate the elements. Should be 'nice'.
     */
    template <typename IteratorType>
    static void shuffle(const RandomHolderKey key, const IteratorType begin, const IteratorType end) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        auto& generator = random_number_generators[static_cast<int>(key)];
        detail::shuffle(begin, end, generator);
    }

    /**
     * @brief Fills all values in [begin, end) with uniformly distributed doubles from [lower_inclusive, upper_exclusive).
     *      Uses the RNG that is associated with the key. There should be a natural number n st. begin + n = end.
     * @param key The type whose RNG shall be used
     * @param begin The iterator that marks the inclusive begin
     * @param end the iterator that marks the exclusive end
     * @param lower_inclusive The lower inclusive bound for the random doubles
     * @param upper_exclusive The upper exclusive bound for the random doubles
     * @tparam IteratorType The iterator type that is used to iterate the elements. Should be 'nice'.
     * @exception Throws a RelearnException if lower_inclusive >= upper_exclusive.
     */
    template <typename IteratorType>
    static void fill(const RandomHolderKey key, const IteratorType begin, const IteratorType end, const double lower_inclusive, const double upper_exclusive) {
        RelearnException::check(lower_inclusive < upper_exclusive, "RandomHolder::fill: Random number from invalid interval [{}, {}) for key {}", lower_inclusive, upper_exclusive, static_cast<int>(key));
        uniform_real_distribution<double> urd(lower_inclusive, upper_exclusive);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        auto& generator = random_number_generators[static_cast<int>(key)];

        for (auto it = begin; it != end; it++) {
            *it = urd(generator);
        }
    }

    /**
     * @brief Seeds the random number generators associated with the key.
     *      The seed used is seed + omp_get_thread_num()
     * @param key The type whose RNG shall be seeded
     * @param seed The base seed that should be used
     */
    static void seed(const RandomHolderKey key, const unsigned int seed) {
        // NOLINTNEXTLINE
#pragma omp parallel shared(key, seed)
        {
            const auto thread_id = omp_get_thread_num();
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            auto& generator = random_number_generators[static_cast<int>(key)];
            generator.seed(seed + thread_id);
        }
    }

private:
    RandomHolder() = default;

    thread_local static inline std::array<mt19937, NUMBER_RANDOM_HOLDER_KEYS> random_number_generators{};
};
