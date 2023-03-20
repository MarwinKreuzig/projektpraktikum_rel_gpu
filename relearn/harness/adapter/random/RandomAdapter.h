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

#include "util/shuffle/shuffle.h"

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <cmath>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

class RandomAdapter {
public:
    template <typename T>
    static T get_random_double(T min, T max, std::mt19937& mt) {
        boost::random::uniform_real_distribution<double> urd(min, max);
        return urd(mt);
    }

    template <typename T>
    static T get_random_integer(T min, T max, std::mt19937& mt) {
        boost::random::uniform_int_distribution<T> uid(min, max);
        return uid(mt);
    }

    template <typename T>
    static T get_random_percentage(std::mt19937& mt) {
        return get_random_double<T>(0.0, std::nextafter(1.0, 1.1), mt);
    }

    static bool get_random_bool(std::mt19937& mt) {
        const auto val = get_random_integer(0, 1, mt);
        return val == 0;
    }

    static std::vector<size_t> get_random_derangement(size_t size, std::mt19937& mt) {
        std::vector<size_t> derangement(size);
        std::iota(derangement.begin(), derangement.end(), 0);

        if(size <= 1) {
            return derangement;
        }

        auto check = [](const std::vector<size_t>& vec) -> bool {
            for (auto i = 0; i < vec.size(); i++) {
                if (i == vec[i]) {
                    return false;
                }
            }
            return true;
        };

        do {
            shuffle(derangement.begin(), derangement.end(), mt);
        } while (!check(derangement));

        return derangement;
    }

    template <typename Iterator>
    static void shuffle(Iterator begin, Iterator end, std::mt19937& mt) {
        detail::shuffle(begin, end, mt);
    }

    template <typename T>
    static std::vector<T> sample(const std::vector<T> vector, size_t sample_size, std::mt19937& mt) {
        std::unordered_set<size_t> indices{};
        while (indices.size() != sample_size) {
            size_t index;
            do {
                index = RandomAdapter::get_random_integer<size_t>(size_t{ 0 }, vector.size() - 1, mt);
            } while (indices.contains(index));
            indices.insert(index);
        }
        std::vector<T> sample{};
        for (const auto index : indices) {
            sample.emplace_back(vector[index]);
        }
        shuffle(sample.begin(), sample.end(), mt);
        return sample;
    }

    template <typename T>
    static std::vector<T> sample(const std::vector<T> vector, std::mt19937& mt) {
        const size_t sample_size = RandomAdapter::get_random_integer<size_t>(size_t{ 0 }, vector.size() - 1, mt);
        return sample(vector, sample_size, mt);
    }
};
