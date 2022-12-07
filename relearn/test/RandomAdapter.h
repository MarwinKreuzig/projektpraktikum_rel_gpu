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

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <cmath>
#include <random>

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
};
