//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This header includes the implementation of the shuffle function from libcxx of the LLVM Project.
// The original source of the algorithm is in `libcxx/include/__algorithm/shuffle.h` and was modified to
// use boost's `uniform_int_distribution` instead of the standard one, to provide portability.

#pragma once

#include <boost/random/uniform_int_distribution.hpp>

namespace detail {
/**
 * @brief Implementation of std::shuffle with a portable random number distribution
 *
 * Original source: libcxx from the LLVM Project
 * Modified: names, formatting, variable scopes, random distribution
 *
 * @tparam RandomAccessIterator iterator type for the range
 * @tparam UniformRandomNumberGenerator type of the random number generator
 * @param first begin iterator
 * @param last end iterator
 * @param gen random number generator
 */
template <class RandomAccessIterator, class UniformRandomNumberGenerator>
void shuffle(RandomAccessIterator first, RandomAccessIterator last,
    UniformRandomNumberGenerator&& gen) {
    using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
    using distribution_type = boost::random::uniform_int_distribution<ptrdiff_t>;
    using distribution_param_type = typename distribution_type::param_type;
    using std::swap;

    if (difference_type distance = last - first; distance > 1) {
        distribution_type uid;
        for (--last, --distance; first < last; ++first, --distance) { // NOLINT(hicpp-use-nullptr,modernize-use-nullptr)
            if (difference_type random_val = uid(gen, distribution_param_type(0, distance));
                random_val != difference_type(0)) {
                swap(*first, *(first + random_val));
            }
        }
    }
}
} // namespace detail
