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

#include "MPIWrapper.h"
#include "Random.h"
#include "RelearnException.h"

#include <list>
#include <memory>
#include <random>

class OctreeNode;

/**
* Type for list elements used to create probability subinterval
*/
struct ProbabilitySubinterval {
    OctreeNode* ptr{ nullptr };
    double probability{ 0.0 };
    MPIWrapper::AsyncToken mpi_request{ MPIWrapper::get_null_request() };
    int request_rank{ -1 };

public:
    explicit ProbabilitySubinterval(OctreeNode* node) noexcept
        : ptr(node) {
    }

    void set_probability(double prob) noexcept {
        probability = prob;
    }

    void set_mpi_request(MPIWrapper::AsyncToken request) noexcept {
        mpi_request = request;
    }

    void set_request_rank(int rank) {
        RelearnException::check(rank >= 0, "ProbabilitySubinterval, rank is smaller zan zero");
        request_rank = rank;
    }

    [[nodiscard]] OctreeNode* get_ptr() const noexcept {
        return ptr;
    }

    [[nodiscard]] double get_probability() const noexcept {
        return probability;
    }

    [[nodiscard]] MPIWrapper::AsyncToken get_mpi_request() const noexcept {
        return mpi_request;
    }

    [[nodiscard]] int get_request_rank() const noexcept {
        return request_rank;
    }

    /**
	     * Randomly select node from probability interval
	     */
    [[nodiscard]] static OctreeNode* select_subinterval(const std::list<std::shared_ptr<ProbabilitySubinterval>>& ps_list, double probabilty = 1.0) {
        if (ps_list.empty()) {
            return nullptr;
        }

        std::uniform_real_distribution<double> random_number_distribution(0.0, std::nextafter(probabilty, probabilty + 1.0));
        std::mt19937& random_number_generator = RandomHolder::get_instance().get_random_generator(RandomHolder::OCTREE);

        // Draw random number from [0,1]
        const double random_number = random_number_distribution(random_number_generator);

        /**
	        * Also check for it != list.end() to account for that, due to numeric inaccuracies in summation,
	        * it might happen that random_number > sum_probabilities in the end
	        */
        auto it = ps_list.cbegin();
        double sum_probabilities = (*it)->get_probability();
        it++; // Point to second element
        while (random_number > sum_probabilities && it != ps_list.cend()) {
            sum_probabilities += (*it)->get_probability();
            it++;
        }
        it--; // Undo it++ before or in loop to get correct subinterval
        return (*it)->get_ptr();
    }
};
using ProbabilitySubintervalList = std::list<std::shared_ptr<ProbabilitySubinterval>>;
