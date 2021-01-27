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
#include <vector>

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
};
using ProbabilitySubintervalVector = std::vector<std::shared_ptr<ProbabilitySubinterval>>;
