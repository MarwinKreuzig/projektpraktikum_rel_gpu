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

#include "ProbabilitySubinterval.h"
#include "SignalType.h"
#include "Vec3.h"

#include <list>
#include <memory>

/**
* Type for vacant axon for which a target neuron needs to be found
*/
class VacantAxon {
public:
    VacantAxon(size_t neuron_id, const Vec3d& pos, SignalType dendrite_type_needed)
        : neuron_id(neuron_id)
        , xyz_pos(pos)
        , dendrite_type_needed(dendrite_type_needed) {
    }

    [[nodiscard]] size_t get_neuron_id() const noexcept {
        return neuron_id;
    }

    [[nodiscard]] const Vec3d& get_xyz_pos() const noexcept {
        return xyz_pos;
    }

    [[nodiscard]] SignalType get_dendrite_type_needed() const noexcept {
        return dendrite_type_needed;
    }

    void add_to_accepted(const std::shared_ptr<ProbabilitySubinterval>& subinterval) {
        nodes_accepted.emplace_back(subinterval);
    }

    void add_to_visit(const std::shared_ptr<ProbabilitySubinterval>& subinterval) {
        nodes_to_visit.emplace_back(subinterval);
    }

    void add_to_accepted(ProbabilitySubintervalVector&& list) {
        nodes_accepted.insert(nodes_accepted.cend(), list.begin(), list.end());
    }

    void add_to_visit(ProbabilitySubintervalVector&& list) {
        nodes_to_visit.insert(nodes_to_visit.cend(), list.begin(), list.end());
    }

    void empty_accepted() noexcept {
        nodes_accepted.clear();
    }

    void empty_visited() noexcept {
        nodes_to_visit.clear();
    }

    [[nodiscard]] size_t get_num_to_visit() const noexcept {
        return nodes_to_visit.size();
    }

    [[nodiscard]] std::shared_ptr<ProbabilitySubinterval> get_first_to_visit() const noexcept {
        return nodes_to_visit.front();
    }

    void remove_first_visit() noexcept {
        nodes_to_visit.erase(nodes_to_visit.cbegin());
    }

    [[nodiscard]] const ProbabilitySubintervalVector& get_nodes_accepted() const noexcept {
        return nodes_accepted;
    }

private:
    size_t neuron_id;
    Vec3d xyz_pos;
    SignalType dendrite_type_needed;

    ProbabilitySubintervalVector nodes_to_visit;
    ProbabilitySubintervalVector nodes_accepted;
};
using VacantAxonList = std::list<std::shared_ptr<VacantAxon>>;
