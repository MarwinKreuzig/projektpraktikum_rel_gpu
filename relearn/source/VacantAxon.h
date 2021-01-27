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

#include "Cell.h"
#include "ProbabilitySubinterval.h"
#include "Vec3.h"

#include <list>
#include <memory>

/**
* Type for vacant axon for which a target neuron needs to be found
*/
struct VacantAxon {
    VacantAxon(size_t neuron_id, const Vec3d& pos, Cell::DendriteType dendrite_type_needed)
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

    [[nodiscard]] Cell::DendriteType get_dendrite_type_needed() const noexcept {
        return dendrite_type_needed;
    }

    void add_to_accepted(const std::shared_ptr<ProbabilitySubinterval>& subinterval) {
        nodes_accepted.emplace_back(subinterval);
    }

    void add_to_visit(const std::shared_ptr<ProbabilitySubinterval>& subinterval) {
        nodes_to_visit.emplace_back(subinterval);
    }

    void add_to_accepted(ProbabilitySubintervalList&& list) {
        nodes_accepted.splice(nodes_accepted.end(), std::move(list));
    }

    void add_to_visit(ProbabilitySubintervalList&& list) {
        nodes_to_visit.splice(nodes_to_visit.end(), std::move(list));
    }

    void empty_accepted() noexcept {
        nodes_accepted.clear();
    }

    void empty_visited() noexcept {
        nodes_to_visit.clear();
    }

    [[nodiscard]] OctreeNode* select_subinterval(double probabilty = 1.0) const {
        return ProbabilitySubinterval::select_subinterval(nodes_accepted, probabilty);
    }

    [[nodiscard]] size_t get_num_to_visit() const noexcept {
        return nodes_to_visit.size();
    }

    [[nodiscard]] std::shared_ptr<ProbabilitySubinterval> get_first_to_visit() const noexcept {
        return nodes_to_visit.front();
    }

    void remove_first_visit() noexcept {
        nodes_to_visit.pop_front();
    }

    ProbabilitySubintervalList nodes_accepted;

    size_t neuron_id;
    Vec3d xyz_pos;
    Cell::DendriteType dendrite_type_needed;

    ProbabilitySubintervalList nodes_to_visit;
};
using VacantAxonList = std::list<std::shared_ptr<VacantAxon>>;
