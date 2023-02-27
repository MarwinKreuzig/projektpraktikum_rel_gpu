/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FiredStatusApproximator.h"

void FiredStatusApproximator::set_local_fired_status(const step_type step, const std::span<const FiredStatus> fired_status) {
    RelearnException::check(accumulated_fired.size() == fired_status.size(), "FiredStatusApproximator::set_local_fired_status: Mismatching sizes: {} vs. {}", accumulated_fired.size(), fired_status.size());

    for (auto i = size_t(0); i < fired_status.size(); i++) {
        if (fired_status[i] == FiredStatus::Fired) {
            accumulated_fired[i]++;
        }
    }
}

void FiredStatusApproximator::exchange_fired_status(const step_type step) {
    // This is a no-op
}

bool FiredStatusApproximator::contains(const MPIRank rank, const NeuronID neuron_id) const {
    return false;
}

void FiredStatusApproximator::notify_of_plasticity_change(const step_type step) {

}
