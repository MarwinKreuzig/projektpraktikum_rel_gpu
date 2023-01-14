/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "CalciumCalculator.h"

#include "mpi/MPIWrapper.h"
#include "neurons/NeuronsExtraInfo.h"
#include "util/Timers.h"

void CalciumCalculator::init(const number_neurons_type number_neurons) {
    RelearnException::check(number_neurons > 0, "CalciumCalculator::init: number_neurons was 0");
    RelearnException::check(initial_calcium_initiator.operator bool(), "CalciumCalculator::init: initial_calcium_initiator is empty");
    RelearnException::check(target_calcium_calculator.operator bool(), "CalciumCalculator::init: target_calcium_calculator is empty");

    calcium.resize(number_neurons);
    target_calcium.resize(number_neurons);

    const auto my_rank = MPIWrapper::get_my_rank();

    for (number_neurons_type neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        calcium[neuron_id] = initial_calcium_initiator(my_rank, neuron_id);
        target_calcium[neuron_id] = target_calcium_calculator(my_rank, neuron_id);
    }
}

void CalciumCalculator::create_neurons(const number_neurons_type number_neurons) {
    RelearnException::check(number_neurons > 0, "CalciumCalculator::create_neurons: number_neurons was 0");
    RelearnException::check(initial_calcium_initiator.operator bool(), "CalciumCalculator::create_neurons: initial_calcium_initiator is empty");
    RelearnException::check(target_calcium_calculator.operator bool(), "CalciumCalculator::create_neurons: target_calcium_calculator is empty");

    const auto old_size = calcium.size();
    const auto new_size = old_size + number_neurons;

    calcium.resize(new_size);
    target_calcium.resize(new_size);

    const auto my_rank = MPIWrapper::get_my_rank();

    for (number_neurons_type neuron_id = old_size; neuron_id < new_size; neuron_id++) {
        calcium[neuron_id] = initial_calcium_initiator(my_rank, neuron_id);
        target_calcium[neuron_id] = target_calcium_calculator(my_rank, neuron_id);
    }
}

void CalciumCalculator::update_calcium(const step_type step, const std::span<const FiredStatus> fired_status) {
    const auto info_size = extra_infos->get_size();
    const auto fired_size = fired_status.size();
    const auto calcium_size = calcium.size();
    const auto target_calcium_size = target_calcium.size();

    const auto all_same_size = info_size == fired_size && fired_size == calcium_size && calcium_size == target_calcium_size;
    RelearnException::check(all_same_size, "CalciumCalculator::update_calcium: The vectors had different sizes!");

    Timers::start(TimerRegion::UPDATE_CALCIUM);
    update_current_calcium(fired_status);
    Timers::stop_and_add(TimerRegion::UPDATE_CALCIUM);

    Timers::start(TimerRegion::UPDATE_TARGET_CALCIUM);
    update_target_calcium(step);
    Timers::stop_and_add(TimerRegion::UPDATE_TARGET_CALCIUM);
}

void CalciumCalculator::update_current_calcium(std::span<const FiredStatus> fired_status) noexcept {
    const auto val = (1.0 / static_cast<double>(h));

    const auto disable_flags = extra_infos->get_disable_flags();

#pragma omp parallel for default(none) shared(disable_flags, fired_status, val)
    for (auto neuron_id = 0; neuron_id < calcium.size(); ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        // Update calcium depending on the firing
        auto c = calcium[neuron_id];
        if (fired_status[neuron_id] == FiredStatus::Inactive) {
            for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
                c += val * (-c / tau_C);
            }
        } else {
            for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
                c += val * (-c / tau_C + beta);
            }
        }
        calcium[neuron_id] = c;
    }
}

void CalciumCalculator::update_target_calcium(const step_type step) noexcept {
    if (decay_type == TargetCalciumDecay::None) {
        return;
    }

    if (step < first_decay_step || step > last_decay_step) {
        return;
    }

    if (step % decay_step != 0) {
        return;
    }

    const auto disable_flags = extra_infos->get_disable_flags();

    if (decay_type == TargetCalciumDecay::Absolute) {
#pragma omp parallel for default(none) shared(disable_flags)
        for (auto neuron_id = 0; neuron_id < target_calcium.size(); ++neuron_id) {
            if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
                continue;
            }

            const auto new_target_calcium = target_calcium[neuron_id] - decay_amount;
            target_calcium[neuron_id] = new_target_calcium;
        }
    }

    if (decay_type == TargetCalciumDecay::Relative) {
#pragma omp parallel for default(none) shared(disable_flags)
        for (auto neuron_id = 0; neuron_id < target_calcium.size(); ++neuron_id) {
            if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
                continue;
            }

            const auto new_target_calcium = target_calcium[neuron_id] * decay_amount;
            target_calcium[neuron_id] = new_target_calcium;
        }
    }
}
