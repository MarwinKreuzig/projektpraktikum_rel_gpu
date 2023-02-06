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

#include "neurons/enums/FiredStatus.h"

#include <queue>
#include <vector>

/**
 * This class adds a delay between the fire of a source neuron and the synaptic input on the target neuron
 */
class TransmissionDelayer {
public:

    /**
     * Constructor
     * @param delay_steps Number of steps that the fire is delayed
     */
    TransmissionDelayer(RelearnTypes::step_type delay_steps) : delay_steps(delay_steps) {
        RelearnException::check(delay_steps>=0, "TransmissionDelayer::TransmissionDelayer: delay_steps must be >= 0");
    }

    /**
     * Constructor without delay
     */
    TransmissionDelayer() : delay_steps(0) {
    }

    /**
     * Takes the fire status from the current step and returns the delayed fire status from the specified steps earlier
     * @param fired_in_current_step The current fire status from the current step
     * @return Delayed fire status
     */
    std::vector<FiredStatus> apply_delay(const std::vector<FiredStatus>& fired_in_current_step) {
        RelearnException::check(!fired_in_current_step.empty(), "TransmissionDelayer::apply_delay: FireStatus vector must be not empty");
        if(delay_steps == 0) {
            return fired_in_current_step;
        }
        saved_fire_states.push(fired_in_current_step);

        if(saved_fire_states.size() == delay_steps + 1) {
            std::vector<FiredStatus> delayed_fire_state = saved_fire_states.front();
            saved_fire_states.pop();
            return delayed_fire_state;
        }
        std::vector<FiredStatus> default_value;
        default_value.resize(fired_in_current_step.size(), FiredStatus::Inactive);
        return default_value;
    }

    /**
     * Clones this object
     * @return New instance of an unique_ptr to a new object with the same parameters
     */
    std::unique_ptr<TransmissionDelayer> clone() {
        return std::make_unique<TransmissionDelayer>(delay_steps);
    }

private:
    RelearnTypes::step_type delay_steps;
    std::queue<std::vector<FiredStatus>> saved_fire_states;

};