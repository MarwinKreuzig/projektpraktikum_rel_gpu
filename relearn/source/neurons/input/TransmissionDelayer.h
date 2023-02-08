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

#include <deque>
#include <vector>

/**
 * This class adds a delay between the fire of a source neuron and the synaptic input on the target neuron
 */
class TransmissionDelayer {
private:
private:
    std::deque<std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::synapse_weight>>>> saved_fire_states{};

public:
    virtual ~TransmissionDelayer() = default;

    TransmissionDelayer() {
    }


    /**
     * Takes the fire status from the current step and returns the delayed fire status from the specified steps earlier
     * @param fired_in_current_step The current fire status from the current step
     * @return Delayed fire status
     */
    void register_fired_input(const NeuronID &target_neuron, const RankNeuronId &source_neuron,
                              RelearnTypes::synapse_weight edge_val, RelearnTypes::number_neurons_type num_neurons) {
        auto delay = get_delay(target_neuron, source_neuron);

        while (saved_fire_states.size() < delay + 1) {
            std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::synapse_weight>>> empty_fire_states;
            empty_fire_states.resize(num_neurons, {});
            saved_fire_states.push_back(empty_fire_states);
        }
        saved_fire_states[delay][target_neuron.get_neuron_id()].emplace_back(source_neuron, edge_val);
    }

    void create_neurons(RelearnTypes::number_neurons_type new_size) {
        for(auto& fire_states : saved_fire_states) {
            fire_states.resize(new_size, {});
        }
    }

    [[nodiscard]] bool has_delayed_inputs() const {
        return !saved_fire_states.empty();
    }

    [[nodiscard]] const std::vector<std::pair<RankNeuronId, RelearnTypes::synapse_weight>> &
    get_delayed_inputs(const NeuronID &target_neuron) const {
        RelearnException::check(!saved_fire_states.empty(),
                                "TransformationDelayer::get_delayed_input: No delayed inputs stored");
        const auto &fired_inputs = saved_fire_states.front();
        RelearnException::check(target_neuron.get_neuron_id() < fired_inputs.size(),
                                "TransformationDelayer::get_delayed_input: Neuron id is too large");

        return fired_inputs[target_neuron.get_neuron_id()];
    }

    /**
     * Clones this object
     * @return New instance of an unique_ptr to a new object with the same parameters
     */
    virtual std::unique_ptr<TransmissionDelayer> clone() = 0;

    void prepare_update() {
        const auto size = saved_fire_states.size();
        if (!saved_fire_states.empty()) {
            saved_fire_states.pop_front();
        }
    }


protected:

    virtual RelearnTypes::step_type get_delay(const NeuronID &target_neuron, const RankNeuronId &source_neuron) = 0;

};

/**
 * Transmission delayer delays for a fixed number of steps
 */
class ConstantTransmissionDelayer : public TransmissionDelayer {
public:

    /**
     * Constructor
     * @param delay_steps Number of steps that the fire is delayed
     */
    ConstantTransmissionDelayer(RelearnTypes::step_type delay_steps) : TransmissionDelayer(), delay_steps(delay_steps) {
        RelearnException::check(delay_steps >= 0, "TransmissionDelayer::TransmissionDelayer: delay_steps must be >= 0");
    }


    /**
     * Takes the fire status from the current step and returns the delayed fire status from the specified steps earlier
     * @param fired_in_current_step The current fire status from the current step
     * @return Delayed fire status
     */
    RelearnTypes::step_type get_delay(const NeuronID &target_neuron, const RankNeuronId &source_neuron) override {
        return delay_steps;
    };

    /**
     * Clones this object
     * @return New instance of an unique_ptr to a new object with the same parameters
     */
    std::unique_ptr<TransmissionDelayer> clone() override {
        return std::make_unique<ConstantTransmissionDelayer>(delay_steps);
    }

private:
    RelearnTypes::step_type delay_steps;
};
//
//class RandomizedTransmissionDelayer : public TransmissionDelayer {
//public:
//
//    /**
//     * Constructor
//     * @param delay_steps Number of steps that the fire is delayed
//     */
//    RandomizedTransmissionDelayer(double mean, double stddev) : TransmissionDelayer(),mean(mean), stddev(stddev) {
//        RelearnException::check(stddev>0.0, "TransmissionDelayer::TransmissionDelayer: standard deviation must be >= 0");
//    }
//
//
//    /**
//     * Takes the fire status from the current step and returns the delayed fire status from the specified steps earlier
//     * @param fired_in_current_step The current fire status from the current step
//     * @return Delayed fire status
//     */
//    std::vector<FiredStatus> apply_delay(const std::vector<FiredStatus>& fired_in_current_step) override {
//        RelearnException::check(!fired_in_current_step.empty(), "TransmissionDelayer::apply_delay: FireStatus vector must be not empty");
//
//
//    }
//
//    /**
//     * Clones this object
//     * @return New instance of an unique_ptr to a new object with the same parameters
//     */
//    std::unique_ptr<TransmissionDelayer> clone() override {
//        return std::make_unique<RandomizedTransmissionDelayer>(mean, stddev);
//    }
//
//private:
//    double mean;
//    double stddev;
//    std::vector<std::vector<FiredStatus>> saved_fire_states;
//
//};