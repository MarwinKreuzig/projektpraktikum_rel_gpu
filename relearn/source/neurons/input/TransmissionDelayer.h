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
#include "util/Random.h"

#include <deque>
#include <vector>

enum TransmissionDelayType {
    Constant,
    Random
};
/**
 * @brief Pretty-prints the transmission delayer type to the chosen stream
 * @param out The stream to which to print the transmission delayer
 * @param element_type The transmission delayer to print
 * @return The argument out, now altered with the transmission delayer
 */
inline std::ostream& operator<<(std::ostream& out, const TransmissionDelayType& transmission_delayer_type) {
    if (transmission_delayer_type == TransmissionDelayType::Constant) {
        return out << "Constant";
    }

    if (transmission_delayer_type == TransmissionDelayType::Random) {
        return out << "Random";
    }

    return out;
}
template <>
struct fmt::formatter<TransmissionDelayType> : ostream_formatter { };

/**
 * This class adds a delay between the fire of a source neuron and the synaptic input on the target neuron
 */
class TransmissionDelayer {
public:
    virtual ~TransmissionDelayer() = default;

    TransmissionDelayer() {
    }


    /**
     * Registers that a source neuron fired in the current step and its transmission target
     * @param target_neuron The target of a single synapse connected to the source neuron
     * @param source_neuron The neuron that fired
     * @param edge_val The weight of the synapse
     * @param num_neurons The number of local neurons on this mpi rank
     */
    void register_fired_input(const NeuronID &target_neuron, const RankNeuronId &source_neuron,
                              RelearnTypes::static_synapse_weight edge_val, RelearnTypes::number_neurons_type num_neurons) {
        auto delay = get_delay(target_neuron, source_neuron);

        while (saved_fire_states.size() < delay + 1) {
            std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>> empty_fire_states;
            empty_fire_states.resize(num_neurons, {});
            saved_fire_states.push_back(empty_fire_states);
        }
        saved_fire_states[delay][target_neuron.get_neuron_id()].emplace_back(source_neuron, edge_val);
    }

    /**
     * Call this method after creating new neurons to adapt its fields
     * @param new_size The new number of local neurons
     */
    void create_neurons(RelearnTypes::number_neurons_type new_size) {
        for(auto& fire_states : saved_fire_states) {
            fire_states.resize(new_size, {});
        }
    }

    /**
     * Returns true, if there are currently registered delayed inputs for target neurons
     * @return true, if there are delayed inputs
     */
    [[nodiscard]] bool has_delayed_inputs() const {
        return !saved_fire_states.empty();
    }

    /**
     * Returns the delayed inputs scheduled for the current step and a single target neuron
     * @param target_neuron The target neuron for which we check the delayed inputs
     * @return Delayed inputs in the current step for the target_neuron
     */
    [[nodiscard]] const std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> &
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

    /**
     * Prepares the transmission delayer for a new simulation step. Call this method before register_fired_input and get_delayed_inputs for a new step
     */
    void prepare_update(RelearnTypes::number_neurons_type num_neurons) {
        if (!saved_fire_states.empty()) {
            saved_fire_states.pop_front();
        }
        if(saved_fire_states.empty()) {
            std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>> empty_fire_states;
            empty_fire_states.resize(num_neurons, {});
            saved_fire_states.push_back(empty_fire_states);
        }
    }


protected:

    /**
     * Returns the delay for a neuron pair in the current step
     * @param target_neuron The neuron that receive the firing
     * @param source_neuron The neuron that fires
     * @return Delay between both neurons
     */
    virtual RelearnTypes::step_type get_delay(const NeuronID &target_neuron, const RankNeuronId &source_neuron) = 0;

private:
    std::deque<std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>>> saved_fire_states{};

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


    RelearnTypes::step_type get_delay(const NeuronID &target_neuron, const RankNeuronId &source_neuron) override {
        return delay_steps;
    };

    std::unique_ptr<TransmissionDelayer> clone() override {
        return std::make_unique<ConstantTransmissionDelayer>(delay_steps);
    }

private:
    RelearnTypes::step_type delay_steps;
};


/**
 * Transmission delayer delays for a fixed number of steps
 */
class RandomizedTransmissionDelayer : public TransmissionDelayer {
public:

    /**
     * Constructor
     * @param delay_steps Number of steps that the fire is delayed
     */
    RandomizedTransmissionDelayer(double mean, double stddev) : TransmissionDelayer(), mean(mean), stddev(stddev) {
        RelearnException::check(stddev > 0, "TransmissionDelayer::TransmissionDelayer: standard deviation must be >= 0");
    }

    RelearnTypes::step_type get_delay(const NeuronID &target_neuron, const RankNeuronId &source_neuron) override {
        const auto d = RandomHolder::get_random_normal_double(RandomHolderKey::TransmissionDelay, mean, stddev);
        if(d<0) {
            return 0.0;
        }
        return static_cast<RelearnTypes::step_type>(d);
    };

    std::unique_ptr<TransmissionDelayer> clone() override {
        return std::make_unique<RandomizedTransmissionDelayer>(mean, stddev);
    }

private:
    double mean;
    double stddev;
};