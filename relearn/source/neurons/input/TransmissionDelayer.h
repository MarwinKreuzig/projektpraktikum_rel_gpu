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

#include "enums/FiredStatus.h"
#include "util/Random.h"
#include "util/Timers.h"

#include <deque>
#include <vector>

enum TransmissionDelayType {
    Constant,
    Random,
    None
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

    TransmissionDelayer(RelearnTypes::step_type max_delay)
        : max_delay(max_delay) {
    }

    using fired_pair_type = std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>;
    using fired_neurons_type = std::vector<fired_pair_type>;

    void init(RelearnTypes::number_neurons_type num_neurons) {
        data = std::vector<std::vector<std::pair<size_t, std::vector<fired_pair_type>>>>(max_delay + 1, std::vector<std::pair<size_t, std::vector<fired_pair_type>>>(num_neurons, std::make_pair<size_t, std::vector<fired_pair_type>>(0, std::vector<fired_pair_type>{ max_num_fired_neurons })));
        initialized = true;
    }

    /**
     * Registers that a source neuron fired in the current step and its transmission target
     * @param target_neuron The target of a single synapse connected to the source neuron
     * @param source_neuron The neuron that fired
     * @param edge_val The weight of the synapse
     * @param num_neurons The number of local neurons on this mpi rank
     */
    void register_fired_input(const NeuronID& target_neuron, const RankNeuronId& source_neuron,
        RelearnTypes::static_synapse_weight edge_val, RelearnTypes::number_neurons_type num_neurons) {
        RelearnException::check(initialized, "TransmissionDelayer::register_fired_input: Not initialized");
        auto delay = get_delay(target_neuron, source_neuron);

        auto& ref_pair = data[(cur_index + delay) % data.size()][target_neuron.get_neuron_id()];
        auto& ref = ref_pair.second;

        auto i = ref_pair.first;
        if (i < max_num_fired_neurons) {
            ref[i] = std::make_pair(source_neuron, edge_val);
        } else {
            ref.emplace_back(source_neuron, edge_val);
        }
        ref_pair.first = i + 1;
    }

    /**
     * Call this method after creating new neurons to adapt its fields
     * @param new_size The new number of local neurons
     */
    void create_neurons(RelearnTypes::number_neurons_type new_size) {
        RelearnException::check(initialized, "TransmissionDelayer::register_fired_input: Not initialized");
        for (auto& fire_states : data) {
            fire_states.resize(new_size, std::make_pair<size_t, std::vector<fired_pair_type>>(0, std::vector<fired_pair_type>{ max_num_fired_neurons }));
        }
    }

    /**
     * Returns true, if there are currently registered delayed inputs for target neurons
     * @return true, if there are delayed inputs
     */
    [[nodiscard]] bool has_delayed_inputs() const {
        return true;
    }

    /**
     * Returns the delayed inputs scheduled for the current step and a single target neuron
     * @param target_neuron The target neuron for which we check the delayed inputs
     * @return Delayed inputs in the current step for the target_neuron
     */
    [[nodiscard]] const fired_neurons_type get_delayed_inputs(const NeuronID& target_neuron) const {
        const auto& [last_valid, vec] = get_delayed_inputs_efficient(target_neuron);
        return { vec.begin(), vec.begin() + last_valid };
    }

    /**
     * Returns the delayed inputs scheduled for the current step and a single target neuron
     * @param target_neuron The target neuron for which we check the delayed inputs
     * @return Delayed inputs in the current step for the target_neuron
     */
    [[nodiscard]] const std::pair<size_t, fired_neurons_type>& get_delayed_inputs_efficient(const NeuronID& target_neuron) const {
        RelearnException::check(initialized, "TransmissionDelayer::register_fired_input: Not initialized");
        const auto& fired_inputs = data[cur_index];
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
        RelearnException::check(initialized, "TransmissionDelayer::register_fired_input: Not initialized");
        Timers::start(TimerRegion::CALC_PREPARE_TRANSMISSION);
        std::fill(data[cur_index].begin(), data[cur_index].end(), std::make_pair<size_t, std::vector<fired_pair_type>>(0, std::vector<fired_pair_type>{ max_num_fired_neurons }));
        cur_index = (cur_index + 1) % data.size();
        Timers::stop_and_add(TimerRegion::CALC_PREPARE_TRANSMISSION);
    }

protected:
    /**
     * Returns the delay for a neuron pair in the current step
     * @param target_neuron The neuron that receive the firing
     * @param source_neuron The neuron that fires
     * @return Delay between both neurons
     */
    virtual RelearnTypes::step_type get_delay(const NeuronID& target_neuron, const RankNeuronId& source_neuron) = 0;

private:
    size_t max_delay{ 0 };

    size_t cur_index{ 0 };

    std::vector<std::vector<std::pair<size_t, std::vector<fired_pair_type>>>> data;

    constexpr static size_t max_num_fired_neurons = 50;

    bool initialized{ false };
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
    ConstantTransmissionDelayer(RelearnTypes::step_type delay_steps)
        : TransmissionDelayer(delay_steps)
        , delay_steps(delay_steps) {
        RelearnException::check(delay_steps >= 0, "TransmissionDelayer::TransmissionDelayer: delay_steps must be >= 0");
    }

    RelearnTypes::step_type get_delay(const NeuronID& target_neuron, const RankNeuronId& source_neuron) override {
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
    RandomizedTransmissionDelayer(double mean, double stddev)
        : TransmissionDelayer(10)
        , mean(mean)
        , stddev(stddev) {
        RelearnException::check(stddev > 0, "TransmissionDelayer::TransmissionDelayer: standard deviation must be >= 0");
    }

    RelearnTypes::step_type get_delay(const NeuronID& target_neuron, const RankNeuronId& source_neuron) override {
        const auto d = RandomHolder::get_random_normal_double(RandomHolderKey::TransmissionDelay, mean, stddev);
        if (d < 0) {
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