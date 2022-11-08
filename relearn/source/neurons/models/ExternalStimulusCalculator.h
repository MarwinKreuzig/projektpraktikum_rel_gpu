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

#include "neurons/UpdateStatus.h"
#include "neurons/models/ModelParameter.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"

#include <memory>
#include <vector>

enum class ExternalStimulusCalculatorType : char {
    Null,
    Function
};

/**
 * @brief Pretty-prints the external stimulus calculator type to the chosen stream
 * @param out The stream to which to print the external stimulus calculator
 * @param calculator_type The external stimulus calculator to print
 * @return The argument out, now altered with the external stimulus calculator
 */
inline std::ostream& operator<<(std::ostream& out, const ExternalStimulusCalculatorType& calculator_type) {
    if (calculator_type == ExternalStimulusCalculatorType::Null) {
        return out << "Null";
    }

    if (calculator_type == ExternalStimulusCalculatorType::Function) {
        return out << "Function";
    }

    return out;
}

/**
 * This class provides an interface to calculate the background activity that neurons receive.
 * It also provides some default/min/max values via public static constexpr members,
 * because the child classes share them.
 */
class ExternalStimulusCalculator {

public:
    /**
     * @brief Construcs a new instance of type SynapticInputCalculator with 0 neurons.
     */
    ExternalStimulusCalculator() = default;

    virtual ~ExternalStimulusCalculator() = default;

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] virtual std::unique_ptr<ExternalStimulusCalculator> clone() const = 0;

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0
     */
    virtual void init(size_t number_neurons) {
        RelearnException::check(number_neurons > 0, "ExternalStimulusCalculator::init: number_neurons was 0");

        number_local_neurons = number_neurons;
        external_stimulus.resize(number_neurons, 0.0);
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    virtual void create_neurons(size_t creation_count) {
        RelearnException::check(number_local_neurons > 0, "BackgroundActivityCalculator::create_neurons: number_local_neurons was 0");
        RelearnException::check(creation_count > 0, "BackgroundActivityCalculator::create_neurons: creation_count was 0");

        const auto current_size = number_local_neurons;
        const auto new_size = current_size + creation_count;

        number_local_neurons = new_size;
        external_stimulus.resize(new_size, 0.0);
    }

    /**
     * @brief Updates the background activity based on which neurons to update
     * @param step The current update step
     * @param disable_flags Which neurons are disabled
     * @exception Throws a RelearnException if the number of local neurons didn't match the sizes of the arguments
     */
    virtual void update_input(const size_t step, const std::vector<UpdateStatus>& disable_flags) = 0;

    /**
     * @brief Returns the calculated background activity for the given neuron. Changes after calls to update_input(...)
     * @param neuron_id The neuron to query
     * @exception Throws a RelearnException if the neuron_id is too large for the stored number of neurons
     * @return The background activity for the given neuron
     */
    [[nodiscard]] double get_external_stimulus(const NeuronID& neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::get_x: id is too large: {}", neuron_id);
        return external_stimulus[local_neuron_id];
    }

    /**
     * @brief Returns the number of neurons that are stored in the object
     * @return The number of neurons that are stored in the object
     */
    [[nodiscard]] size_t get_number_neurons() const noexcept {
        return number_local_neurons;
    }

    /**
     * @brief Returns the parameters of this instance, i.e., the attributes which change the behavior when calculating the input
     * @return The parameters
     */
    [[nodiscard]] virtual std::vector<ModelParameter> get_parameter() {
        return {};
    }

protected:
    /**
     * @brief Sets the background activity for the given neuron
     * @param neuron_id The local neuron
     * @param value The new background activity
     * @exception Throws a RelearnException if the neuron_id is to large
     */
    void set_external_stimulus(const size_t neuron_id, const double value) {
        RelearnException::check(neuron_id < number_local_neurons, "SynapticInputCalculator::set_background_activity: neuron_id was too large: {} vs {}", neuron_id, number_local_neurons);
        external_stimulus[neuron_id] = value;
    }

    void reset_external_stimulus() {
        std::fill(external_stimulus.begin(), external_stimulus.end(), 0.0);
    }

private:
    size_t number_local_neurons{};

    std::vector<double> external_stimulus{};
};