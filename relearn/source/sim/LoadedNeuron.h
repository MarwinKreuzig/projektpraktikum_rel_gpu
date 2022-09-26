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

#include "Types.h"

#include "neurons/SignalType.h"

#include <cstdint>
#include <string>

/**
 * This struct represents a neuron loaded from a file. It is made up of:
 * (1) Its position
 * (2) Its id
 * (3) Its signal type
 * (4) Its area name
 */
struct LoadedNeuron {
    RelearnTypes::position_type pos{ 0 };
    NeuronID id{ NeuronID::uninitialized_id() };
    SignalType signal_type{ SignalType::Excitatory };
    std::string area_name{ "NOT SET" };
};

/**
 * This struct summarizes neurons loaded from a file. It is made up of:
 * The minimum and maximum of (x, y, z)-positions of the neurons,
 * the number of loaded excitatory neurons, and the number
 * of loaded inhibitory neurons
 */
struct LoadedNeuronsInfo {
    RelearnTypes::position_type minimum{};
    RelearnTypes::position_type maximum{};
    std::uint64_t number_excitatory_neurons{};
    std::uint64_t number_inhibitory_neurons{};
};
