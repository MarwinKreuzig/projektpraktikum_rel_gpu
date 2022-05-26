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

struct LoadedNeuron {
    RelearnTypes::position_type pos{ 0 };
    NeuronID id{ NeuronID::uninitialized_id() };
    SignalType signal_type{ SignalType::Excitatory };
    std::string area_name{ "NOT SET" };
};

struct LoadedNeuronsInfo {
    RelearnTypes::position_type minimum{};
    RelearnTypes::position_type maximum{};
    std::uint64_t number_excitatory_neurons{};
    std::uint64_t number_inhibitory_neurons{};
};
