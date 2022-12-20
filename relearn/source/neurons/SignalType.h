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

#include "fmt/ostream.h"

#include <ostream>

/**
 * An instance of this enum classifies a synaptic elements as either excitatory or inhibitory.
 */
enum class SignalType { Excitatory,
    Inhibitory };

/**
 * @brief Pretty-prints the signal type to the chosen stream
 * @param out The stream to which to print the signal type
 * @param signal_type The signal type to print
 * @return The argument out, now altered with the signal type
 */
inline std::ostream& operator<<(std::ostream& out, const SignalType signal_type) {
    if (signal_type == SignalType::Excitatory) {
        return out << "Excitatory";
    }

    if (signal_type == SignalType::Inhibitory) {
        return out << "Inhibitory";
    }

    return out;
}

template <>
struct fmt::formatter<SignalType> : ostream_formatter { };
