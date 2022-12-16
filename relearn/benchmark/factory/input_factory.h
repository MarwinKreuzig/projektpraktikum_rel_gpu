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

#include "neurons/models/SynapticInputCalculator.h"
#include "neurons/models/SynapticInputCalculators.h"

#include <memory>

class InputFactory {
public:
    static std::unique_ptr<SynapticInputCalculator> construct_linear_input(const double synapse_conductance = 1.0) {
        return std::make_unique<LinearSynapticInputCalculator>(synapse_conductance);
    }

    static std::unique_ptr<SynapticInputCalculator> construct_logarithmic_input(const double synapse_conductance = 1.0, const double scaling_factor = 1.0) {
        return std::make_unique<LogarithmicSynapticInputCalculator>(synapse_conductance, scaling_factor);
    }
};
