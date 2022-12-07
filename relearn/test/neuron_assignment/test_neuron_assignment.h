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

#include "RelearnTest.hpp"

class NeuronAssignmentTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }

    double calculate_box_length(const size_t number_neurons, const double um_per_neuron) const noexcept {
        return ceil(pow(static_cast<double>(number_neurons), 1 / 3.)) * um_per_neuron;
    }

    void generate_random_neurons(std::vector<Vec3d>& positions,
        std::vector<RelearnTypes::area_id>& neuron_id_to_area_ids, std::vector<RelearnTypes::area_name>& area_id_to_area_name, std::vector<SignalType>& types);
};
