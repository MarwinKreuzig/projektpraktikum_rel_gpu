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

class NetworkGraph;

class LinearSynapticInputCalculator : public SynapticInputCalculator {
public:
    LinearSynapticInputCalculator(const double k, const double base_background_activity, const double background_activity_mean, const double background_activity_stddev)
        : SynapticInputCalculator(k, base_background_activity, background_activity_mean, background_activity_stddev){};

    [[nodiscard]] virtual std::unique_ptr<SynapticInputCalculator> clone() const {
        return std::make_unique<LinearSynapticInputCalculator>(get_k(), get_base_background_activity(), get_background_activity_mean(), get_background_activity_stddev());
    }

    void update_synaptic_input(const NetworkGraph& network_graph, const std::vector<FiredStatus> fired, const std::vector<UpdateStatus>& disable_flags) override;

    void update_background_activity(const std::vector<UpdateStatus>& disable_flags) override;
};
