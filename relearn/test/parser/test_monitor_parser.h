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

#include "neurons/helper/RankNeuronId.h"

#include <sstream>
#include <string>
#include <utility>

class MonitorParserTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }

    std::string codify_rank_neuron_id(const RankNeuronId& rni) {
        std::stringstream ss{};
        ss << rni.get_rank() << ':' << rni.get_neuron_id();
        return ss.str();
    }

    std::pair<RankNeuronId, std::string> generate_random_rank_neuron_id_description() {
        auto rank_neuron_id = generate_random_rank_neuron_id();
        auto description = codify_rank_neuron_id(rank_neuron_id);
        return { std::move(rank_neuron_id), std::move(description) };
    }
};
