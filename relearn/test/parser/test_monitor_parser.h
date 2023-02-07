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

#include "adapter/helper/RankNeuronIdAdapter.h"

#include "neurons/helper/RankNeuronId.h"

#include <sstream>
#include <string>
#include <utility>

class MonitorParserTest : public RelearnTest {
protected:
    std::string codify_rank_neuron_id(const RankNeuronId& rni) {
        std::stringstream ss{};
        ss << rni.get_rank().get_rank() << ':' << (rni.get_neuron_id().get_neuron_id() + 1);
        return ss.str();
    }

    std::pair<RankNeuronId, std::string> generate_random_rank_neuron_id_description() {
        auto rank_neuron_id = RankNeuronIdAdapter::generate_random_rank_neuron_id(mt);
        auto description = codify_rank_neuron_id(rank_neuron_id);
        return { std::move(rank_neuron_id), std::move(description) };
    }

    RankNeuronId add_one_to_neuron_id(const RankNeuronId& rni) {
        const auto& [rank, neuron_id] = rni;
        const auto id = neuron_id.get_neuron_id();
        return RankNeuronId(rank, NeuronID(id + 1));
    }
};
