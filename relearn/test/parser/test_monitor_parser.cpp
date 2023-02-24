/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_monitor_parser.h"

#include "adapter/random/RandomAdapter.h"
#include "adapter/helper/RankNeuronIdAdapter.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/tagged_id/TaggedIdAdapter.h"

#include "io/parser/MonitorParser.h"
#include "neurons/LocalAreaTranslator.h"

#include <memory>
#include <sstream>
#include <vector>

TEST_F(MonitorParserTest, testParseIds) {
    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt);
    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);

    std::vector<RankNeuronId> rank_neuron_ids{};
    rank_neuron_ids.reserve(number_ranks * TaggedIdAdapter::upper_bound_num_neurons);

    size_t my_number_neurons = 0;

    for (const auto rank : MPIRank::range(number_ranks)) {
        const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);

        if (rank == my_rank) {
            my_number_neurons = number_neurons;
        }

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            rank_neuron_ids.emplace_back(rank, NeuronID(neuron_id));
        }
    }

    std::shuffle(rank_neuron_ids.begin(), rank_neuron_ids.end(), mt);

    std::stringstream ss{};
    ss << "0:1";

    for (const auto& rni : rank_neuron_ids) {
        if (rni.get_rank() != my_rank) {
            ss << ';' << RankNeuronIdAdapter::codify_rank_neuron_id(rni);
            continue;
        }

        const auto use_default = RandomAdapter::get_random_bool(mt);

        if (use_default) {
            ss << ";-1:" << rni.get_neuron_id().get_neuron_id() + 1;
        } else {
            ss << ';' << RankNeuronIdAdapter::codify_rank_neuron_id(rni);
        }
    }

    auto translator = std::make_shared<LocalAreaTranslator>(std::vector<RelearnTypes::area_name>({ "random" }), std::vector<RelearnTypes::area_id>({ 0 }));

    const auto& parsed_ids = MonitorParser::parse_my_ids(ss.str(), my_rank, translator);

    ASSERT_EQ(parsed_ids.size(), my_number_neurons);

    for (auto i = 0; i < my_number_neurons; i++) {
        ASSERT_EQ(parsed_ids[i], NeuronID(i));
    }
}
