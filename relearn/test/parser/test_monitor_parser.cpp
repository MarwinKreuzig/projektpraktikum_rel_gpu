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

#include "adapter/helper/RankNeuronIdAdapter.h"
#include "adapter/random/RandomAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"

#include "io/parser/MonitorParser.h"
#include "neurons/LocalAreaTranslator.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"
#include "util/ranges/Functional.hpp"
#include "util/shuffle/shuffle.h"

#include <memory>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/generate.hpp>
#include <range/v3/view/map.hpp>
#include <sstream>
#include <vector>

TEST_F(MonitorParserTest, testParseIds) {
    const auto number_ranks = MPIRankAdapter::get_random_number_ranks(mt);
    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(number_ranks, mt);

    size_t my_number_neurons = 0;

    const auto random_number_neurons = [this]() { return NeuronIdAdapter::get_random_number_neurons(mt); };

    const auto create_rank_neuron_ids = [this, my_rank, &my_number_neurons](const auto& rank_num_neurons_pair) {
        const auto& rank = std::get<0>(rank_num_neurons_pair);
        const auto& number_neurons = std::get<1>(rank_num_neurons_pair);

        if (rank == my_rank) {
            my_number_neurons = number_neurons;
        }

        return NeuronID::range(number_neurons)
            | ranges::views::transform([rank](const NeuronID& neuron_id) -> RankNeuronId { return { rank, neuron_id }; });
    };

    const auto rank_neuron_ids = ranges::views::zip(
                               MPIRank::range(number_ranks),
                               ranges::views::generate(random_number_neurons))
        | ranges::views::for_each(create_rank_neuron_ids)
        | ranges::to_vector
        | actions::shuffle(mt);

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

    for (const auto neuron_id : NeuronID::range_id(my_number_neurons)) {
        ASSERT_EQ(parsed_ids[neuron_id], NeuronID(neuron_id));
    }
}
