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

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/tagged_id/TaggedIdAdapter.h"

#include "io/parser/MonitorParser.h"
#include "neurons/LocalAreaTranslator.h"

#include "gtest/gtest.h"
#include <memory>
#include <sstream>
#include <vector>

TEST_F(MonitorParserTest, testParseDescriptionFixed) {
    auto checker = [](std::string_view description, MPIRank rank, NeuronID::value_type neuron_id) {
        auto opt_rni = MonitorParser::parse_description(description, rank);
        ASSERT_TRUE(opt_rni.has_value());

        const auto& parsed_rni = opt_rni.value();
        RankNeuronId rni{ rank, NeuronID(neuron_id) };
        ASSERT_EQ(rni, parsed_rni);
    };

    checker("0:1", MPIRank(0), 1);
    checker("2:1", MPIRank(2), 1);
    checker("155:377",  MPIRank(155), 377);
    checker("-1:17",MPIRank(5), 17);
}

TEST_F(MonitorParserTest, testParseDescriptionFail) {
    auto checker = [](std::string_view description, MPIRank default_rank) {
        auto opt_rni = MonitorParser::parse_description(description, default_rank);
        ASSERT_FALSE(opt_rni.has_value());
    };

    checker("0:1:0", MPIRank::root_rank());
    checker("5:-4", MPIRank::root_rank());
    checker("+0:1", MPIRank::root_rank());
    checker("AB:1", MPIRank::root_rank());
    checker("-5:2", MPIRank::root_rank());
    checker("0:", MPIRank::root_rank());
    checker("5;2", MPIRank::root_rank());
    checker("", MPIRank::root_rank());
}

TEST_F(MonitorParserTest, testParseDescriptionException) {
    auto checker = [](std::string_view description, MPIRank default_rank) {
        ASSERT_THROW(auto opt_rni = MonitorParser::parse_description(description, default_rank);, RelearnException);
    };

    checker("0:0", MPIRank::root_rank());
    checker("1:0", MPIRank::root_rank());
    checker("-1:0", MPIRank::root_rank());
    checker("24575:0", MPIRank::root_rank());
}

TEST_F(MonitorParserTest, testParseDescriptionRandom) {
    for (auto i = 0; i < 10000; i++) {
        const auto& [rni, descr] = generate_random_rank_neuron_id_description();

        auto opt_rni = MonitorParser::parse_description(descr, MPIRank(0));
        ASSERT_TRUE(opt_rni.has_value());

        const auto& parsed_rni = opt_rni.value();
        ASSERT_EQ(add_one_to_neuron_id(rni), parsed_rni);
    }
}

TEST_F(MonitorParserTest, testParseDescriptions) {
    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);

    std::vector<RankNeuronId> rank_neuron_ids{};
    rank_neuron_ids.reserve(number_neurons + 1);

    std::stringstream ss{};

    const auto& [first_rni, first_description] = generate_random_rank_neuron_id_description();
    rank_neuron_ids.emplace_back(add_one_to_neuron_id(first_rni));

    ss << first_description;

    for (auto i = 0; i < number_neurons; i++) {
        const auto& [new_rni, new_description] = generate_random_rank_neuron_id_description();
        ss << ';' << new_description;

        rank_neuron_ids.emplace_back(add_one_to_neuron_id(new_rni));
    }

    const auto& parsed_rnis = MonitorParser::parse_multiple_description(ss.str(), MPIRank(3));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis);
}

TEST_F(MonitorParserTest, testParseDescriptionsFixed) {
    std::vector<RankNeuronId> rank_neuron_ids{
        { MPIRank(2), NeuronID(100) },
        { MPIRank(5), NeuronID(6) },
        { MPIRank(0), NeuronID(122) },
        { MPIRank(2), NeuronID(100) },
        { MPIRank(1674), NeuronID(1) },
        { MPIRank(89512), NeuronID(6) },
        { MPIRank(0), NeuronID(1) },
        { MPIRank(0), NeuronID(1) },
    };

    constexpr auto description_1 = "2:100;5:6;0:122;2:100;1674:1;89512:6;0:1;0:1";
    constexpr auto description_2 = "2:100;-1:6;0:122;2:100;1674:1;89512:6;0:1;0:1";
    constexpr auto description_3 = "2:100;5:6;-1:122;2:100;1674:1;89512:6;-1:1;0:1";
    constexpr auto description_4 = "2:100;5:6;-1:122;-8:800;2:100;6:;1674:1;-999:5;89512:6;-1:1;0:1";

    const auto& parsed_rnis_1 = MonitorParser::parse_multiple_description(description_1, MPIRank(3));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_1);

    const auto& parsed_rnis_2 = MonitorParser::parse_multiple_description(description_2, MPIRank(5));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_2);

    const auto& parsed_rnis_3 = MonitorParser::parse_multiple_description(description_3, MPIRank(0));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_3);

    const auto& parsed_rnis_4 = MonitorParser::parse_multiple_description(description_4, MPIRank(0));
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_4);
}

TEST_F(MonitorParserTest, testExtractNeuronIDs) {
    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);

    std::vector<RankNeuronId> rank_neuron_ids{};
    rank_neuron_ids.reserve(number_neurons + 2);

    const auto my_rank = MPIRankAdapter::get_random_mpi_rank(100, mt);

    for (auto i = 0; i < number_neurons; i++) {
        const auto& [new_rni, _] = generate_random_rank_neuron_id_description();
        rank_neuron_ids.emplace_back(new_rni);
    }

    const auto position_1 = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);
    const auto position_2 = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);

    rank_neuron_ids.insert(rank_neuron_ids.begin() + position_1, RankNeuronId(my_rank, NeuronID(42)));
    rank_neuron_ids.insert(rank_neuron_ids.begin() + position_2, RankNeuronId(my_rank, NeuronID(9874)));

    std::vector<RankNeuronId> filtered{};
    std::copy_if(rank_neuron_ids.begin(), rank_neuron_ids.end(), std::back_inserter(filtered), [my_rank](const RankNeuronId& rni) { const auto& [rank, id] = rni; return rank == my_rank; });

    std::vector<NeuronID> golden_ids{};
    std::transform(filtered.begin(), filtered.end(), std::back_inserter(golden_ids), [](const RankNeuronId& rni) { const auto& [rank, id] = rni; return NeuronID(id.get_neuron_id() - 1); });

    const auto& extracted_ids = MonitorParser::extract_my_ids(rank_neuron_ids, my_rank);

    ASSERT_EQ(golden_ids, extracted_ids);
}

TEST_F(MonitorParserTest, testRemoveAndSort) {
    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);

    std::vector<NeuronID> neuron_ids{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0; i < number_neurons; i++) {
        neuron_ids.emplace_back(TaggedIdAdapter::get_random_neuron_id(number_neurons, mt));
    }

    const auto& unique_and_filtered = MonitorParser::remove_duplicates_and_sort(neuron_ids);

    for (auto i = 0; i < unique_and_filtered.size() - 1; i++) {
        ASSERT_LE(unique_and_filtered[i].get_neuron_id(), unique_and_filtered[i + 1].get_neuron_id());
    }

    for (const auto& original_id : neuron_ids) {
        const auto pos = std::ranges::find(unique_and_filtered, original_id);
        ASSERT_NE(pos, unique_and_filtered.end());
    }

    for (const auto& new_id : unique_and_filtered) {
        const auto pos = std::ranges::find(neuron_ids, new_id);
        ASSERT_NE(pos, neuron_ids.end());
    }
}

TEST_F(MonitorParserTest, testRemoveAndSortException1) {
    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);

    std::vector<NeuronID> neuron_ids{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0; i < number_neurons; i++) {
        neuron_ids.emplace_back(TaggedIdAdapter::get_random_neuron_id(number_neurons, mt));
    }

    const auto virtual_rma = RandomAdapter::get_random_integer<NeuronID::value_type>(0, 100000, mt);
    const auto position = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);

    neuron_ids.insert(neuron_ids.begin() + position, NeuronID(true, virtual_rma));

    ASSERT_THROW(const auto& unique_and_filtered = MonitorParser::remove_duplicates_and_sort(neuron_ids);, RelearnException);
}

TEST_F(MonitorParserTest, testRemoveAndSortException2) {
    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);

    std::vector<NeuronID> neuron_ids{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0; i < number_neurons; i++) {
        neuron_ids.emplace_back(TaggedIdAdapter::get_random_neuron_id(number_neurons, mt));
    }

    const auto position = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);

    neuron_ids.insert(neuron_ids.begin() + position, NeuronID{});

    ASSERT_THROW(const auto& unique_and_filtered = MonitorParser::remove_duplicates_and_sort(neuron_ids);, RelearnException);
}

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
            ss << ';' << codify_rank_neuron_id(rni);
            continue;
        }

        const auto use_default = RandomAdapter::get_random_bool(mt);

        if (use_default) {
            ss << ";-1:" << rni.get_neuron_id().get_neuron_id() + 1;
        } else {
            ss << ';' << codify_rank_neuron_id(rni);
        }
    }

    auto translator = std::make_shared<LocalAreaTranslator>(std::vector<RelearnTypes::area_name>({ "random" }), std::vector<RelearnTypes::area_id>({ 0 }));

    const auto& parsed_ids = MonitorParser::parse_my_ids(ss.str(), my_rank, translator);

    ASSERT_EQ(parsed_ids.size(), my_number_neurons);

    for (auto i = 0; i < my_number_neurons; i++) {
        ASSERT_EQ(parsed_ids[i], NeuronID(i));
    }
}
