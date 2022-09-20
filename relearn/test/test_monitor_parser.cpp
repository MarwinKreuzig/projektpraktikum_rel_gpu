#include "gtest/gtest.h"

#include "RelearnTest.hpp"

#include "util/MonitorParser.h"

#include <sstream>

TEST_F(MonitorParserTest, testParseDescriptionFixed) {
    auto checker = [](std::string_view description, int default_rank, int rank, NeuronID::value_type neuron_id) {
        auto opt_rni = MonitorParser::parse_description(description, default_rank);
        ASSERT_TRUE(opt_rni.has_value());

        const auto& parsed_rni = opt_rni.value();
        RankNeuronId rni{ rank, NeuronID(neuron_id) };
        ASSERT_EQ(rni, parsed_rni);
    };

    checker("0:0", 0, 0, 0);
    checker("2:0", 0, 2, 0);
    checker("155:377", 17, 155, 377);
    checker("-1:17", 5, 5, 17);
}

TEST_F(MonitorParserTest, testParseDescriptionFail) {
    auto checker = [](std::string_view description, int default_rank) {
        auto opt_rni = MonitorParser::parse_description(description, default_rank);
        ASSERT_FALSE(opt_rni.has_value());
    };

    checker("0:0:0", 0);
    checker("5:-4", 0);
    checker("+0:0", 0);
    checker("AB:0", 0);
    checker("-5:2", 0);
    checker("0:", 0);
    checker("5;2", 0);
    checker("", 0);
}

TEST_F(MonitorParserTest, testParseDescriptionRandom) {
    for (auto i = 0; i < 10000; i++) {
        const auto& [rni, descr] = generate_random_rank_neuron_id_description();

        auto opt_rni = MonitorParser::parse_description(descr, 0);
        ASSERT_TRUE(opt_rni.has_value());

        const auto& parsed_rni = opt_rni.value();
        ASSERT_EQ(rni, parsed_rni);
    }
}

TEST_F(MonitorParserTest, testParseDescriptions) {
    const auto number_neurons = get_random_number_neurons();

    std::vector<RankNeuronId> rank_neuron_ids{};
    rank_neuron_ids.reserve(number_neurons + 1);

    std::stringstream ss{};

    const auto& [first_rni, first_description] = generate_random_rank_neuron_id_description();
    rank_neuron_ids.emplace_back(first_rni);

    ss << first_description;

    for (auto i = 0; i < number_neurons; i++) {
        const auto& [new_rni, new_description] = generate_random_rank_neuron_id_description();
        ss << ';' << new_description;

        rank_neuron_ids.emplace_back(new_rni);
    }

    const auto& parsed_rnis = MonitorParser::parse_multiple_description(ss.str(), 3);
    ASSERT_EQ(rank_neuron_ids, parsed_rnis);
}

TEST_F(MonitorParserTest, testParseDescriptionsFixed) {
    std::vector<RankNeuronId> rank_neuron_ids{
        { 2, NeuronID(100) },
        { 5, NeuronID(6) },
        { 0, NeuronID(122) },
        { 2, NeuronID(100) },
        { 1674, NeuronID(0) },
        { 89512, NeuronID(6) },
        { 0, NeuronID(0) },
        { 0, NeuronID(0) },
    };

    constexpr auto description_1 = "2:100;5:6;0:122;2:100;1674:0;89512:6;0:0;0:0";
    constexpr auto description_2 = "2:100;-1:6;0:122;2:100;1674:0;89512:6;0:0;0:0";
    constexpr auto description_3 = "2:100;5:6;-1:122;2:100;1674:0;89512:6;-1:0;0:0";
    constexpr auto description_4 = "2:100;5:6;-1:122;-8:800;2:100;6:;1674:0;-999:5;89512:6;-1:0;0:0";

    const auto& parsed_rnis_1 = MonitorParser::parse_multiple_description(description_1, 3);
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_1);

    const auto& parsed_rnis_2 = MonitorParser::parse_multiple_description(description_2, 5);
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_2);

    const auto& parsed_rnis_3 = MonitorParser::parse_multiple_description(description_3, 0);
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_3);

    const auto& parsed_rnis_4 = MonitorParser::parse_multiple_description(description_4, 0);
    ASSERT_EQ(rank_neuron_ids, parsed_rnis_4);
}

TEST_F(MonitorParserTest, testParseDescriptionsException) {
    constexpr auto description = "2:100;5:6;0:122;2:100;1674:0;89512:6;0:0;0:0";
    ASSERT_THROW(MonitorParser::parse_multiple_description(description, -2), RelearnException);
    ASSERT_THROW(MonitorParser::parse_multiple_description(description, -3), RelearnException);
    ASSERT_THROW(MonitorParser::parse_multiple_description(description, -543), RelearnException);
    ASSERT_THROW(MonitorParser::parse_multiple_description(description, -78923), RelearnException);
    ASSERT_THROW(MonitorParser::parse_multiple_description(description, -36464665431), RelearnException);
}

TEST_F(MonitorParserTest, testExtractNeuronIDs) {
    const auto number_neurons = get_random_number_neurons();

    std::vector<RankNeuronId> rank_neuron_ids{};
    rank_neuron_ids.reserve(number_neurons + 2);

    const auto my_rank = get_random_rank(100);

    for (auto i = 0; i < number_neurons; i++) {
        const auto& [new_rni, _] = generate_random_rank_neuron_id_description();
        rank_neuron_ids.emplace_back(new_rni);
    }

    const auto position_1 = get_random_integer<size_t>(0, number_neurons);
    const auto position_2 = get_random_integer<size_t>(0, number_neurons);

    rank_neuron_ids.insert(rank_neuron_ids.begin() + position_1, RankNeuronId(my_rank, NeuronID(42)));
    rank_neuron_ids.insert(rank_neuron_ids.begin() + position_2, RankNeuronId(my_rank, NeuronID(9874)));

    std::vector<RankNeuronId> filtered{};
    std::copy_if(rank_neuron_ids.begin(), rank_neuron_ids.end(), std::back_inserter(filtered), [my_rank](const RankNeuronId& rni) { const auto& [rank, id] = rni; return rank == my_rank; });

    std::vector<NeuronID> golden_ids{};
    std::transform(filtered.begin(), filtered.end(), std::back_inserter(golden_ids), [](const RankNeuronId& rni) { const auto& [rank, id] = rni; return id; });

    const auto& extracted_ids = MonitorParser::extract_my_ids(rank_neuron_ids, my_rank);

    ASSERT_EQ(golden_ids, extracted_ids);
}

TEST_F(MonitorParserTest, testRemoveAndSort) {
    const auto number_neurons = get_random_number_neurons();

    std::vector<NeuronID> neuron_ids{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0; i < number_neurons; i++) {
        neuron_ids.emplace_back(get_random_neuron_id(number_neurons));
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
    const auto number_neurons = get_random_number_neurons();

    std::vector<NeuronID> neuron_ids{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0; i < number_neurons; i++) {
        neuron_ids.emplace_back(get_random_neuron_id(number_neurons));
    }

    const auto virtual_rma = get_random_integer<NeuronID::value_type>(0, 100000);
    const auto position = get_random_integer<size_t>(0, number_neurons);

    neuron_ids.insert(neuron_ids.begin() + position, NeuronID(true, virtual_rma));

    ASSERT_THROW(const auto& unique_and_filtered = MonitorParser::remove_duplicates_and_sort(neuron_ids);, RelearnException);
}

TEST_F(MonitorParserTest, testRemoveAndSortException2) {
    const auto number_neurons = get_random_number_neurons();

    std::vector<NeuronID> neuron_ids{};
    neuron_ids.reserve(number_neurons);

    for (auto i = 0; i < number_neurons; i++) {
        neuron_ids.emplace_back(get_random_neuron_id(number_neurons));
    }

    const auto position = get_random_integer<size_t>(0, number_neurons);

    neuron_ids.insert(neuron_ids.begin() + position, NeuronID{});

    ASSERT_THROW(const auto& unique_and_filtered = MonitorParser::remove_duplicates_and_sort(neuron_ids);, RelearnException);
}

TEST_F(MonitorParserTest, testParseIds) {
    const auto number_ranks = get_random_number_ranks();
    const auto my_rank = get_random_rank(number_ranks);

    std::vector<RankNeuronId> rank_neuron_ids{};
    rank_neuron_ids.reserve(number_ranks * upper_bound_num_neurons);

    auto my_number_neurons = 0;

    for (auto rank = 0; rank < number_ranks; rank++) {
        const auto number_neurons = get_random_number_neurons();

        if (rank == my_rank) {
            my_number_neurons = number_neurons;
        }

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            rank_neuron_ids.emplace_back(rank, NeuronID(neuron_id));
        }
    }

    std::shuffle(rank_neuron_ids.begin(), rank_neuron_ids.end(), mt);

    std::stringstream ss{};
    ss << "0:0";

    for (const auto& rni : rank_neuron_ids) {
        if (rni.get_rank() != my_rank) {
            ss << ';' << codify_rank_neuron_id(rni);
            continue;
        }

        const auto use_default = get_random_bool();

        if (use_default) {
            ss << ";-1:" << rni.get_neuron_id();
        } else {
            ss << ';' << codify_rank_neuron_id(rni);
        }
    }

    const auto& parsed_ids = MonitorParser::parse_my_ids(ss.str(), my_rank, my_rank);

    ASSERT_EQ(parsed_ids.size(), my_number_neurons);

    for (auto i = 0; i < my_number_neurons; i++) {
        ASSERT_EQ(parsed_ids[i], NeuronID(i));
    }
}
