#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/neurons/Neurons.h"
#include "../source/neurons/models/NeuronModels.h"

#include "../source/neurons/NetworkGraph.h"

#include "../source/structure/Partition.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <stack>
#include <tuple>
#include <vector>

void assert_empty(const NeuronsExtraInfo& nei, size_t number_neurons) {
    const auto& area_names = nei.get_area_names();
    ASSERT_EQ(0, area_names.size());

    for (auto neuron_id = 0; neuron_id < number_neurons * 2; neuron_id++) {
        ASSERT_THROW(auto tmp = nei.get_position(neuron_id), RelearnException);
        ASSERT_THROW(auto tmp = nei.get_area_name(neuron_id), RelearnException);
    }
}

void assert_contains(const NeuronsExtraInfo& nei, size_t number_neurons, size_t num_neurons_check, const std::vector<std::string>& expected_area_names,
    const std::vector<Vec3d>& positions) {

    ASSERT_EQ(num_neurons_check, expected_area_names.size());
    ASSERT_EQ(num_neurons_check, positions.size());

    const auto& actual_area_names = nei.get_area_names();

    ASSERT_EQ(actual_area_names.size(), number_neurons);

    for (auto neuron_id = 0; neuron_id < num_neurons_check; neuron_id++) {
        ASSERT_EQ(expected_area_names[neuron_id], actual_area_names[neuron_id]);
        ASSERT_EQ(expected_area_names[neuron_id], nei.get_area_name(neuron_id));
        ASSERT_EQ(nei.get_position(neuron_id), positions[neuron_id]);
    }

    for (auto neuron_id = number_neurons; neuron_id < number_neurons * 2; neuron_id++) {
        ASSERT_THROW(auto tmp = nei.get_position(neuron_id), RelearnException);
        ASSERT_THROW(auto tmp = nei.get_area_name(neuron_id), RelearnException);
    }
}

TEST_F(NeuronsTest, testNeuronsConstructor) {
    auto partition = std::make_shared<Partition>(1, 0);

    auto model = std::make_unique<models::PoissonModel>();

    auto dends_ex = std::make_unique<SynapticElements>(ElementType::DENDRITE, 0.2);
    auto dends_in = std::make_unique<SynapticElements>(ElementType::DENDRITE, 0.2);
    auto axs = std::make_unique<SynapticElements>(ElementType::AXON, 0.2);

    Neurons neurons{ partition, std::move(model), std::move(axs), std::move(dends_ex), std::move(dends_in) };
}

TEST_F(NeuronsTest, testNeuronRankIdValid) {
    for (auto it = 0; it < iterations; it++) {
        std::uniform_int_distribution<int> urd_rank(0, 100);
        std::uniform_int_distribution<size_t> urd_id(0, Constants::uninitialized - 1);

        for (auto i = 0; i < 1000; i++) {
            const auto rank = urd_rank(mt);
            const auto id = urd_id(mt);

            const RankNeuronId rni{ rank, id };

            ASSERT_EQ(rni.get_neuron_id(), id);
            ASSERT_EQ(rni.get_rank(), rank);
        }
    }
}

TEST_F(NeuronsTest, testNeuronRankIdInvalidRank) {
    for (auto it = 0; it < iterations; it++) {
        std::uniform_int_distribution<int> urd_rank(-1000000, -1);
        std::uniform_int_distribution<size_t> urd_id(0, Constants::uninitialized - 1);

        for (auto i = 0; i < 1000; i++) {
            const auto rank = urd_rank(mt);
            const auto id = urd_id(mt);

            RankNeuronId rni(rank, id);

            ASSERT_NO_THROW(auto tmp = rni.get_neuron_id());
            ASSERT_THROW(auto tmp = rni.get_rank(), RelearnException);
        }
    }
}

TEST_F(NeuronsTest, testNeuronRankIdInvalidId) {
    for (auto it = 0; it < iterations; it++) {
        std::uniform_int_distribution<int> urd_rank(0, 100);
        std::uniform_int_distribution<size_t> urd_id(0, Constants::uninitialized - 1);

        for (auto i = 0; i < 1000; i++) {
            const auto rank = urd_rank(mt);
            const auto id = urd_id(mt);

            RankNeuronId rni(rank, id + Constants::uninitialized);

            ASSERT_NO_THROW(auto tmp = rni.get_rank());
            ASSERT_THROW(auto tmp = rni.get_neuron_id(), RelearnException);
        }
    }
}

TEST_F(NeuronsTest, testNeuronRankIdEquality) {
    std::uniform_int_distribution<int> uid_rank(0, 10);
    std::uniform_int_distribution<size_t> uid_id(0, 500);

    for (auto it = 0; it < iterations; it++) {
        for (auto i = 0; i < 1000; i++) {
            const auto rank_1 = uid_rank(mt);
            const auto id_1 = uid_id(mt);

            const auto rank_2 = uid_rank(mt);
            const auto id_2 = uid_id(mt);

            const RankNeuronId rni_1(rank_1, id_1);
            const RankNeuronId rni_2(rank_2, id_2);

            if (rank_1 == rank_2 && id_1 == id_2) {
                ASSERT_EQ(rni_1, rni_2);
            } else {
                ASSERT_NE(rni_1, rni_2);
            }
        }
    }
}

TEST_F(NeuronsTest, testNeuronsExtraInfo) {
    std::uniform_int_distribution<size_t> uid_id(0, 10000);

    for (auto it = 0; it < iterations; it++) {
        NeuronsExtraInfo nei{};

        assert_empty(nei, 1000);

        ASSERT_THROW(nei.set_area_names(std::vector<std::string>{}), RelearnException);

        assert_empty(nei, 1000);

        const auto new_size = uid_id(mt);

        ASSERT_THROW(nei.set_area_names(std::vector<std::string>(new_size)), RelearnException);

        assert_empty(nei, 1000);
    }
}

TEST_F(NeuronsTest, testNeuronsExtraInfoInit) {
    std::uniform_int_distribution<size_t> uid_id(0, 10000);
    std::uniform_real_distribution<double> urd_pos(-10000.0, 10000.0);

    for (auto it = 0; it < iterations; it++) {
        NeuronsExtraInfo nei{};

        const auto number_neurons = uid_id(mt);

        nei.init(number_neurons);
        assert_empty(nei, number_neurons);

        auto num_neurons_wrong = uid_id(mt);
        if (num_neurons_wrong == number_neurons) {
            num_neurons_wrong++;
        }

        std::vector<Vec3d> positions_wrong(num_neurons_wrong);
        std::vector<std::string> area_names_wrong(num_neurons_wrong);

        ASSERT_THROW(nei.set_positions(positions_wrong), RelearnException);
        ASSERT_THROW(nei.set_area_names(area_names_wrong), RelearnException);

        assert_empty(nei, number_neurons);

        std::vector<Vec3d> positions_right(number_neurons);
        std::vector<std::string> area_names_right(number_neurons);

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            positions_right[neuron_id] = { urd_pos(mt), urd_pos(mt), urd_pos(mt) };
            area_names_right[neuron_id] = std::to_string(urd_pos(mt));
        }

        nei.set_positions(positions_right);
        nei.set_area_names(area_names_right);

        assert_contains(nei, number_neurons, number_neurons, area_names_right, positions_right);

        std::vector<Vec3d> positions_right_2(number_neurons);
        std::vector<std::string> area_names_right_2(number_neurons);

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            positions_right_2[neuron_id] = { urd_pos(mt), urd_pos(mt), urd_pos(mt) };
            area_names_right_2[neuron_id] = std::to_string(urd_pos(mt));
        }

        nei.set_positions(positions_right_2);
        nei.set_area_names(area_names_right_2);

        assert_contains(nei, number_neurons, number_neurons, area_names_right_2, positions_right_2);
    }
}

TEST_F(NeuronsTest, testNeuronsExtraInfoCreate) {

    std::uniform_int_distribution<size_t> uid_id(1, 10000);
    std::uniform_real_distribution<double> urd_pos(-10000.0, 10000.0);

    for (auto it = 0; it < iterations; it++) {
        NeuronsExtraInfo nei{};

        const auto num_neurons_init = uid_id(mt);
        const auto num_neurons_create_1 = uid_id(mt);
        const auto num_neurons_create_2 = uid_id(mt);

        const auto num_neurons_total_1 = num_neurons_init + num_neurons_create_1;
        const auto num_neurons_total_2 = num_neurons_total_1 + num_neurons_create_2;

        nei.init(num_neurons_init);

        ASSERT_THROW(nei.create_neurons(num_neurons_create_1), RelearnException);

        assert_empty(nei, num_neurons_init);

        std::vector<Vec3d> positions_right(num_neurons_init);
        std::vector<std::string> area_names_right(num_neurons_init);

        for (auto neuron_id = 0; neuron_id < num_neurons_init; neuron_id++) {
            positions_right[neuron_id] = { urd_pos(mt), urd_pos(mt), urd_pos(mt) };
            area_names_right[neuron_id] = std::to_string(urd_pos(mt));
        }

        nei.set_positions(positions_right);
        nei.set_area_names(area_names_right);

        nei.create_neurons(num_neurons_create_1);

        assert_contains(nei, num_neurons_total_1, num_neurons_init, area_names_right, positions_right);

        std::vector<Vec3d> positions_right_2(num_neurons_total_1);
        std::vector<std::string> area_names_right_2(num_neurons_total_1);

        for (auto neuron_id = 0; neuron_id < num_neurons_total_1; neuron_id++) {
            positions_right_2[neuron_id] = { urd_pos(mt), urd_pos(mt), urd_pos(mt) };
            area_names_right_2[neuron_id] = std::to_string(urd_pos(mt));
        }

        nei.set_positions(positions_right_2);
        nei.set_area_names(area_names_right_2);

        assert_contains(nei, num_neurons_total_1, num_neurons_total_1, area_names_right_2, positions_right_2);

        nei.create_neurons(num_neurons_create_2);

        assert_contains(nei, num_neurons_total_2, num_neurons_total_1, area_names_right_2, positions_right_2);

        std::vector<Vec3d> positions_right_3(num_neurons_total_2);
        std::vector<std::string> area_names_right_3(num_neurons_total_2);

        for (auto neuron_id = 0; neuron_id < num_neurons_total_2; neuron_id++) {
            positions_right_3[neuron_id] = { urd_pos(mt), urd_pos(mt), urd_pos(mt) };
            area_names_right_3[neuron_id] = std::to_string(urd_pos(mt));
        }

        nei.set_positions(positions_right_3);
        nei.set_area_names(area_names_right_3);

        assert_contains(nei, num_neurons_total_2, num_neurons_total_2, area_names_right_3, positions_right_3);
    }
}
