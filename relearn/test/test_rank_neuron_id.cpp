#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../../../source/neurons/helper/RankNeuronId.h"

TEST_F(RankNeuronIdTest, testNeuronRankIdValid) {
    for (auto i = 0; i < 1000; i++) {
        const auto rank = static_cast<int>(get_random_number_ranks());
        const auto id = get_random_number_neurons();

        const RankNeuronId rni{ rank, id };

        ASSERT_EQ(rni.get_neuron_id(), id);
        ASSERT_EQ(rni.get_rank(), rank);
    }
}

TEST_F(RankNeuronIdTest, testNeuronRankIdInvalidRank) {
    for (auto i = 0; i < 1000; i++) {
        const auto rank = get_random_integer<int>(-1000, -1);
        const auto id = get_random_number_neurons();

        RankNeuronId rni(rank, id);

        ASSERT_NO_THROW(auto tmp = rni.get_neuron_id());
        ASSERT_THROW(auto tmp = rni.get_rank(), RelearnException);
    }
}

TEST_F(RankNeuronIdTest, testNeuronRankIdInvalidId) {
    for (auto i = 0; i < 1000; i++) {
        const auto rank = static_cast<int>(get_random_number_ranks());
        const auto id = get_random_number_neurons();

        RankNeuronId rni(rank, id + Constants::uninitialized);

        ASSERT_NO_THROW(auto tmp = rni.get_rank());
        ASSERT_THROW(auto tmp = rni.get_neuron_id(), RelearnException);
    }
}

TEST_F(RankNeuronIdTest, testNeuronRankIdEquality) {
    for (auto i = 0; i < 1000; i++) {
        const auto rank_1 = static_cast<int>(get_random_number_ranks());
        const auto id_1 = get_random_number_neurons();

        const auto rank_2 = static_cast<int>(get_random_number_ranks());
        const auto id_2 = get_random_number_neurons();

        const RankNeuronId rni_1(rank_1, id_1);
        const RankNeuronId rni_2(rank_2, id_2);

        if (rank_1 == rank_2 && id_1 == id_2) {
            ASSERT_EQ(rni_1, rni_2);
        } else {
            ASSERT_NE(rni_1, rni_2);
        }
    }
}
