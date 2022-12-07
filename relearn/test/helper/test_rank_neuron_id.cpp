/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_rank_neuron_id.h"

#include "neurons/helper/RankNeuronId.h"

TEST_F(RankNeuronIdTest, testNeuronRankIdValid) {
    for (auto i = 0; i < 1000; i++) {
        const auto rank = static_cast<int>(get_random_number_ranks());
        const auto id = get_random_number_neurons();

        const RankNeuronId rni{ rank, NeuronID{ id } };

        ASSERT_EQ(rni.get_neuron_id(), NeuronID{ id });
        ASSERT_EQ(rni.get_rank(), rank);
    }
}

TEST_F(RankNeuronIdTest, testNeuronRankIdInvalidRank) {
    for (auto i = 0; i < 1000; i++) {
        const auto rank = get_random_integer<int>(-1000, -1);
        const auto id = get_random_number_neurons();

        ASSERT_THROW(RankNeuronId rni(rank, NeuronID{ id });, RelearnException);
    }
}

TEST_F(RankNeuronIdTest, testNeuronRankIdInvalidId) {
    for (auto i = 0; i < 1000; i++) {
        const auto rank = static_cast<int>(get_random_number_ranks());

        RankNeuronId rni(rank, NeuronID::uninitialized_id());

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

        const RankNeuronId rni_1(rank_1, NeuronID{ id_1 });
        const RankNeuronId rni_2(rank_2, NeuronID{ id_2 });

        if (rank_1 == rank_2 && id_1 == id_2) {
            ASSERT_EQ(rni_1, rni_2);
        } else {
            ASSERT_NE(rni_1, rni_2);
        }
    }
}
