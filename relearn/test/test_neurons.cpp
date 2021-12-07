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
