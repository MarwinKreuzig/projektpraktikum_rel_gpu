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

void assert_empty(const NeuronsExtraInfo& nei, size_t num_neurons) {
    const auto& x_dims = nei.get_x_dims();
    const auto& y_dims = nei.get_y_dims();
    const auto& z_dims = nei.get_z_dims();
    const auto& area_names = nei.get_area_names();

    ASSERT_EQ(0, x_dims.size());
    ASSERT_EQ(0, y_dims.size());
    ASSERT_EQ(0, z_dims.size());
    ASSERT_EQ(0, area_names.size());

    for (auto neuron_id = 0; neuron_id < num_neurons * 2; neuron_id++) {
        ASSERT_THROW(nei.get_position(neuron_id), RelearnException);
        ASSERT_THROW(nei.get_x(neuron_id), RelearnException);
        ASSERT_THROW(nei.get_y(neuron_id), RelearnException);
        ASSERT_THROW(nei.get_z(neuron_id), RelearnException);
        ASSERT_THROW(nei.get_area_name(neuron_id), RelearnException);
    }
}

void assert_contains(const NeuronsExtraInfo& nei, size_t num_neurons, size_t num_neurons_check, const std::vector<std::string>& expected_area_names,
    const std::vector<double>& expected_x_dims, const std::vector<double>& expected_y_dims, const std::vector<double>& expected_z_dims) {

    ASSERT_EQ(num_neurons_check, expected_area_names.size());
    ASSERT_EQ(num_neurons_check, expected_x_dims.size());
    ASSERT_EQ(num_neurons_check, expected_y_dims.size());
    ASSERT_EQ(num_neurons_check, expected_z_dims.size());

    const auto& actual_x_dims = nei.get_x_dims();
    const auto& actual_y_dims = nei.get_y_dims();
    const auto& actual_z_dims = nei.get_z_dims();
    const auto& actual_area_names = nei.get_area_names();

    ASSERT_EQ(actual_x_dims.size(), num_neurons);
    ASSERT_EQ(actual_y_dims.size(), num_neurons);
    ASSERT_EQ(actual_z_dims.size(), num_neurons);
    ASSERT_EQ(actual_area_names.size(), num_neurons);

    for (auto neuron_id = 0; neuron_id < num_neurons_check; neuron_id++) {
        ASSERT_EQ(expected_x_dims[neuron_id], actual_x_dims[neuron_id]);
        ASSERT_EQ(expected_x_dims[neuron_id], nei.get_x(neuron_id));

        ASSERT_EQ(expected_y_dims[neuron_id], actual_y_dims[neuron_id]);
        ASSERT_EQ(expected_y_dims[neuron_id], nei.get_y(neuron_id));

        ASSERT_EQ(expected_z_dims[neuron_id], actual_z_dims[neuron_id]);
        ASSERT_EQ(expected_z_dims[neuron_id], nei.get_z(neuron_id));

        ASSERT_EQ(expected_area_names[neuron_id], actual_area_names[neuron_id]);
        ASSERT_EQ(expected_area_names[neuron_id], nei.get_area_name(neuron_id));

        Vec3d pos{ expected_x_dims[neuron_id], expected_y_dims[neuron_id], expected_z_dims[neuron_id] };
        ASSERT_EQ(nei.get_position(neuron_id), pos);
    }

    for (auto neuron_id = num_neurons; neuron_id < num_neurons * 2; neuron_id++) {
        ASSERT_THROW(nei.get_position(neuron_id), RelearnException);
        ASSERT_THROW(nei.get_x(neuron_id), RelearnException);
        ASSERT_THROW(nei.get_y(neuron_id), RelearnException);
        ASSERT_THROW(nei.get_z(neuron_id), RelearnException);
        ASSERT_THROW(nei.get_area_name(neuron_id), RelearnException);
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

            ASSERT_NO_THROW(rni.get_neuron_id());
            ASSERT_THROW(rni.get_rank(), RelearnException);
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

            ASSERT_NO_THROW(rni.get_rank());
            ASSERT_THROW(rni.get_neuron_id(), RelearnException);
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
        ASSERT_THROW(nei.set_x_dims(std::vector<double>{}), RelearnException);
        ASSERT_THROW(nei.set_y_dims(std::vector<double>{}), RelearnException);
        ASSERT_THROW(nei.set_z_dims(std::vector<double>{}), RelearnException);

        assert_empty(nei, 1000);

        const auto new_size = uid_id(mt);

        ASSERT_THROW(nei.set_area_names(std::vector<std::string>(new_size)), RelearnException);
        ASSERT_THROW(nei.set_x_dims(std::vector<double>(new_size)), RelearnException);
        ASSERT_THROW(nei.set_y_dims(std::vector<double>(new_size)), RelearnException);
        ASSERT_THROW(nei.set_z_dims(std::vector<double>(new_size)), RelearnException);

        assert_empty(nei, 1000);
    }
}

TEST_F(NeuronsTest, testNeuronsExtraInfoInit) {
    std::uniform_int_distribution<size_t> uid_id(0, 10000);
    std::uniform_real_distribution<double> urd_pos(-10000.0, 10000.0);

    for (auto it = 0; it < iterations; it++) {
        NeuronsExtraInfo nei{};

        const auto num_neurons = uid_id(mt);

        nei.init(num_neurons);
        assert_empty(nei, num_neurons);

        auto num_neurons_wrong = uid_id(mt);
        if (num_neurons_wrong == num_neurons) {
            num_neurons_wrong++;
        }

        std::vector<double> x_dims_wrong(num_neurons_wrong);
        std::vector<double> y_dims_wrong(num_neurons_wrong);
        std::vector<double> z_dims_wrong(num_neurons_wrong);
        std::vector<std::string> area_names_wrong(num_neurons_wrong);

        ASSERT_THROW(nei.set_x_dims(x_dims_wrong), RelearnException);
        ASSERT_THROW(nei.set_y_dims(y_dims_wrong), RelearnException);
        ASSERT_THROW(nei.set_z_dims(z_dims_wrong), RelearnException);
        ASSERT_THROW(nei.set_area_names(area_names_wrong), RelearnException);

        assert_empty(nei, num_neurons);

        std::vector<double> x_dims_right(num_neurons);
        std::vector<double> y_dims_right(num_neurons);
        std::vector<double> z_dims_right(num_neurons);
        std::vector<std::string> area_names_right(num_neurons);

        for (auto neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            x_dims_right[neuron_id] = urd_pos(mt);
            y_dims_right[neuron_id] = urd_pos(mt);
            z_dims_right[neuron_id] = urd_pos(mt);

            area_names_right[neuron_id] = std::to_string(urd_pos(mt));
        }

        nei.set_x_dims(x_dims_right);
        nei.set_y_dims(y_dims_right);
        nei.set_z_dims(z_dims_right);
        nei.set_area_names(area_names_right);

        assert_contains(nei, num_neurons, num_neurons, area_names_right, x_dims_right, y_dims_right, z_dims_right);

        std::vector<double> x_dims_right_2(num_neurons);
        std::vector<double> y_dims_right_2(num_neurons);
        std::vector<double> z_dims_right_2(num_neurons);
        std::vector<std::string> area_names_right_2(num_neurons);

        for (auto neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            x_dims_right_2[neuron_id] = urd_pos(mt);
            y_dims_right_2[neuron_id] = urd_pos(mt);
            z_dims_right_2[neuron_id] = urd_pos(mt);

            area_names_right_2[neuron_id] = std::to_string(urd_pos(mt));
        }

        nei.set_x_dims(x_dims_right_2);
        nei.set_y_dims(y_dims_right_2);
        nei.set_z_dims(z_dims_right_2);
        nei.set_area_names(area_names_right_2);

        assert_contains(nei, num_neurons, num_neurons, area_names_right_2, x_dims_right_2, y_dims_right_2, z_dims_right_2);
    }
}

TEST_F(NeuronsTest, testNeuronsExtraInfoCreate) {

    std::uniform_int_distribution<size_t> uid_id(0, 10000);
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

        std::vector<double> x_dims_right(num_neurons_init);
        std::vector<double> y_dims_right(num_neurons_init);
        std::vector<double> z_dims_right(num_neurons_init);
        std::vector<std::string> area_names_right(num_neurons_init);

        for (auto neuron_id = 0; neuron_id < num_neurons_init; neuron_id++) {
            x_dims_right[neuron_id] = urd_pos(mt);
            y_dims_right[neuron_id] = urd_pos(mt);
            z_dims_right[neuron_id] = urd_pos(mt);

            area_names_right[neuron_id] = std::to_string(urd_pos(mt));
        }

        nei.set_x_dims(x_dims_right);
        nei.set_y_dims(y_dims_right);
        nei.set_z_dims(z_dims_right);
        nei.set_area_names(area_names_right);

        nei.create_neurons(num_neurons_create_1);

        assert_contains(nei, num_neurons_total_1, num_neurons_init, area_names_right, x_dims_right, y_dims_right, z_dims_right);

        std::vector<double> x_dims_right_2(num_neurons_total_1);
        std::vector<double> y_dims_right_2(num_neurons_total_1);
        std::vector<double> z_dims_right_2(num_neurons_total_1);
        std::vector<std::string> area_names_right_2(num_neurons_total_1);

        for (auto neuron_id = 0; neuron_id < num_neurons_total_1; neuron_id++) {
            x_dims_right_2[neuron_id] = urd_pos(mt);
            y_dims_right_2[neuron_id] = urd_pos(mt);
            z_dims_right_2[neuron_id] = urd_pos(mt);

            area_names_right_2[neuron_id] = std::to_string(urd_pos(mt));
        }

        nei.set_x_dims(x_dims_right_2);
        nei.set_y_dims(y_dims_right_2);
        nei.set_z_dims(z_dims_right_2);
        nei.set_area_names(area_names_right_2);

        assert_contains(nei, num_neurons_total_1, num_neurons_total_1, area_names_right_2, x_dims_right_2, y_dims_right_2, z_dims_right_2);

        nei.create_neurons(num_neurons_create_2);

        assert_contains(nei, num_neurons_total_2, num_neurons_total_1, area_names_right_2, x_dims_right_2, y_dims_right_2, z_dims_right_2);

        std::vector<double> x_dims_right_3(num_neurons_total_2);
        std::vector<double> y_dims_right_3(num_neurons_total_2);
        std::vector<double> z_dims_right_3(num_neurons_total_2);
        std::vector<std::string> area_names_right_3(num_neurons_total_2);

        for (auto neuron_id = 0; neuron_id < num_neurons_total_2; neuron_id++) {
            x_dims_right_3[neuron_id] = urd_pos(mt);
            y_dims_right_3[neuron_id] = urd_pos(mt);
            z_dims_right_3[neuron_id] = urd_pos(mt);

            area_names_right_3[neuron_id] = std::to_string(urd_pos(mt));
        }

        nei.set_x_dims(x_dims_right_3);
        nei.set_y_dims(y_dims_right_3);
        nei.set_z_dims(z_dims_right_3);
        nei.set_area_names(area_names_right_3);

        assert_contains(nei, num_neurons_total_2, num_neurons_total_2, area_names_right_3, x_dims_right_3, y_dims_right_3, z_dims_right_3);
    }
}
