#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/algorithm/BarnesHut.h"
#include "../source/structure/Cell.h"
#include "../source/util/RelearnException.h"
#include "../source/util/Vec3.h"

#include <algorithm>
#include <numeric>
#include <random>

using AdditionalCellAttributes = BarnesHutCell;

TEST_F(OctreeTest, testCellSize) {
    for (auto i = 0; i < iterations; i++) {
        Cell<AdditionalCellAttributes> cell{};

        const auto& box_sizes_1 = get_random_simulation_box_size(mt);
        const auto& min_1 = std::get<0>(box_sizes_1);
        const auto& max_1 = std::get<1>(box_sizes_1);

        cell.set_size(min_1, max_1);

        const auto& res_1 = cell.get_size();

        ASSERT_EQ(min_1, std::get<0>(res_1));
        ASSERT_EQ(max_1, std::get<1>(res_1));

        const auto& box_sizes_2 = get_random_simulation_box_size(mt);
        const auto& min_2 = std::get<0>(box_sizes_2);
        const auto& max_2 = std::get<1>(box_sizes_2);

        cell.set_size(min_2, max_2);

        const auto& res_2 = cell.get_size();

        ASSERT_EQ(min_2, std::get<0>(res_2));
        ASSERT_EQ(max_2, std::get<1>(res_2));

        ASSERT_EQ(cell.get_maximal_dimension_difference(), (max_2 - min_2).get_maximum());
    }
}

TEST_F(OctreeTest, testCellPosition) {
    for (auto i = 0; i < iterations; i++) {
        Cell<AdditionalCellAttributes> cell{};

        const auto& box_sizes = get_random_simulation_box_size(mt);
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        const Vec3d pos_ex_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_excitatory_dendrites_position(pos_ex_1);

        ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_EQ(pos_ex_1, cell.get_excitatory_dendrites_position().value());

        ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());
        ASSERT_EQ(pos_ex_1, cell.get_dendrites_position_for(SignalType::EXCITATORY).value());

        cell.set_excitatory_dendrites_position({});
        ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());

        const Vec3d pos_ex_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_excitatory_dendrites_position(pos_ex_2);

        ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_EQ(pos_ex_2, cell.get_excitatory_dendrites_position().value());

        ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());
        ASSERT_EQ(pos_ex_2, cell.get_dendrites_position_for(SignalType::EXCITATORY).value());

        cell.set_excitatory_dendrites_position({});
        ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());

        const Vec3d pos_in_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_inhibitory_dendrites_position(pos_in_1);

        ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
        ASSERT_EQ(pos_in_1, cell.get_inhibitory_dendrites_position().value());

        ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::INHIBITORY).has_value());
        ASSERT_EQ(pos_in_1, cell.get_dendrites_position_for(SignalType::INHIBITORY).value());

        cell.set_inhibitory_dendrites_position({});
        ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());

        const Vec3d pos_in_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_inhibitory_dendrites_position(pos_in_2);

        ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
        ASSERT_EQ(pos_in_2, cell.get_inhibitory_dendrites_position().value());

        ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::INHIBITORY).has_value());
        ASSERT_EQ(pos_in_2, cell.get_dendrites_position_for(SignalType::INHIBITORY).value());

        cell.set_inhibitory_dendrites_position({});
        ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());
    }
}

TEST_F(OctreeTest, testCellPositionException) {
    for (auto i = 0; i < iterations; i++) {
        Cell<AdditionalCellAttributes> cell{};

        const auto& box_sizes = get_random_simulation_box_size(mt);
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        const Vec3d pos_ex_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_excitatory_dendrites_position(pos_ex_1);

        const Vec3d pos_ex_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_x_max), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_y_max), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_z_max), RelearnException);

        const Vec3d pos_ex_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_x_min), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_y_min), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_z_min), RelearnException);

        ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_EQ(pos_ex_1, cell.get_excitatory_dendrites_position().value());

        ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());
        ASSERT_EQ(pos_ex_1, cell.get_dendrites_position_for(SignalType::EXCITATORY).value());

        const Vec3d pos_ex_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_excitatory_dendrites_position(pos_ex_2);

        const Vec3d pos_ex_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_x_max), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_y_max), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_z_max), RelearnException);

        const Vec3d pos_ex_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_x_min), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_y_min), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_z_min), RelearnException);

        ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_EQ(pos_ex_2, cell.get_excitatory_dendrites_position().value());

        ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());
        ASSERT_EQ(pos_ex_2, cell.get_dendrites_position_for(SignalType::EXCITATORY).value());

        const Vec3d pos_in_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_inhibitory_dendrites_position(pos_in_1);

        const Vec3d pos_in_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_x_max), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_y_max), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_z_max), RelearnException);

        const Vec3d pos_in_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_x_min), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_y_min), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_z_min), RelearnException);

        ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
        ASSERT_EQ(pos_in_1, cell.get_inhibitory_dendrites_position().value());

        ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::INHIBITORY).has_value());
        ASSERT_EQ(pos_in_1, cell.get_dendrites_position_for(SignalType::INHIBITORY).value());

        const Vec3d pos_in_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_inhibitory_dendrites_position(pos_in_2);

        const Vec3d pos_in_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_x_max), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_y_max), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_z_max), RelearnException);

        const Vec3d pos_in_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_x_min), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_y_min), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_z_min), RelearnException);

        ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
        ASSERT_EQ(pos_in_2, cell.get_inhibitory_dendrites_position().value());

        ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::INHIBITORY).has_value());
        ASSERT_EQ(pos_in_2, cell.get_dendrites_position_for(SignalType::INHIBITORY).value());
    }
}

TEST_F(OctreeTest, testCellPositionCombined) {
    for (auto i = 0; i < iterations; i++) {
        Cell<AdditionalCellAttributes> cell{};

        const auto& box_sizes = get_random_simulation_box_size(mt);
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        const Vec3d pos_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        const Vec3d pos_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        const Vec3d pos_3{ urd_x(mt), urd_y(mt), urd_z(mt) };
        const Vec3d pos_4{ urd_x(mt), urd_y(mt), urd_z(mt) };

        cell.set_dendrites_position({});

        ASSERT_FALSE(cell.get_dendrites_position().has_value());

        cell.set_excitatory_dendrites_position(pos_1);
        cell.set_inhibitory_dendrites_position(pos_1);

        ASSERT_TRUE(cell.get_dendrites_position().has_value());
        ASSERT_EQ(cell.get_dendrites_position().value(), pos_1);

        cell.set_excitatory_dendrites_position({});
        cell.set_inhibitory_dendrites_position({});

        ASSERT_FALSE(cell.get_dendrites_position().has_value());

        cell.set_excitatory_dendrites_position(pos_2);

        ASSERT_THROW(auto tmp = cell.get_dendrites_position(), RelearnException);

        cell.set_inhibitory_dendrites_position(pos_3);

        if (pos_2 == pos_3) {
            ASSERT_TRUE(cell.get_dendrites_position().has_value());
            ASSERT_EQ(cell.get_dendrites_position().value(), pos_2);
        } else {
            ASSERT_THROW(auto tmp = cell.get_dendrites_position(), RelearnException);
        }

        cell.set_dendrites_position({});

        ASSERT_FALSE(cell.get_dendrites_position().has_value());

        cell.set_excitatory_dendrites_position(pos_4);
        cell.set_inhibitory_dendrites_position(pos_4);

        ASSERT_TRUE(cell.get_dendrites_position().has_value());
        ASSERT_EQ(cell.get_dendrites_position().value(), pos_4);
    }
}

TEST_F(OctreeTest, testCellSetNumDendrites) {
    for (auto i = 0; i < iterations; i++) {
        Cell<AdditionalCellAttributes> cell{};

        const auto num_dends_ex_1 = get_random_number_neurons(mt);
        const auto num_dends_in_1 = get_random_number_neurons(mt);

        cell.set_number_excitatory_dendrites(num_dends_ex_1);
        cell.set_number_inhibitory_dendrites(num_dends_in_1);

        ASSERT_EQ(num_dends_ex_1, cell.get_number_excitatory_dendrites());
        ASSERT_EQ(num_dends_ex_1, cell.get_number_dendrites_for(SignalType::EXCITATORY));
        ASSERT_EQ(num_dends_in_1, cell.get_number_inhibitory_dendrites());
        ASSERT_EQ(num_dends_in_1, cell.get_number_dendrites_for(SignalType::INHIBITORY));

        const auto num_dends_ex_2 = get_random_number_neurons(mt);
        const auto num_dends_in_2 = get_random_number_neurons(mt);

        cell.set_number_excitatory_dendrites(num_dends_ex_2);
        cell.set_number_inhibitory_dendrites(num_dends_in_2);

        ASSERT_EQ(num_dends_ex_2, cell.get_number_excitatory_dendrites());
        ASSERT_EQ(num_dends_ex_2, cell.get_number_dendrites_for(SignalType::EXCITATORY));
        ASSERT_EQ(num_dends_in_2, cell.get_number_inhibitory_dendrites());
        ASSERT_EQ(num_dends_in_2, cell.get_number_dendrites_for(SignalType::INHIBITORY));
    }
}

TEST_F(OctreeTest, testCellSetNeuronId) {
    for (auto i = 0; i < iterations; i++) {
        Cell<AdditionalCellAttributes> cell{};

        const auto neuron_id_1 = get_random_number_neurons(mt);
        cell.set_neuron_id(neuron_id_1);
        ASSERT_EQ(neuron_id_1, cell.get_neuron_id());

        const auto neuron_id_2 = get_random_number_neurons(mt);
        cell.set_neuron_id(neuron_id_2);
        ASSERT_EQ(neuron_id_2, cell.get_neuron_id());
    }
}

TEST_F(OctreeTest, testCellOctants) {
    for (auto i = 0; i < iterations; i++) {
        Cell<AdditionalCellAttributes> cell{};

        const auto& box_sizes = get_random_simulation_box_size(mt);
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        const auto midpoint = (min + max) / 2;

        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        for (auto id = 0; id < 1000; id++) {
            const Vec3d position{
                urd_x(mt), urd_y(mt), urd_z(mt)
            };

            const auto larger_x = position.get_x() >= midpoint.get_x() ? 1 : 0;
            const auto larger_y = position.get_y() >= midpoint.get_y() ? 2 : 0;
            const auto larger_z = position.get_z() >= midpoint.get_z() ? 4 : 0;

            const auto expected_octant_idx = larger_x + larger_y + larger_z;

            const auto received_idx = cell.get_octant_for_position(position);

            ASSERT_EQ(expected_octant_idx, received_idx);
        }
    }
}

TEST_F(OctreeTest, testCellOctantsException) {
    for (auto i = 0; i < iterations; i++) {
        Cell<AdditionalCellAttributes> cell{};

        const auto& box_sizes = get_random_simulation_box_size(mt);
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        const Vec3d pos_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        const Vec3d pos_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_x_max), RelearnException);
        ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_y_max), RelearnException);
        ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_z_max), RelearnException);
        ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_x_min), RelearnException);
        ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_y_min), RelearnException);
        ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_z_min), RelearnException);
    }
}

TEST_F(OctreeTest, testCellOctantsSize) {
    for (auto i = 0; i < iterations; i++) {
        Cell<AdditionalCellAttributes> cell{};

        const auto& box_sizes = get_random_simulation_box_size(mt);
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        const auto midpoint = (min + max) / 2;

        for (auto id = 0; id < 8; id++) {
            const auto larger_x = ((id & 1) == 0) ? 0 : 1;
            const auto larger_y = ((id & 2) == 0) ? 0 : 1;
            const auto larger_z = ((id & 4) == 0) ? 0 : 1;

            auto subcell_min = min;
            auto subcell_max = midpoint;

            if (larger_x == 1) {
                subcell_min += Vec3d{ midpoint.get_x() - min.get_x(), 0, 0 };
                subcell_max += Vec3d{ midpoint.get_x() - min.get_x(), 0, 0 };
            }

            if (larger_y == 1) {
                subcell_min += Vec3d{ 0, midpoint.get_y() - min.get_y(), 0 };
                subcell_max += Vec3d{ 0, midpoint.get_y() - min.get_y(), 0 };
            }

            if (larger_z == 1) {
                subcell_min += Vec3d{ 0, 0, midpoint.get_z() - min.get_z() };
                subcell_max += Vec3d{ 0, 0, midpoint.get_z() - min.get_z() };
            }

            const auto& subcell_received_dims = cell.get_size_for_octant(id);
            const auto& subcell_received_min = std::get<0>(subcell_received_dims);
            const auto& subcell_received_max = std::get<1>(subcell_received_dims);

            const auto diff_subcell_min = subcell_min - subcell_received_min;
            const auto diff_subcell_max = subcell_max - subcell_received_max;

            ASSERT_NEAR(diff_subcell_min.calculate_p_norm(2), 0.0, eps);
            ASSERT_NEAR(diff_subcell_max.calculate_p_norm(2), 0.0, eps);
        }
    }
}
