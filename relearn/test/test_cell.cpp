#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/algorithm/Algorithms.h"
#include "../source/algorithm/Cells.h"
#include "../source/structure/Cell.h"
#include "../source/util/Vec3.h"

#include <algorithm>
#include <numeric>
#include <random>

template <typename AdditionalCellAttributes>
void CellTest::test_cell_size() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min_1, max_1] = get_random_simulation_box_size();
    cell.set_size(min_1, max_1);

    const auto& [res_min_1, res_max_1] = cell.get_size();

    ASSERT_EQ(min_1, res_min_1);
    ASSERT_EQ(max_1, res_max_1);

    const auto& [min_2, max_2] = get_random_simulation_box_size();
    cell.set_size(min_2, max_2);

    const auto& [res_min_2, res_max_2] = cell.get_size();

    ASSERT_EQ(min_2, res_min_2);
    ASSERT_EQ(max_2, res_max_2);

    ASSERT_EQ(cell.get_maximal_dimension_difference(), (max_2 - min_2).get_maximum());
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_position() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = get_random_simulation_box_size();
    cell.set_size(min, max);

    const auto& pos_ex_1 = get_random_position_in_box(min, max);
    cell.set_excitatory_dendrites_position(pos_ex_1);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_1, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());
    ASSERT_EQ(pos_ex_1, cell.get_dendrites_position_for(SignalType::EXCITATORY).value());

    cell.set_excitatory_dendrites_position({});
    ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());

    const auto& pos_ex_2 = get_random_position_in_box(min, max);
    cell.set_excitatory_dendrites_position(pos_ex_2);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_2, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());
    ASSERT_EQ(pos_ex_2, cell.get_dendrites_position_for(SignalType::EXCITATORY).value());

    cell.set_excitatory_dendrites_position({});
    ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());

    const auto& pos_in_1 = get_random_position_in_box(min, max);
    cell.set_inhibitory_dendrites_position(pos_in_1);

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_1, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::INHIBITORY).has_value());
    ASSERT_EQ(pos_in_1, cell.get_dendrites_position_for(SignalType::INHIBITORY).value());

    cell.set_inhibitory_dendrites_position({});
    ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());

    const auto& pos_in_2 = get_random_position_in_box(min, max);
    cell.set_inhibitory_dendrites_position(pos_in_2);

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_2, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::INHIBITORY).has_value());
    ASSERT_EQ(pos_in_2, cell.get_dendrites_position_for(SignalType::INHIBITORY).value());

    cell.set_inhibitory_dendrites_position({});
    ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_FALSE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_position_exception() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = get_random_simulation_box_size();
    cell.set_size(min, max);

    const auto& pos_ex_1 = get_random_position_in_box(min, max);
    cell.set_excitatory_dendrites_position(pos_ex_1);

    const auto& pos_ex_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_ex_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_ex_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_z_max), RelearnException);

    const auto& pos_ex_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_ex_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_ex_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_1, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());
    ASSERT_EQ(pos_ex_1, cell.get_dendrites_position_for(SignalType::EXCITATORY).value());

    const auto& pos_ex_2 = get_random_position_in_box(min, max);
    cell.set_excitatory_dendrites_position(pos_ex_2);

    const auto& pos_ex_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_ex_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_ex_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_z_max), RelearnException);

    const auto& pos_ex_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_ex_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_ex_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
    ASSERT_EQ(pos_ex_2, cell.get_excitatory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::EXCITATORY).has_value());
    ASSERT_EQ(pos_ex_2, cell.get_dendrites_position_for(SignalType::EXCITATORY).value());

    const auto& pos_in_1 = get_random_position_in_box(min, max);
    cell.set_inhibitory_dendrites_position(pos_in_1);

    const auto& pos_in_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_in_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_in_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_z_max), RelearnException);

    const auto& pos_in_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_in_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_in_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_1, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::INHIBITORY).has_value());
    ASSERT_EQ(pos_in_1, cell.get_dendrites_position_for(SignalType::INHIBITORY).value());

    const auto& pos_in_2 = get_random_position_in_box(min, max);
    cell.set_inhibitory_dendrites_position(pos_in_2);

    const auto& pos_in_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_in_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_in_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_x_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_y_max), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_z_max), RelearnException);

    const auto& pos_in_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_in_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_in_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_x_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_y_min), RelearnException);
    ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_z_min), RelearnException);

    ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
    ASSERT_EQ(pos_in_2, cell.get_inhibitory_dendrites_position().value());

    ASSERT_TRUE(cell.get_dendrites_position_for(SignalType::INHIBITORY).has_value());
    ASSERT_EQ(pos_in_2, cell.get_dendrites_position_for(SignalType::INHIBITORY).value());
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_position_combined() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = get_random_simulation_box_size();
    cell.set_size(min, max);

    const auto& pos_1 = get_random_position_in_box(min, max);
    const auto& pos_2 = get_random_position_in_box(min, max);
    const auto& pos_3 = get_random_position_in_box(min, max);
    const auto& pos_4 = get_random_position_in_box(min, max);

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

template <typename AdditionalCellAttributes>
void CellTest::test_cell_set_number_dendrites() {
    Cell<AdditionalCellAttributes> cell{};

    const auto num_dends_ex_1 = get_random_number_neurons();
    const auto num_dends_in_1 = get_random_number_neurons();

    cell.set_number_excitatory_dendrites(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_ex_1));
    cell.set_number_inhibitory_dendrites(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_in_1));

    ASSERT_EQ(num_dends_ex_1, cell.get_number_excitatory_dendrites());
    ASSERT_EQ(num_dends_ex_1, cell.get_number_dendrites_for(SignalType::EXCITATORY));
    ASSERT_EQ(num_dends_in_1, cell.get_number_inhibitory_dendrites());
    ASSERT_EQ(num_dends_in_1, cell.get_number_dendrites_for(SignalType::INHIBITORY));

    const auto num_dends_ex_2 = get_random_number_neurons();
    const auto num_dends_in_2 = get_random_number_neurons();

    cell.set_number_excitatory_dendrites(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_ex_2));
    cell.set_number_inhibitory_dendrites(static_cast<typename Cell<AdditionalCellAttributes>::counter_type>(num_dends_in_2));

    ASSERT_EQ(num_dends_ex_2, cell.get_number_excitatory_dendrites());
    ASSERT_EQ(num_dends_ex_2, cell.get_number_dendrites_for(SignalType::EXCITATORY));
    ASSERT_EQ(num_dends_in_2, cell.get_number_inhibitory_dendrites());
    ASSERT_EQ(num_dends_in_2, cell.get_number_dendrites_for(SignalType::INHIBITORY));
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_set_neuron_id() {
    Cell<AdditionalCellAttributes> cell{};

    const auto neuron_id_1 = get_random_number_neurons();
    const auto id1 = NeuronID{ neuron_id_1 };
    cell.set_neuron_id(id1);
    ASSERT_EQ(id1, cell.get_neuron_id());

    const auto neuron_id_2 = get_random_number_neurons();
    const auto id2 = NeuronID{ neuron_id_2 };
    cell.set_neuron_id(id2);
    ASSERT_EQ(id2, cell.get_neuron_id());
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_octants() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = get_random_simulation_box_size();
    cell.set_size(min, max);

    const auto midpoint = (min + max) / 2;

    for (auto id = 0; id < 1000; id++) {
        const auto& position = get_random_position_in_box(min, max);

        const auto larger_x = position.get_x() >= midpoint.get_x() ? 1 : 0;
        const auto larger_y = position.get_y() >= midpoint.get_y() ? 2 : 0;
        const auto larger_z = position.get_z() >= midpoint.get_z() ? 4 : 0;

        const auto expected_octant_idx = larger_x + larger_y + larger_z;

        const auto received_idx = cell.get_octant_for_position(position);

        ASSERT_EQ(expected_octant_idx, received_idx);
    }
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_octants_exception() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = get_random_simulation_box_size();
    cell.set_size(min, max);

    const auto& pos_invalid_x_max = max + Vec3d{ 1, 0, 0 };
    const auto& pos_invalid_y_max = max + Vec3d{ 0, 1, 0 };
    const auto& pos_invalid_z_max = max + Vec3d{ 0, 0, 1 };

    const auto& pos_invalid_x_min = min - Vec3d{ 1, 0, 0 };
    const auto& pos_invalid_y_min = min - Vec3d{ 0, 1, 0 };
    const auto& pos_invalid_z_min = min - Vec3d{ 0, 0, 1 };

    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_x_max), RelearnException);
    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_y_max), RelearnException);
    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_z_max), RelearnException);
    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_x_min), RelearnException);
    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_y_min), RelearnException);
    ASSERT_THROW(auto tmp = cell.get_octant_for_position(pos_invalid_z_min), RelearnException);
}

template <typename AdditionalCellAttributes>
void CellTest::test_cell_octants_size() {
    Cell<AdditionalCellAttributes> cell{};

    const auto& [min, max] = get_random_simulation_box_size();
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

        const auto& [subcell_received_min, subcell_received_max] = cell.get_size_for_octant(id);

        const auto diff_subcell_min = subcell_min - subcell_received_min;
        const auto diff_subcell_max = subcell_max - subcell_received_max;

        ASSERT_NEAR(diff_subcell_min.calculate_p_norm(2), 0.0, eps);
        ASSERT_NEAR(diff_subcell_max.calculate_p_norm(2), 0.0, eps);
    }
}

template <typename VirtualPlasticityElement>
void CellTest::test_vpe_number_elements() {
    VirtualPlasticityElement vpe{};

    const auto& number_initially_free_elements = vpe.get_number_free_elements();
    ASSERT_EQ(number_initially_free_elements, 0) << number_initially_free_elements;

    const auto nfe_1 = get_random_number_neurons();
    vpe.set_number_free_elements(static_cast<typename VirtualPlasticityElement::counter_type>(nfe_1));

    const auto& number_free_elements_1 = vpe.get_number_free_elements();
    ASSERT_EQ(number_free_elements_1, nfe_1) << number_free_elements_1 << ' ' << nfe_1;

    const auto nfe_2 = get_random_number_neurons();
    vpe.set_number_free_elements(static_cast<typename VirtualPlasticityElement::counter_type>(nfe_2));

    const auto& number_free_elements_2 = vpe.get_number_free_elements();
    ASSERT_EQ(number_free_elements_2, nfe_2) << number_free_elements_2 << ' ' << nfe_2;
}

template <typename VirtualPlasticityElement>
void CellTest::test_vpe_position() {
    VirtualPlasticityElement vpe{};

    const auto& initial_position = vpe.get_position();
    ASSERT_FALSE(initial_position.has_value());

    const auto& [pos_1, pos_3] = get_random_simulation_box_size();

    vpe.set_position(pos_1);
    const auto& position_1 = vpe.get_position();
    ASSERT_TRUE(position_1.has_value());
    ASSERT_EQ(position_1.value(), pos_1);

    vpe.set_position({});
    const auto& position_2 = vpe.get_position();
    ASSERT_FALSE(position_2.has_value());

    vpe.set_position(pos_3);
    const auto& position_3 = vpe.get_position();
    ASSERT_TRUE(position_3.has_value());
    ASSERT_EQ(position_3.value(), pos_3);
}

template <typename VirtualPlasticityElement>
void CellTest::test_vpe_mixed() {
    VirtualPlasticityElement vpe{};

    const auto& initial_position = vpe.get_position();
    ASSERT_FALSE(initial_position.has_value());

    const auto& [pos_1, pos_3] = get_random_simulation_box_size();

    vpe.set_position(pos_1);
    const auto& position_1 = vpe.get_position();
    ASSERT_TRUE(position_1.has_value());
    ASSERT_EQ(position_1.value(), pos_1);

    const auto& number_initially_free_elements = vpe.get_number_free_elements();
    ASSERT_EQ(number_initially_free_elements, 0) << number_initially_free_elements;

    const auto nfe_1 = get_random_number_neurons();
    vpe.set_number_free_elements(nfe_1);

    const auto& number_free_elements_1 = vpe.get_number_free_elements();
    ASSERT_EQ(number_free_elements_1, nfe_1) << number_free_elements_1 << ' ' << nfe_1;

    vpe.set_position({});
    const auto& position_2 = vpe.get_position();
    ASSERT_FALSE(position_2.has_value());

    vpe.set_position(pos_3);
    const auto& position_3 = vpe.get_position();
    ASSERT_TRUE(position_3.has_value());
    ASSERT_EQ(position_3.value(), pos_3);

    const auto nfe_2 = get_random_number_neurons();
    vpe.set_number_free_elements(nfe_2);

    const auto& number_free_elements_2 = vpe.get_number_free_elements();
    ASSERT_EQ(number_free_elements_2, nfe_2) << number_free_elements_2 << ' ' << nfe_2;
}

TEST_F(CellTest, testBarnesHutCellSize) {
    test_cell_size<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellPosition) {
    test_cell_position<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellPositionException) {
    test_cell_position_exception<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellPositionCombined) {
    test_cell_position_combined<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellSetNumDendrites) {
    test_cell_set_number_dendrites<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellSetNeuronId) {
    test_cell_set_neuron_id<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellOctants) {
    test_cell_octants<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellOctantsException) {
    test_cell_octants_exception<BarnesHutCell>();
}

TEST_F(CellTest, testBarnesHutCellOctantsSize) {
    test_cell_octants_size<BarnesHutCell>();
}

TEST_F(CellTest, testVPEManualNumberFreeElements) {
    test_vpe_number_elements<VirtualPlasticityElementManual>();
}

TEST_F(CellTest, testVPEManualPosition) {
    test_vpe_position<VirtualPlasticityElementManual>();
}

TEST_F(CellTest, testVPEOptionalNumberFreeElements) {
    test_vpe_number_elements<VirtualPlasticityElementOptional>();
}

TEST_F(CellTest, testVPEOptionalPosition) {
    test_vpe_position<VirtualPlasticityElementOptional>();
}
