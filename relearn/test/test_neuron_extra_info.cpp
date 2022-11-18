#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/neurons/NeuronsExtraInfo.h"

void NeuronsTest::assert_empty(const NeuronsExtraInfo& nei, size_t number_neurons) {
    const auto& area_ids = nei.get_neuron_id_vs_area_id();
    const auto& positions = nei.get_positions();

    const auto& area_size = area_ids.size();
    const auto& positions_size = positions.size();

    ASSERT_EQ(0, area_size) << area_size;
    ASSERT_EQ(0, positions_size) << positions_size;

    for (auto i = 0; i < number_neurons_out_of_scope; i++) {
        const auto neuron_id = get_random_neuron_id(number_neurons, 1);

        ASSERT_THROW(const auto& tmp = nei.get_area_id_for_neuron_id(neuron_id), RelearnException) << "assert empty area name" << neuron_id;
        ASSERT_THROW(const auto& tmp = nei.get_position(neuron_id), RelearnException) << "assert empty position" << neuron_id;
    }
}

void NeuronsTest::assert_contains(const NeuronsExtraInfo& nei, size_t number_neurons, size_t num_neurons_check,
    const std::vector<RelearnTypes::area_id>& expected_area_ids, const std::vector<Vec3d>& expected_positions) {

    const auto& expected_area_names_size = expected_area_ids.size();
    const auto& expected_positions_size = expected_positions.size();

    ASSERT_EQ(num_neurons_check, expected_area_names_size) << num_neurons_check << ' ' << expected_area_names_size;
    ASSERT_EQ(num_neurons_check, expected_positions_size) << num_neurons_check << ' ' << expected_positions_size;

    const auto& actual_area_names = nei.get_neuron_id_vs_area_id();
    const auto& actual_positions = nei.get_positions();

    const auto& area_names_size = actual_area_names.size();
    const auto& positions_size = actual_positions.size();

    ASSERT_EQ(area_names_size, number_neurons) << area_names_size << ' ' << number_neurons;
    ASSERT_EQ(positions_size, number_neurons) << positions_size << ' ' << number_neurons;

    for (auto neuron_id : NeuronID::range(num_neurons_check)) {
        ASSERT_EQ(expected_area_ids[neuron_id.get_neuron_id()], actual_area_names[neuron_id.get_neuron_id()]) << neuron_id;
        ASSERT_EQ(expected_area_ids[neuron_id.get_neuron_id()], nei.get_area_id_for_neuron_id(neuron_id)) << neuron_id;

        ASSERT_EQ(expected_positions[neuron_id.get_neuron_id()], actual_positions[neuron_id.get_neuron_id()]) << neuron_id;
        ASSERT_EQ(expected_positions[neuron_id.get_neuron_id()], nei.get_position(neuron_id)) << neuron_id;
    }

    for (auto i = 0; i < number_neurons_out_of_scope; i++) {
        const auto neuron_id = get_random_neuron_id(number_neurons, number_neurons);

        ASSERT_THROW(const auto& tmp = nei.get_position(neuron_id), RelearnException) << neuron_id;
        ASSERT_THROW(const auto& tmp = nei.get_area_id_for_neuron_id(neuron_id), RelearnException) << neuron_id;
    }
}

TEST_F(NeuronsTest, testNeuronsExtraInfo) {
    NeuronsExtraInfo nei{};

    assert_empty(nei, upper_bound_num_neurons);

    ASSERT_THROW(nei.set_area_id_vs_area_name({}), RelearnException);
    ASSERT_THROW(nei.set_neuron_id_vs_area_id({}), RelearnException);
    ASSERT_THROW(nei.set_positions(std::vector<NeuronsExtraInfo::position_type>{}), RelearnException);

    assert_empty(nei, upper_bound_num_neurons);

    const auto new_size = get_random_number_neurons();

    nei.set_area_id_vs_area_name(get_random_area_names(new_size));
    ASSERT_THROW(nei.set_neuron_id_vs_area_id({}), RelearnException);
    ASSERT_THROW(nei.set_neuron_id_vs_area_id(std::vector<RelearnTypes::area_id>(new_size)), RelearnException);
    ASSERT_THROW(nei.set_positions(std::vector<NeuronsExtraInfo::position_type>(new_size)), RelearnException);

    assert_empty(nei, upper_bound_num_neurons);
}

TEST_F(NeuronsTest, testAreaIdsBeforeNames) {
    NeuronsExtraInfo nei{};
    const auto number_neurons = get_random_number_neurons();

    nei.init(number_neurons);
    assert_empty(nei, number_neurons);

    const auto& area_names = get_random_area_names(number_neurons);
    const auto& area_ids = get_random_area_ids(area_names.size(), number_neurons);
    ASSERT_THROW(nei.set_neuron_id_vs_area_id(area_ids), RelearnException);
    nei.set_area_id_vs_area_name(area_names);
    nei.set_neuron_id_vs_area_id(area_ids);
}

TEST_F(NeuronsTest, testNeuronsExtraInfoInit) {
    NeuronsExtraInfo nei{};

    const auto number_neurons = get_random_number_neurons();

    nei.init(number_neurons);
    assert_empty(nei, number_neurons);

    auto num_neurons_wrong = get_random_number_neurons();
    if (num_neurons_wrong == number_neurons) {
        num_neurons_wrong++;
    }

    const auto& area_names = get_random_area_names(number_neurons);

    std::vector<Vec3d> positions_wrong(num_neurons_wrong);
    std::vector<RelearnTypes::area_id> area_ids_wrong(num_neurons_wrong);

    ASSERT_THROW(nei.set_positions(positions_wrong), RelearnException);
    nei.set_area_id_vs_area_name(area_names);
    ASSERT_THROW(nei.set_neuron_id_vs_area_id(area_ids_wrong), RelearnException);

    assert_empty(nei, number_neurons);

    std::vector<Vec3d> positions_right(number_neurons);
    std::vector<RelearnTypes::area_id> area_ids_right = get_random_area_ids(area_names.size(), number_neurons);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        positions_right[neuron_id] = get_random_position();
    }

    nei.set_positions(positions_right);
    nei.set_neuron_id_vs_area_id(area_ids_right);

    assert_contains(nei, number_neurons, number_neurons, area_ids_right, positions_right);

    std::vector<Vec3d> positions_right_2(number_neurons);
    std::vector<RelearnTypes::area_id> area_ids_right_2 = get_random_area_ids(area_names.size(), number_neurons);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        positions_right_2[neuron_id] = get_random_position();
    }

    nei.set_positions(positions_right_2);
    nei.set_neuron_id_vs_area_id(area_ids_right_2);

    assert_contains(nei, number_neurons, number_neurons, area_ids_right_2, positions_right_2);
}

TEST_F(NeuronsTest, testNeuronsExtraInfoCreate) {
    NeuronsExtraInfo nei{};

    const auto num_neurons_init = get_random_number_neurons();
    const auto num_neurons_create_1 = get_random_number_neurons();
    const auto num_neurons_create_2 = get_random_number_neurons();

    const auto num_neurons_total_1 = num_neurons_init + num_neurons_create_1;
    const auto num_neurons_total_2 = num_neurons_total_1 + num_neurons_create_2;

    nei.init(num_neurons_init);

    ASSERT_THROW(nei.create_neurons(num_neurons_create_1), RelearnException);

    assert_empty(nei, num_neurons_init);

    std::vector<Vec3d> positions_right(num_neurons_init);
    std::vector<RelearnTypes::area_name> area_names = get_random_area_names(num_neurons_init);
    std::vector<RelearnTypes::area_id> area_ids_right = get_random_area_ids(area_names.size(), num_neurons_init);

    for (auto neuron_id = 0; neuron_id < num_neurons_init; neuron_id++) {
        positions_right[neuron_id] = get_random_position();
    }

    nei.set_positions(positions_right);
    nei.set_area_id_vs_area_name(area_names);
    nei.set_neuron_id_vs_area_id(area_ids_right);

    nei.create_neurons(num_neurons_create_1);

    assert_contains(nei, num_neurons_total_1, num_neurons_init, area_ids_right, positions_right);

    std::vector<Vec3d> positions_right_2(num_neurons_total_1);
    std::vector<RelearnTypes::area_id> area_ids_right_2 = get_random_area_ids(area_names.size(), num_neurons_total_1);

    for (auto neuron_id = 0; neuron_id < num_neurons_total_1; neuron_id++) {
        positions_right_2[neuron_id] = get_random_position();
    }

    nei.set_positions(positions_right_2);
    nei.set_neuron_id_vs_area_id(area_ids_right_2);

    assert_contains(nei, num_neurons_total_1, num_neurons_total_1, area_ids_right_2, positions_right_2);

    nei.create_neurons(num_neurons_create_2);

    assert_contains(nei, num_neurons_total_2, num_neurons_total_1, area_ids_right_2, positions_right_2);

    std::vector<Vec3d> positions_right_3(num_neurons_total_2);
    std::vector<RelearnTypes::area_id> area_ids_right_3 = get_random_area_ids(area_names.size(), num_neurons_total_2);

    for (auto neuron_id = 0; neuron_id < num_neurons_total_2; neuron_id++) {
        positions_right_3[neuron_id] = get_random_position();
    }

    nei.set_positions(positions_right_3);
    nei.set_neuron_id_vs_area_id(area_ids_right_3);

    assert_contains(nei, num_neurons_total_2, num_neurons_total_2, area_ids_right_3, positions_right_3);
}
