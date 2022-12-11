#include "test_neuron_extra_info.h"

#include "simulation/simulation_adapter.h"
#include "tagged_id/tagged_id_adapter.h"

#include "neurons/NeuronsExtraInfo.h"

void NeuronsExtraInfoTest::assert_empty(const NeuronsExtraInfo& nei, size_t number_neurons) {
    const auto& positions = nei.get_positions();

    const auto& positions_size = positions.size();

    ASSERT_EQ(0, positions_size) << positions_size;

    for (auto i = 0; i < number_neurons_out_of_scope; i++) {
        const auto neuron_id = TaggedIdAdapter::get_random_neuron_id(number_neurons, 1, mt);

        ASSERT_THROW(const auto& tmp = nei.get_position(neuron_id), RelearnException) << "assert empty position" << neuron_id;
    }
}

void NeuronsExtraInfoTest::assert_contains(const NeuronsExtraInfo& nei, size_t number_neurons, size_t num_neurons_check, const std::vector<Vec3d>& expected_positions) {

    const auto& expected_positions_size = expected_positions.size();

    ASSERT_EQ(num_neurons_check, expected_positions_size) << num_neurons_check << ' ' << expected_positions_size;

    const auto& actual_positions = nei.get_positions();

    const auto& positions_size = actual_positions.size();

    ASSERT_EQ(positions_size, number_neurons) << positions_size << ' ' << number_neurons;

    for (auto neuron_id : NeuronID::range(num_neurons_check)) {

        ASSERT_EQ(expected_positions[neuron_id.get_neuron_id()], actual_positions[neuron_id.get_neuron_id()]) << neuron_id;
        ASSERT_EQ(expected_positions[neuron_id.get_neuron_id()], nei.get_position(neuron_id)) << neuron_id;
    }

    for (auto i = 0; i < number_neurons_out_of_scope; i++) {
        const auto neuron_id = TaggedIdAdapter::get_random_neuron_id(number_neurons, number_neurons, mt);

        ASSERT_THROW(const auto& tmp = nei.get_position(neuron_id), RelearnException) << neuron_id;
    }
}

TEST_F(NeuronsExtraInfoTest, testNeuronsExtraInfo) {
    NeuronsExtraInfo nei{};

    assert_empty(nei, TaggedIdAdapter::upper_bound_num_neurons);

    ASSERT_THROW(nei.set_positions(std::vector<NeuronsExtraInfo::position_type>{}), RelearnException);

    assert_empty(nei, TaggedIdAdapter::upper_bound_num_neurons);

    const auto new_size = TaggedIdAdapter::get_random_number_neurons(mt);

    ASSERT_THROW(nei.set_positions(std::vector<NeuronsExtraInfo::position_type>(new_size)), RelearnException);

    assert_empty(nei, TaggedIdAdapter::upper_bound_num_neurons);
}

TEST_F(NeuronsExtraInfoTest, testNeuronsExtraInfoInit) {
    NeuronsExtraInfo nei{};

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);

    nei.init(number_neurons);
    assert_empty(nei, number_neurons);

    auto num_neurons_wrong = TaggedIdAdapter::get_random_number_neurons(mt);
    if (num_neurons_wrong == number_neurons) {
        num_neurons_wrong++;
    }

    std::vector<Vec3d> positions_wrong(num_neurons_wrong);

    ASSERT_THROW(nei.set_positions(positions_wrong), RelearnException);

    assert_empty(nei, number_neurons);

    std::vector<Vec3d> positions_right(number_neurons);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        positions_right[neuron_id] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions_right);

    assert_contains(nei, number_neurons, number_neurons, positions_right);

    std::vector<Vec3d> positions_right_2(number_neurons);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        positions_right_2[neuron_id] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions_right_2);

    assert_contains(nei, number_neurons, number_neurons, positions_right_2);
}

TEST_F(NeuronsExtraInfoTest, testNeuronsExtraInfoCreate) {
    NeuronsExtraInfo nei{};

    const auto num_neurons_init = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto num_neurons_create_1 = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto num_neurons_create_2 = TaggedIdAdapter::get_random_number_neurons(mt);

    const auto num_neurons_total_1 = num_neurons_init + num_neurons_create_1;
    const auto num_neurons_total_2 = num_neurons_total_1 + num_neurons_create_2;

    nei.init(num_neurons_init);

    ASSERT_THROW(nei.create_neurons(num_neurons_create_1), RelearnException);

    assert_empty(nei, num_neurons_init);

    std::vector<Vec3d> positions_right(num_neurons_init);

    for (auto neuron_id = 0; neuron_id < num_neurons_init; neuron_id++) {
        positions_right[neuron_id] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions_right);

    nei.create_neurons(num_neurons_create_1);

    assert_contains(nei, num_neurons_total_1, num_neurons_init, positions_right);

    std::vector<Vec3d> positions_right_2(num_neurons_total_1);

    for (auto neuron_id = 0; neuron_id < num_neurons_total_1; neuron_id++) {
        positions_right_2[neuron_id] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions_right_2);

    assert_contains(nei, num_neurons_total_1, num_neurons_total_1, positions_right_2);

    nei.create_neurons(num_neurons_create_2);

    assert_contains(nei, num_neurons_total_2, num_neurons_total_1, positions_right_2);

    std::vector<Vec3d> positions_right_3(num_neurons_total_2);

    for (auto neuron_id = 0; neuron_id < num_neurons_total_2; neuron_id++) {
        positions_right_3[neuron_id] = SimulationAdapter::get_random_position(mt);
    }

    nei.set_positions(positions_right_3);

    assert_contains(nei, num_neurons_total_2, num_neurons_total_2, positions_right_3);
}
