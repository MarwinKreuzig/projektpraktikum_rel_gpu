#include "RelearnTest.hpp"

#include "neurons/LocalAreaTranslator.h"

#include "gtest/gtest.h"

#include <algorithm>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-result"
TEST_F(LocalAreaTranslatorTest, simpleTest) {
    const auto num_neurons = get_random_number_neurons();
    const auto num_areas_max = std::min(size_t{ 50 }, num_neurons);
    auto area_id_to_area_name = get_random_area_names(num_areas_max);
    auto neuron_id_to_area_id = get_random_area_ids(area_id_to_area_name.size(), num_neurons);
    auto cp_area_id_to_area_name = std::vector<RelearnTypes::area_name>{};
    auto cp_neuron_id_to_area_id = std::vector<RelearnTypes::area_id>{};
    std::copy(area_id_to_area_name.begin(), area_id_to_area_name.end(), std::back_inserter(cp_area_id_to_area_name));
    std::copy(neuron_id_to_area_id.begin(), neuron_id_to_area_id.end(), std::back_inserter(cp_neuron_id_to_area_id));
    LocalAreaTranslator translator(area_id_to_area_name, neuron_id_to_area_id);

    ASSERT_EQ(area_id_to_area_name.size(), translator.get_number_of_areas());
    ASSERT_EQ(num_neurons, translator.get_number_neurons_in_total());

    for (auto neuron_id : NeuronID::range(num_neurons)) {
        ASSERT_EQ(cp_neuron_id_to_area_id[neuron_id.get_neuron_id()], translator.get_area_id_for_neuron_id(neuron_id.get_neuron_id()));
        ASSERT_EQ(area_id_to_area_name[cp_neuron_id_to_area_id[neuron_id.get_neuron_id()]], translator.get_area_name_for_neuron_id(neuron_id.get_neuron_id()));
    }

    for (int area_id = 0; area_id < area_id_to_area_name.size(); area_id++) {
        ASSERT_EQ(cp_area_id_to_area_name[area_id], translator.get_area_name_for_area_id(area_id));
    }
}

TEST_F(LocalAreaTranslatorTest, simpleExceptionTest) {
    const auto num_neurons = get_random_number_neurons();
    auto too_many_area_id_to_area_name = get_random_area_names_specific(num_neurons + 1);
    auto neuron_id_to_area_id = get_random_area_ids(num_neurons, num_neurons);
    auto area_id_to_area_name = get_random_area_names_specific(num_neurons);

    auto one_wrong_area_id = std::vector<RelearnTypes::area_id>{};
    std::copy(neuron_id_to_area_id.begin(), neuron_id_to_area_id.end(), std::back_inserter(one_wrong_area_id));
    auto i1 = get_random_integer(size_t{ 0 }, one_wrong_area_id.size()) - 1;
    one_wrong_area_id[i1] = num_neurons;

    auto duplicated_area_name = std::vector<RelearnTypes::area_name>{};
    std::copy(area_id_to_area_name.begin(), area_id_to_area_name.end(), std::back_inserter(duplicated_area_name));
    auto i2 = get_random_integer(size_t{ 0 }, duplicated_area_name.size()) - 1;
    size_t i3;
    do {
        i3 = get_random_integer(size_t{ 0 }, duplicated_area_name.size() - 1);
    } while (i3 == i2);
    duplicated_area_name[i2] = duplicated_area_name[i3];

    ASSERT_THROW(LocalAreaTranslator(too_many_area_id_to_area_name, neuron_id_to_area_id), RelearnException);
    ASSERT_THROW(LocalAreaTranslator(duplicated_area_name, neuron_id_to_area_id), RelearnException);
    ASSERT_THROW(LocalAreaTranslator(area_id_to_area_name, one_wrong_area_id), RelearnException);

    ASSERT_THROW(LocalAreaTranslator(std::vector<RelearnTypes::area_name>({}), neuron_id_to_area_id), RelearnException);
    ASSERT_THROW(LocalAreaTranslator(std::vector<RelearnTypes::area_name>({}), std::vector<RelearnTypes::area_id>({})), RelearnException);
    ASSERT_THROW(LocalAreaTranslator(area_id_to_area_name, std::vector<RelearnTypes::area_id>({})), RelearnException);

    LocalAreaTranslator translator(area_id_to_area_name, neuron_id_to_area_id);
    ASSERT_EQ(num_neurons, translator.get_number_of_areas());
    ASSERT_EQ(num_neurons, translator.get_number_neurons_in_total());
}

TEST_F(LocalAreaTranslatorTest, getterAreaTest) {
    auto num_neurons = get_random_number_neurons();
    auto area_id_to_area_name = get_random_area_names_specific(2);
    std::vector<RelearnTypes::area_id> neuron_id_to_area_id{};
    std::vector<RelearnTypes::neuron_id> area0{};
    std::vector<RelearnTypes::neuron_id> area1{};
    for (int i = 0; i < num_neurons; i++) {
        if (get_random_bool()) {
            neuron_id_to_area_id.emplace_back(0);
            area0.emplace_back(i);
        } else {
            neuron_id_to_area_id.emplace_back(1);
            area1.emplace_back(i);
        }
    }
    LocalAreaTranslator translator(area_id_to_area_name, neuron_id_to_area_id);

    ASSERT_EQ(area0.size(), translator.get_number_neurons_in_area(0));
    ASSERT_EQ(area1.size(), translator.get_number_neurons_in_area(1));

    auto read_area0 = translator.get_neuron_ids_in_area(0);
    auto read_area1 = translator.get_neuron_ids_in_area(1);
    for (auto i : read_area0) {
        ASSERT_TRUE(std::find(area0.begin(), area0.end(), i.get_neuron_id()) != area0.end());
    }
    for (auto i : read_area1) {
        ASSERT_TRUE(std::find(area1.begin(), area1.end(), i.get_neuron_id()) != area1.end());
    }

    auto read2_area0 = translator.get_neuron_ids_in_areas({ 0 });
    auto read2_area1 = translator.get_neuron_ids_in_areas({ 1 });
    for (auto i : read2_area0) {
        ASSERT_TRUE(std::find(area0.begin(), area0.end(), i.get_neuron_id()) != area0.end());
    }
    for (auto i : read2_area1) {
        ASSERT_TRUE(std::find(area1.begin(), area1.end(), i.get_neuron_id()) != area1.end());
    }

    auto read_all = translator.get_neuron_ids_in_areas({ 0, 1 });
    ASSERT_EQ(num_neurons, read_all.size());
}

TEST_F(LocalAreaTranslatorTest, getterExceptionTest) {
    auto num_neurons = get_random_number_neurons() + 1;
    auto area_id_to_area_name = get_random_area_names_specific(get_random_integer(size_t{ 1 }, num_neurons));
    auto num_areas = area_id_to_area_name.size();
    std::vector<RelearnTypes::area_id> neuron_id_to_area_id = get_random_area_ids(area_id_to_area_name.size(), num_neurons);

    LocalAreaTranslator translator(area_id_to_area_name, neuron_id_to_area_id);

    ASSERT_THROW(auto val = translator.get_area_name_for_neuron_id(num_neurons), RelearnException);
    ASSERT_THROW(auto val = translator.get_area_id_for_neuron_id(num_neurons), RelearnException);
    ASSERT_THROW(auto val = translator.get_area_name_for_area_id(num_areas), RelearnException);
    ASSERT_THROW(auto val = translator.get_area_id_for_area_name(std::to_string(get_random_percentage())), RelearnException);
}
#pragma clang diagnostic pop