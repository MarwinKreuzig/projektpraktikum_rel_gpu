/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_fast_multipole_method.h"

#include "Config.h"
#include "algorithm/FMMInternal/FastMultipoleMethodBase.h"

#include "adapter/fast_multipole_method/FMMAdapter.h"
#include "adapter/octree/OctreeAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/random/RandomAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "structure/OctreeNode.h"
#include "util/ranges/Functional.hpp"
#include "util/shuffle/shuffle.h"

#include <range/v3/action/sort.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/algorithm/any_of.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/repeat_n.hpp>

#include <algorithm>
#include <vector>

TEST_F(FMMTest, testMultiIndexGetNumberOfIndices) {
    ASSERT_EQ(MultiIndex::get_number_of_indices(), Constants::p3

    );
}

TEST_F(FMMTest, testMultIndexGetIndices) {
    std::vector<Vec3u> expected_indices{};
    expected_indices.

        reserve(MultiIndex::get_number_of_indices());

    for (
        auto x = 0U;
        x < Constants::p;
        x++) {
        for (
            auto y = 0U;
            y < Constants::p;
            y++) {
            for (
                auto z = 0U;
                z < Constants::p;
                z++) {
                expected_indices.emplace_back(x, y, z);
            }
        }
    }
    ranges::sort(expected_indices, std::less{});

    std::vector<Vec3u> actual_indices_vector = MultiIndex::get_indices() | ranges::to_vector | ranges::actions::sort(std::less{});

    ASSERT_EQ(expected_indices, actual_indices_vector);
}

TEST_F(FMMTest, testH) {
    const auto t = RandomAdapter::get_random_double<double>(-10.0, 10.0, mt);
    const auto alpha = RandomAdapter::get_random_integer<unsigned int>(0, 8, mt);

    const auto squared_t = t * t;
    const auto exponented_t = std::exp(-squared_t);

    const auto hermite_t_n = std::hermite(alpha, t);

    const auto multiplied = exponented_t * hermite_t_n;

    ASSERT_NEAR(multiplied, FastMultipoleMethodBase::h(alpha, t), eps) << t << ' ' << alpha << '\n';
}

TEST_F(FMMTest, testHMultiIndex) {
    const auto multi_index = FMMAdapter::get_random_multi_index(mt);
    const auto position = SimulationAdapter::get_random_position(mt);

    const auto actual_value = FastMultipoleMethodBase::h_multi_index(multi_index, position);

    const auto [multi_index_x, multi_index_y, multi_index_z] = multi_index;
    const auto [position_x, position_y, position_z] = position;

    const auto val_x = FastMultipoleMethodBase::h(multi_index_x, position_x);
    const auto val_y = FastMultipoleMethodBase::h(multi_index_y, position_y);
    const auto val_z = FastMultipoleMethodBase::h(multi_index_z, position_z);

    const auto product = val_x * val_y * val_z;

    ASSERT_NEAR(actual_value, product, eps);
}

TEST_F(FMMTest, testExtractElement) {
    const auto number_pointers = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto number_nullptrs = NeuronIdAdapter::get_random_number_neurons(mt);

    std::vector<OctreeNode<FastMultipoleMethodCell>> memory_holder(number_pointers);

    const std::vector<OctreeNode<FastMultipoleMethodCell>*>
        pointers = ranges::views::concat(
                       memory_holder | ranges::views::transform([](auto& Val) { return &Val; }),
                       ranges::views::repeat_n(
                           static_cast<OctreeNode<FastMultipoleMethodCell>*>(nullptr),
                           number_nullptrs))
        | ranges::to_vector | actions::shuffle(mt);

    std::vector<OctreeNode<FastMultipoleMethodCell>*> received_pointers{};
    received_pointers.reserve(number_pointers);

    for (
        auto pointer_index = 0;
        pointer_index < number_pointers;
        pointer_index++) {
        auto* ptr = FastMultipoleMethodBase::extract_element(pointers, pointer_index);
        ASSERT_NE(ptr, nullptr);

        received_pointers.emplace_back(ptr);
    }

    ranges::sort(received_pointers);

    for (
        auto i = 0;
        i < number_pointers;
        i++) {
        ASSERT_EQ(received_pointers[i], &memory_holder[i]);
    }

    for (
        auto pointer_index = number_pointers;
        pointer_index < number_pointers + number_nullptrs + number_neurons_out_of_scope;
        pointer_index++) {
        auto* ptr = FastMultipoleMethodBase::extract_element(pointers, pointer_index);
        ASSERT_EQ(ptr, nullptr);
    }
}

TEST_F(FMMTest, testCheckCalculationRequirementsException) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);
    const auto level = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeNode<FastMultipoleMethodCell> node{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.

        set_cell_neuron_id(NeuronID::virtual_id());

    node.set_cell_neuron_position(own_position);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(nullptr, nullptr, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(nullptr, nullptr, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(nullptr, nullptr, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(nullptr, nullptr, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(&node, nullptr, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(&node, nullptr, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(&node, nullptr, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(&node, nullptr, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(nullptr, &node, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(nullptr, &node, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(nullptr, &node, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::check_calculation_requirements(nullptr, &node, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);
}

TEST_F(FMMTest, testCheckCalculationRequirementsLeaf) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);

    OctreeNode<FastMultipoleMethodCell> node_1{};
    node_1.set_level(0);
    node_1.set_rank(my_rank);
    node_1.set_cell_size(min, max);
    node_1.set_cell_neuron_id(NeuronID(0));
    node_1.set_cell_neuron_position(SimulationAdapter::get_random_position_in_box(min, max, this->mt));
    node_1.set_cell_number_axons(1, 1);
    node_1.set_cell_number_dendrites(1, 1);

    OctreeNode<FastMultipoleMethodCell> node_2{};
    node_2.set_level(0);
    node_2.set_rank(my_rank);
    node_2.set_cell_size(min, max);
    node_2.set_cell_neuron_id(NeuronID(0));
    node_2.set_cell_neuron_position(SimulationAdapter::get_random_position_in_box(min, max, this->mt));
    node_2.set_cell_number_axons(0, 0);
    node_2.set_cell_number_dendrites(0, 0);

    auto root = OctreeAdapter::get_standard_tree<FastMultipoleMethodCell>(20, min, max, this->mt);

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &node_2, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &node_2, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &node_2, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &node_2, ElementType::Axon, SignalType::Inhibitory));

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &node_1, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &node_1, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &node_1, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &node_1, ElementType::Axon, SignalType::Inhibitory));

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &root, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &root, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &root, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &root, ElementType::Axon, SignalType::Inhibitory));

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_1, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_1, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_1, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_1, ElementType::Axon, SignalType::Inhibitory));

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &root, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &root, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &root, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &root, ElementType::Axon, SignalType::Inhibitory));

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_2, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_2, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_2, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_2, ElementType::Axon, SignalType::Inhibitory));
}

TEST_F(FMMTest, testCheckCalculationRequirements) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);

    const auto number_neurons_in_source = static_cast<NeuronID::value_type>(Constants::max_neurons_in_source * 0.75);
    const auto number_neurons_in_target = static_cast<NeuronID::value_type>(Constants::max_neurons_in_target * 0.75);

    auto source = OctreeAdapter::get_standard_tree<FastMultipoleMethodCell>(number_neurons_in_source, min, max, this->mt);
    auto target = OctreeAdapter::get_standard_tree<FastMultipoleMethodCell>(number_neurons_in_target, min, max, this->mt);

    const auto& source_cell = source.get_cell();
    const auto& target_cell = target.get_cell();

    auto check_combi = [&](const ElementType e, const SignalType s) {
        const auto has_enough_in_source = source_cell.get_number_elements_for(get_other_element_type(e), s) > Constants::max_neurons_in_source;
        const auto has_enough_in_target = target_cell.get_number_elements_for(e, s) > Constants::max_neurons_in_target;

        const auto type = FastMultipoleMethodBase::check_calculation_requirements(&source, &target, e, s);

        if (has_enough_in_source && has_enough_in_target) {
            ASSERT_EQ(CalculationType::Hermite, type);
        } else if (has_enough_in_target) {
            ASSERT_EQ(CalculationType::Taylor, type);
        } else {
            ASSERT_EQ(CalculationType::Direct, type);
        }
    };

    check_combi(ElementType::Axon, SignalType::Excitatory);
    check_combi(ElementType::Dendrite, SignalType::Excitatory);
    check_combi(ElementType::Axon, SignalType::Inhibitory);
    check_combi(ElementType::Dendrite, SignalType::Inhibitory);
}

TEST_F(FMMTest, testDirectGaussException) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);
    const auto level = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeNode<FastMultipoleMethodCell> node{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.

        set_cell_neuron_id(NeuronID::virtual_id());

    node.set_cell_neuron_position(own_position);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(nullptr, nullptr, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(nullptr, nullptr, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(nullptr, nullptr, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(nullptr, nullptr, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(&node, nullptr, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(&node, nullptr, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(&node, nullptr, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(&node, nullptr, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(nullptr, &node, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(nullptr, &node, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(nullptr, &node, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_direct_gauss(nullptr, &node, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);
}

TEST_F(FMMTest, testDirectGauss) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);

    const auto number_neurons_in_source = static_cast<NeuronID::value_type>(Constants::max_neurons_in_source * 0.2);
    const auto number_neurons_in_target = static_cast<NeuronID::value_type>(Constants::max_neurons_in_target * 0.2);

    auto source = OctreeAdapter::get_standard_tree<FastMultipoleMethodCell>(number_neurons_in_source, min, max, this->mt);
    auto target = OctreeAdapter::get_standard_tree<FastMultipoleMethodCell>(number_neurons_in_target, min, max, this->mt);

    const auto& source_leaves = OctreeAdapter::extract_leaf_nodes(&source);
    const auto& target_leaves = OctreeAdapter::extract_leaf_nodes(&target);

    auto check_combi = [&](const ElementType e, const SignalType s) {
        auto sum = 0.0;

        for (const auto* source_leaf : source_leaves) {
            for (const auto* target_leaf : target_leaves) {
                const auto vacant_sources = source_leaf->get_cell().get_number_elements_for(get_other_element_type(e), s);
                const auto vacant_targets = target_leaf->get_cell().get_number_elements_for(e, s);

                const auto product = vacant_sources * vacant_targets;
                if (product == 0) {
                    continue;
                }

                const auto attraction = FastMultipoleMethodBase::kernel(
                    source_leaf->get_cell().get_position_for(get_other_element_type(e), s).value(),
                    target_leaf->get_cell().get_position_for(e, s).value(), GaussianDistributionKernel::get_sigma());

                sum += attraction * product;
            }
        }

        ASSERT_NEAR(sum, FastMultipoleMethodBase::calc_direct_gauss(&source, &target, e, s), eps);
    };

    check_combi(ElementType::Axon, SignalType::Excitatory);
    check_combi(ElementType::Dendrite, SignalType::Excitatory);
    check_combi(ElementType::Axon, SignalType::Inhibitory);
    check_combi(ElementType::Dendrite, SignalType::Inhibitory);
}

TEST_F(FMMTest, testHermiteCoefficientsException) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);

    OctreeNode<FastMultipoleMethodCell> node{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.

        set_cell_neuron_id(NeuronID::virtual_id());

    node.set_cell_neuron_position(own_position);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite_coefficients(nullptr, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite_coefficients(nullptr, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite_coefficients(nullptr, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite_coefficients(nullptr, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(
        auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&node, ElementType::Axon, SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(
        auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&node, ElementType::Axon, SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&node, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&node, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);
}

TEST_F(FMMTest, testHermiteCoefficientsException2) {
    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 10;

    auto no_axon_tree = OctreeAdapter::get_tree_no_axons<FastMultipoleMethodCell>(number_neurons, min, max, mt);
    auto no_dendrite_tree = OctreeAdapter::get_tree_no_dendrites<FastMultipoleMethodCell>(number_neurons, min, max, mt);
    auto no_synaptic_elements_tree = OctreeAdapter::get_tree_no_synaptic_elements<FastMultipoleMethodCell>(number_neurons,
        min, max, mt);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&no_axon_tree, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&no_axon_tree, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&no_dendrite_tree, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&no_dendrite_tree, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(
        auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&no_synaptic_elements_tree, ElementType::Axon,
            SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(
        auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&no_synaptic_elements_tree, ElementType::Axon,
            SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(
        auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&no_synaptic_elements_tree, ElementType::Dendrite,
            SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(
        auto val = FastMultipoleMethodBase::calc_hermite_coefficients(&no_synaptic_elements_tree, ElementType::Dendrite,
            SignalType::Inhibitory),
        RelearnException);
}

TEST_F(FMMTest, testHermiteCoefficientsForm) {
    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 10;

    auto tree = OctreeAdapter::get_standard_tree<FastMultipoleMethodCell>(number_neurons, min, max, mt);

    const auto coefficients_a_e = FastMultipoleMethodBase::calc_hermite_coefficients(&tree, ElementType::Axon,
        SignalType::Excitatory);
    ASSERT_EQ(coefficients_a_e
                  .

              size(),
        Constants::p3

    );
    ASSERT_TRUE(ranges::any_of(coefficients_a_e, not_equal_to(0.0)));

    const auto coefficients_a_i = FastMultipoleMethodBase::calc_hermite_coefficients(&tree, ElementType::Axon,
        SignalType::Inhibitory);
    ASSERT_EQ(coefficients_a_i
                  .

              size(),
        Constants::p3

    );
    ASSERT_TRUE(ranges::any_of(coefficients_a_i, not_equal_to(0.0)));

    const auto coefficients_d_e = FastMultipoleMethodBase::calc_hermite_coefficients(&tree, ElementType::Dendrite,
        SignalType::Excitatory);
    ASSERT_EQ(coefficients_d_e
                  .

              size(),
        Constants::p3

    );
    ASSERT_TRUE(ranges::any_of(coefficients_d_e, not_equal_to(0.0)));

    const auto coefficients_d_i = FastMultipoleMethodBase::calc_hermite_coefficients(&tree, ElementType::Dendrite,
        SignalType::Inhibitory);
    ASSERT_EQ(coefficients_d_i
                  .

              size(),
        Constants::p3

    );
    ASSERT_TRUE(ranges::any_of(coefficients_d_i, not_equal_to(0.0)));
}

TEST_F(FMMTest, testTaylorCoefficientsException) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);
    const auto& other_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);

    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    OctreeNode<FastMultipoleMethodCell> node{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.

        set_cell_neuron_id(NeuronID::virtual_id());

    node.set_cell_neuron_position(own_position);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_taylor_coefficients(nullptr, other_position, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_taylor_coefficients(nullptr, other_position, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(
        auto val = FastMultipoleMethodBase::calc_taylor_coefficients(nullptr, other_position, ElementType::Dendrite,
            SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(
        auto val = FastMultipoleMethodBase::calc_taylor_coefficients(nullptr, other_position, ElementType::Dendrite,
            SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_taylor_coefficients(&node, other_position, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_taylor_coefficients(&node, other_position, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_taylor_coefficients(&node, other_position, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_taylor_coefficients(&node, other_position, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);
}

TEST_F(FMMTest, testTaylorCoefficientsZero) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& other_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 10;

    auto no_axon_tree = OctreeAdapter::get_tree_no_axons<FastMultipoleMethodCell>(number_neurons, min, max, mt);
    auto no_dendrite_tree = OctreeAdapter::get_tree_no_dendrites<FastMultipoleMethodCell>(number_neurons, min, max, mt);
    auto no_synaptic_elements_tree = OctreeAdapter::get_tree_no_synaptic_elements<FastMultipoleMethodCell>(number_neurons,
        min, max, mt);

    std::vector<double> coefficients(Constants::p3, 0.0);

    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(&no_axon_tree, other_position, ElementType::Axon,
                  SignalType::Excitatory),
        coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(&no_axon_tree, other_position, ElementType::Axon,
                  SignalType::Inhibitory),
        coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(&no_dendrite_tree, other_position, ElementType::Dendrite,
                  SignalType::Excitatory),
        coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(&no_dendrite_tree, other_position, ElementType::Dendrite,
                  SignalType::Inhibitory),
        coefficients);

    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(&no_synaptic_elements_tree, other_position,
                  ElementType::Axon, SignalType::Excitatory),
        coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(&no_synaptic_elements_tree, other_position,
                  ElementType::Axon, SignalType::Inhibitory),
        coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(&no_synaptic_elements_tree, other_position,
                  ElementType::Dendrite, SignalType::Excitatory),
        coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(&no_synaptic_elements_tree, other_position,
                  ElementType::Dendrite, SignalType::Inhibitory),
        coefficients);
}

TEST_F(FMMTest, testTaylorCoefficientsForm) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& other_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);
    const auto number_neurons = NeuronIdAdapter::get_random_number_neurons(mt) + 10;

    auto tree = OctreeAdapter::get_standard_tree<FastMultipoleMethodCell>(number_neurons, min, max, mt);

    const auto coefficients_a_e = FastMultipoleMethodBase::calc_taylor_coefficients(&tree, other_position,
        ElementType::Axon,
        SignalType::Excitatory);
    ASSERT_EQ(coefficients_a_e
                  .

              size(),
        Constants::p3

    );
    ASSERT_TRUE(ranges::any_of(coefficients_a_e, not_equal_to(0.0)));

    const auto coefficients_a_i = FastMultipoleMethodBase::calc_taylor_coefficients(&tree, other_position,
        ElementType::Axon,
        SignalType::Inhibitory);
    ASSERT_EQ(coefficients_a_i
                  .

              size(),
        Constants::p3

    );
    ASSERT_TRUE(ranges::any_of(coefficients_a_i, not_equal_to(0.0)));

    const auto coefficients_d_e = FastMultipoleMethodBase::calc_taylor_coefficients(&tree, other_position,
        ElementType::Dendrite,
        SignalType::Excitatory);
    ASSERT_EQ(coefficients_d_e
                  .

              size(),
        Constants::p3

    );
    ASSERT_TRUE(ranges::any_of(coefficients_d_e, not_equal_to(0.0)));

    const auto coefficients_d_i = FastMultipoleMethodBase::calc_taylor_coefficients(&tree, other_position,
        ElementType::Dendrite,
        SignalType::Inhibitory);
    ASSERT_EQ(coefficients_d_i
                  .

              size(),
        Constants::p3

    );
    ASSERT_TRUE(ranges::any_of(coefficients_d_i, not_equal_to(0.0)));
}

TEST_F(FMMTest, testCalcHermiteException) {
    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationAdapter::get_random_position_in_box(min, max, this->mt);

    OctreeNode<FastMultipoleMethodCell> node{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.

        set_cell_neuron_id(NeuronID::virtual_id());

    node.set_cell_neuron_position(own_position);

    std::vector<double> coefficients(Constants::p3, 0.0);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(nullptr, nullptr, coefficients, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(nullptr, nullptr, coefficients, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(nullptr, nullptr, coefficients, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(nullptr, nullptr, coefficients, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&node, nullptr, coefficients, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&node, nullptr, coefficients, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&node, nullptr, coefficients, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&node, nullptr, coefficients, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(nullptr, &node, coefficients, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(nullptr, &node, coefficients, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(nullptr, &node, coefficients, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(nullptr, &node, coefficients, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);

    auto source_tree = OctreeAdapter::get_standard_tree<FastMultipoleMethodCell>(30, min, max, this->mt);
    auto target_tree = OctreeAdapter::get_standard_tree<FastMultipoleMethodCell>(40, min, max, this->mt);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &target_tree, {}, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &target_tree, {}, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &target_tree, {}, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &target_tree, {}, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &target_tree, std::vector{ 1.0, 2.9 },
                     ElementType::Axon, SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &target_tree, std::vector{ 1.0, 2.9 },
                     ElementType::Axon, SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &target_tree, std::vector{ 1.0, 2.9 },
                     ElementType::Dendrite, SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &target_tree, std::vector{ 1.0, 2.9 },
                     ElementType::Dendrite, SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &node, coefficients, ElementType::Axon,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &node, coefficients, ElementType::Axon,
                     SignalType::Inhibitory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &node, coefficients, ElementType::Dendrite,
                     SignalType::Excitatory),
        RelearnException);

    ASSERT_THROW(auto val = FastMultipoleMethodBase::calc_hermite(&source_tree, &node, coefficients, ElementType::Dendrite,
                     SignalType::Inhibitory),
        RelearnException);
}

TEST_F(FMMTest, testCalcTaylorException) {
}
