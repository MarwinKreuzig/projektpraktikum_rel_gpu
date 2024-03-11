/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_kernel.h"

#include "adapter/random/RandomAdapter.h"

#include "adapter/kernel/KernelAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/kernel/KernelAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"

#include "algorithm/Cells.h"
#include "algorithm/Kernel/Kernel.h"
#include "util/Random.h"

#include <array>
#include <iostream>
#include <tuple>

#include <gtest/gtest.h>
#include <range/v3/algorithm/contains.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/indices.hpp>

TEST_F(KernelTest, testCalculateAttractivenessSameNode) {
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(1000, mt);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<BarnesHutCell>(mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    OctreeNode<BarnesHutCell> node{};
    node.set_cell_neuron_id(neuron_id);
    node.

        set_rank(MPIRank::root_rank());

    const auto attractiveness = Kernel<BarnesHutCell>::calculate_attractiveness_to_connect(
        { MPIRank::root_rank(), neuron_id }, position, &node, element_type, signal_type);
    ASSERT_EQ(attractiveness,
        0.0);
}

TEST_F(KernelTest, testCalculateAttractivenessException) {
    const auto& neuron_id_1 = NeuronIdAdapter::get_random_neuron_id(1000, mt);
    const auto& neuron_id_2 = NeuronIdAdapter::get_random_neuron_id(1000, 1000, mt);

    const auto& source_position = SimulationAdapter::get_random_position(mt);

    const auto& number_vacant_excitatory_axons = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
    const auto& number_vacant_inhibitory_axons = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
    const auto& number_vacant_excitatory_dendrites = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15,
        mt);
    const auto& number_vacant_inhibitory_dendrites = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15,
        mt);

    OctreeNode<FastMultipoleMethodCell> node{};
    node.set_cell_neuron_id(neuron_id_1);
    node.

        set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position()

        );

    node.set_cell_excitatory_axons_position({});
    node.set_cell_inhibitory_axons_position({});
    node.set_cell_excitatory_dendrites_position({});
    node.set_cell_inhibitory_dendrites_position({});

    node.set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
    node.set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodCell>(mt);

    using type = Kernel<FastMultipoleMethodCell>;

    ASSERT_THROW(const auto attr_exc_axons = type::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id_2 },
                     source_position, &node,
                     ElementType::Axon,
                     SignalType::Excitatory);

                 , RelearnException);

    ASSERT_THROW(const auto attr_inh_axons = type::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id_2 },
                     source_position, &node,
                     ElementType::Axon,
                     SignalType::Inhibitory);

                 , RelearnException);

    ASSERT_THROW(
        const auto attr_exc_dendrites = type::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id_2 },
            source_position, &node,
            ElementType::Dendrite,
            SignalType::Excitatory);

        , RelearnException);

    ASSERT_THROW(
        const auto attr_inh_dendrites = type::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id_2 },
            source_position, &node,
            ElementType::Dendrite,
            SignalType::Inhibitory);

        , RelearnException);
}

TEST_F(KernelTest, testCreateProbabilityIntervalEmptyVector) {
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodCell>(mt);

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodCell>::create_probability_interval(
        { MPIRank::root_rank(), neuron_id }, position, {}, element_type, signal_type);

    ASSERT_EQ(sum,
        0.0);
    ASSERT_EQ(0, attrs.

                 size()

    );
}

TEST_F(KernelTest, testCreateProbabilityIntervalAutapseVector) {
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto number_nodes = NeuronIdAdapter::get_random_number_neurons(mt);

    std::vector<OctreeNode<FastMultipoleMethodCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodCell>*> node_pointers{
        number_nodes, nullptr
    };

    for (
        const auto i :
        ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(neuron_id);
        nodes[i].

            set_rank(MPIRank::root_rank());

        nodes[i].

            set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position()

            );

        const auto& target_excitatory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_excitatory_dendrite_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_dendrite_position = SimulationAdapter::get_random_position(mt);

        const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodCell>(mt);

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodCell>::create_probability_interval(
        { MPIRank::root_rank(), neuron_id }, position, node_pointers, element_type, signal_type);

    ASSERT_EQ(sum,
        0.0);
    ASSERT_EQ(0, attrs.

                 size()

    );
}

TEST_F(KernelTest, testCreateProbabilityIntervalVectorException) {
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto number_nodes = NeuronIdAdapter::get_random_number_neurons(mt);

    std::vector<OctreeNode<FastMultipoleMethodCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodCell>*> node_pointers{
        number_nodes, nullptr
    };

    for (
        const auto i :
        ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(NeuronIdAdapter::get_random_neuron_id(1000, 1000, mt));
        nodes[i].

            set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position()

            );

        const auto& target_excitatory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_excitatory_dendrite_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_dendrite_position = SimulationAdapter::get_random_position(mt);

        const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto nullptr_index = RandomAdapter::get_random_integer<size_t>(0, number_nodes - 1, mt);
    node_pointers[nullptr_index] = nullptr;

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodCell>(mt);

    using TT = Kernel<FastMultipoleMethodCell>;

    ASSERT_THROW(
        const auto& val = TT::create_probability_interval({ MPIRank::root_rank(), neuron_id }, position, node_pointers,
            element_type, signal_type);

        , RelearnException);
}

TEST_F(KernelTest, testCreateProbabilityIntervalRandomVector) {
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto number_nodes = NeuronIdAdapter::get_random_number_neurons(mt);

    std::vector<OctreeNode<FastMultipoleMethodCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodCell>*> node_pointers{
        number_nodes, nullptr
    };

    for (
        const auto i :
        ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(NeuronIdAdapter::get_random_neuron_id(1000, 1000, mt));
        nodes[i].

            set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position()

            );
        nodes[i].set_rank(MPIRank(0));

        const auto& target_excitatory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_excitatory_dendrite_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_dendrite_position = SimulationAdapter::get_random_position(mt);

        const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodCell>(mt);

    auto total_attractiveness = 0.0;
    std::vector<double> attractivenesses{};
    for (
        const auto i :
        ranges::views::indices(number_nodes)) {
        const auto attr = Kernel<FastMultipoleMethodCell>::calculate_attractiveness_to_connect(
            { MPIRank::root_rank(), neuron_id }, position, &nodes[i], element_type, signal_type);

        attractivenesses.emplace_back(attr);
        total_attractiveness += attr;
    }

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodCell>::create_probability_interval(
        { MPIRank::root_rank(), neuron_id }, position, node_pointers, element_type, signal_type);

    if (total_attractiveness > 0.0) {
        ASSERT_NEAR(sum, total_attractiveness, eps);
        ASSERT_EQ(attractivenesses
                      .

                  size(),
            attrs

                .

            size()

        );

        for (
            const auto i :
            ranges::views::indices(attrs
                                       .

                                   size()

                    )) {
            ASSERT_NEAR(attrs[i], attractivenesses[i], eps);
        }
    }
}

TEST_F(KernelTest, testCreateProbabilityIntervalEdgeCase) {
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt) + 1000000000;

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto number_nodes = NeuronIdAdapter::get_random_number_neurons(mt);

    std::vector<OctreeNode<FastMultipoleMethodCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodCell>*> node_pointers{
        number_nodes, nullptr
    };

    for (
        const auto i :
        ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(NeuronIdAdapter::get_random_neuron_id(1000, 1000, mt));
        nodes[i].

            set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position()

            );
        nodes[i].set_rank(MPIRank(0));

        const auto& target_excitatory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_excitatory_dendrite_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_dendrite_position = SimulationAdapter::get_random_position(mt);

        const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodCell>(mt);

    auto total_attractiveness = 0.0;
    std::vector<double> attractivenesses{};
    for (
        const auto i :
        ranges::views::indices(number_nodes)) {
        const auto attr = Kernel<FastMultipoleMethodCell>::calculate_attractiveness_to_connect(
            { MPIRank::root_rank(), neuron_id }, position, &nodes[i], element_type, signal_type);

        attractivenesses.emplace_back(attr);
        total_attractiveness += attr;
    }

    if (total_attractiveness == 0.0) {
        std::cerr << "testProbabilityIntervalEdgeCase: Please increase the overhead";
        return;
    }

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodCell>::create_probability_interval(
        { MPIRank::root_rank(), neuron_id }, position, node_pointers, element_type, signal_type);

    attractivenesses.resize(0);

    for (
        const auto i :
        ranges::views::indices(number_nodes)) {
        const auto number_values = nodes[i].get_cell().get_number_elements_for(element_type, signal_type);
        if (number_values == 0) {
            attractivenesses.emplace_back(0);
            continue;
        }

        const auto distance = (position - nodes[i].get_cell().get_position_for(element_type, signal_type).value()).calculate_2_norm();
        const auto attr = number_values / distance;

        attractivenesses.emplace_back(attr);
        total_attractiveness += attr;
    }

    ASSERT_NEAR(sum, total_attractiveness, eps);
    ASSERT_EQ(attractivenesses
                  .

              size(),
        attrs

            .

        size()

    );

    for (
        auto i = 0;
        i < attrs.

            size();

        i++) {
        ASSERT_NEAR(attrs[i], attractivenesses[i], eps);
    }
}

TEST_F(KernelTest, testPickTargetEmpty2) {
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<BarnesHutCell>(mt);

    auto* result = Kernel<BarnesHutCell>::pick_target({ MPIRank::root_rank(), neuron_id }, position, {}, element_type,
        signal_type);

    ASSERT_EQ(result, nullptr);
}

TEST_F(KernelTest, testPickTargetException) {
    const auto number_nodes = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    std::vector<OctreeNode<FastMultipoleMethodCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodCell>*> node_pointers{
        number_nodes, nullptr
    };

    for (
        const auto i :
        ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(NeuronIdAdapter::get_random_neuron_id(1000, 1000, mt));
        nodes[i].

            set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position()

            );

        const auto& target_excitatory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_excitatory_dendrite_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_dendrite_position = SimulationAdapter::get_random_position(mt);

        const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto nullptr_index = RandomAdapter::get_random_integer<size_t>(0, number_nodes - 1, mt);
    node_pointers[nullptr_index] = nullptr;

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodCell>(mt);

    using TT = Kernel<FastMultipoleMethodCell>;

    ASSERT_THROW(auto* result = TT::pick_target({ MPIRank::root_rank(), neuron_id }, position, node_pointers, element_type,
                     signal_type);

                 , RelearnException);
}

TEST_F(KernelTest, testPickTargetRandom2) {
    const auto number_nodes = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto& neuron_id = NeuronIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    std::vector<OctreeNode<FastMultipoleMethodCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodCell>*> node_pointers{
        number_nodes, nullptr
    };

    for (
        const auto i :
        ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(NeuronIdAdapter::get_random_neuron_id(1000, 1000, mt));
        nodes[i].

            set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position()

            );

        const auto& target_excitatory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_excitatory_dendrite_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_dendrite_position = SimulationAdapter::get_random_position(mt);

        const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);
        const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(
            0, 15, mt);

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodCell>(mt);

    for (
        const auto i :
        ranges::views::indices(number_nodes)) {
        auto* result = Kernel<FastMultipoleMethodCell>::
            pick_target({ MPIRank::root_rank(), neuron_id }, position, node_pointers, element_type, signal_type);

        if (result == nullptr) {
            for (
                auto* ptr : node_pointers) {
                const auto attraction = Kernel<FastMultipoleMethodCell>::calculate_attractiveness_to_connect(
                    { MPIRank::root_rank(), neuron_id }, position, ptr, element_type, signal_type);
                ASSERT_EQ(attraction,
                    0.0);
            }

            continue;
        }

        ASSERT_TRUE(ranges::contains(node_pointers, result));
    }
}
