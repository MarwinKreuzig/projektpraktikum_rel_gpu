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

#include "RandomAdapter.h"

#include "kernel_adapter.h"
#include "neurons/neuron_types_adapter.h"
#include "simulation/simulation_adapter.h"
#include "tagged_id/tagged_id_adapter.h"

#include "algorithm/Kernel/Kernel.h"
#include "algorithm/Cells.h"
#include "util/Random.h"

#include <array>
#include <iostream>
#include <tuple>

TEST_F(KernelTest, testKernelSameNode) {
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(1000, mt);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<BarnesHutCell>(mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    OctreeNode<BarnesHutCell> node{};
    node.set_cell_neuron_id(neuron_id);
    node.set_rank(MPIRank::root_rank());

    const auto attractiveness = Kernel<BarnesHutCell>::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id }, position, &node, element_type, signal_type);
    ASSERT_EQ(attractiveness, 0.0);
}

TEST_F(KernelTest, testKernelException) {
    const auto& neuron_id_1 = TaggedIdAdapter::get_random_neuron_id(1000, mt);
    const auto& neuron_id_2 = TaggedIdAdapter::get_random_neuron_id(1000, 1000, mt);

    const auto& source_position = SimulationAdapter::get_random_position(mt);

    const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
    const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
    const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
    const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);

    OctreeNode<FastMultipoleMethodsCell> node{};
    node.set_cell_neuron_id(neuron_id_1);
    node.set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position());

    node.set_cell_excitatory_axons_position({});
    node.set_cell_inhibitory_axons_position({});
    node.set_cell_excitatory_dendrites_position({});
    node.set_cell_inhibitory_dendrites_position({});

    node.set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
    node.set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodsCell>(mt);

    using type = Kernel<FastMultipoleMethodsCell>;
    ASSERT_THROW(const auto attr_exc_axons = type::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Axon, SignalType::Excitatory);, RelearnException);
    ASSERT_THROW(const auto attr_inh_axons = type::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Axon, SignalType::Inhibitory);, RelearnException);
    ASSERT_THROW(const auto attr_exc_dendrites = type::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Dendrite, SignalType::Excitatory);, RelearnException);
    ASSERT_THROW(const auto attr_inh_dendrites = type::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Dendrite, SignalType::Inhibitory);, RelearnException);
}

TEST_F(KernelTest, testKernelEmptyVector) {
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodsCell>(mt);

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodsCell>::create_probability_interval(
        { MPIRank::root_rank(), neuron_id }, position, {}, element_type, signal_type);

    ASSERT_EQ(sum, 0.0);
    ASSERT_EQ(0, attrs.size());
}

TEST_F(KernelTest, testKernelAutapseVector) {
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto number_nodes = TaggedIdAdapter::get_random_number_neurons(mt);

    std::vector<OctreeNode<FastMultipoleMethodsCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodsCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodsCell>*> node_pointers{ number_nodes, nullptr };

    for (auto i = 0; i < number_nodes; i++) {
        nodes[i].set_cell_neuron_id(neuron_id);
        nodes[i].set_rank(MPIRank::root_rank());
        nodes[i].set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position());

        const auto& target_excitatory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_excitatory_dendrite_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_dendrite_position = SimulationAdapter::get_random_position(mt);

        const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodsCell>(mt);

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodsCell>::create_probability_interval(
        { MPIRank::root_rank(), neuron_id }, position, node_pointers, element_type, signal_type);

    ASSERT_EQ(sum, 0.0);
    ASSERT_EQ(0, attrs.size());
}

TEST_F(KernelTest, testKernelVectorException) {
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto number_nodes = TaggedIdAdapter::get_random_number_neurons(mt);

    std::vector<OctreeNode<FastMultipoleMethodsCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodsCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodsCell>*> node_pointers{ number_nodes, nullptr };

    for (auto i = 0; i < number_nodes; i++) {
        nodes[i].set_cell_neuron_id(TaggedIdAdapter::get_random_neuron_id(1000, 1000, mt));
        nodes[i].set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position());

        const auto& target_excitatory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_excitatory_dendrite_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_dendrite_position = SimulationAdapter::get_random_position(mt);

        const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);

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

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodsCell>(mt);

    using TT = Kernel<FastMultipoleMethodsCell>;

    ASSERT_THROW(const auto& val = TT::create_probability_interval({ MPIRank::root_rank(), neuron_id }, position, node_pointers, element_type, signal_type);, RelearnException);
}

TEST_F(KernelTest, testKernelRandomVector) {
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto number_nodes = TaggedIdAdapter::get_random_number_neurons(mt);

    std::vector<OctreeNode<FastMultipoleMethodsCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodsCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodsCell>*> node_pointers{ number_nodes, nullptr };

    for (auto i = 0; i < number_nodes; i++) {
        nodes[i].set_cell_neuron_id(TaggedIdAdapter::get_random_neuron_id(1000, 1000, mt));
        nodes[i].set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position());
        nodes[i].set_rank(MPIRank(0));

        const auto& target_excitatory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_excitatory_dendrite_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_dendrite_position = SimulationAdapter::get_random_position(mt);

        const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodsCell>(mt);

    auto total_attractiveness = 0.0;
    std::vector<double> attractivenesses{};
    for (auto i = 0; i < number_nodes; i++) {
        const auto attr = Kernel<FastMultipoleMethodsCell>::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id }, position, &nodes[i], element_type, signal_type);

        attractivenesses.emplace_back(attr);
        total_attractiveness += attr;
    }

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodsCell>::create_probability_interval(
        { MPIRank::root_rank(), neuron_id }, position, node_pointers, element_type, signal_type);

    if (total_attractiveness > 0.0) {
        ASSERT_NEAR(sum, total_attractiveness, eps);
        ASSERT_EQ(attractivenesses.size(), attrs.size());

        for (auto i = 0; i < attrs.size(); i++) {
            ASSERT_NEAR(attrs[i], attractivenesses[i], eps);
        }
    }
}

TEST_F(KernelTest, testPickTargetEmpty) {
    const std::vector<OctreeNode<BarnesHutCell>*> targets{};
    const std::vector<double> probabilities{};

    const auto random_number = RandomAdapter::get_random_double<double>(0.0, 100.0, mt);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<BarnesHutCell>(mt);

    using TT = Kernel<BarnesHutCell>;

    ASSERT_THROW(const auto& val = TT::pick_target(targets, probabilities, random_number);, RelearnException);
}

TEST_F(KernelTest, testPickTargetMismatchSize) {
    const auto number_targets = RandomAdapter::get_random_integer<size_t>(1, 1000, mt);
    auto number_probabilities = RandomAdapter::get_random_integer<size_t>(1, 1000, mt);
    while (number_probabilities == number_targets) {
        number_probabilities = RandomAdapter::get_random_integer<size_t>(1, 1000, mt);
    }

    std::vector<OctreeNode<BarnesHutCell>*> targets{ number_targets };
    std::vector<double> probabilities = std::vector<double>(number_probabilities, 0.0);

    const auto random_number = RandomAdapter::get_random_double<double>(0.0, 100.0, mt);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<BarnesHutCell>(mt);

    using TT = Kernel<BarnesHutCell>;

    ASSERT_THROW(const auto& val = TT::pick_target(targets, probabilities, random_number);, RelearnException);
}

TEST_F(KernelTest, testPickTargetNegativeRandomNumber) {
    const auto number_targets = RandomAdapter::get_random_integer<size_t>(1, 1000, mt);

    std::vector<OctreeNode<BarnesHutCell>*> targets{ number_targets };
    std::vector<double> probabilities = std::vector<double>(number_targets, 0.0);

    const auto random_number = -RandomAdapter::get_random_double<double>(0.001, 100.0, mt);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<BarnesHutCell>(mt);

    using TT = Kernel<BarnesHutCell>;

    ASSERT_THROW(const auto& val = TT::pick_target(targets, probabilities, random_number);, RelearnException);
}

TEST_F(KernelTest, testPickTargetRandom) {
    const auto number_nodes = TaggedIdAdapter::get_random_number_neurons(mt);

    std::vector<OctreeNode<BarnesHutCell>> nodes{ number_nodes, OctreeNode<BarnesHutCell>{} };
    std::vector<OctreeNode<BarnesHutCell>*> node_pointers{ number_nodes, nullptr };
    std::vector<double> probabilities{};

    for (auto i = 0; i < number_nodes; i++) {
        node_pointers[i] = &nodes[i];
        probabilities.emplace_back(RandomAdapter::get_random_percentage<double>(mt));
    }

    const auto total_probability = std::reduce(probabilities.begin(), probabilities.end(), 0.0);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<BarnesHutCell>(mt);

    for (auto it = 0; it < number_nodes; it++) {
        const auto random_number = RandomAdapter::get_random_double<double>(0.0, total_probability, mt);

        auto current_probability = random_number;
        auto index = std::numeric_limits<size_t>::max();

        for (auto i = 0; i < number_nodes; i++) {
            current_probability -= probabilities[i];
            if (current_probability <= 0.0) {
                index = i;
                break;
            }
        }

        if (index == std::numeric_limits<size_t>::max()) {
            std::cerr << "testPickTargetRandom: the index was -1\n";
            index = number_nodes - 1;
        }

        const auto* chosen_node = Kernel<BarnesHutCell>::pick_target(node_pointers, probabilities, random_number);

        ASSERT_EQ(chosen_node, node_pointers[index]);
    }
}

TEST_F(KernelTest, testPickTargetTooLarge) {
    const auto number_nodes = TaggedIdAdapter::get_random_number_neurons(mt);

    std::vector<OctreeNode<BarnesHutCell>> nodes{ number_nodes, OctreeNode<BarnesHutCell>{} };
    std::vector<OctreeNode<BarnesHutCell>*> node_pointers{ number_nodes, nullptr };
    std::vector<double> probabilities{};

    for (auto i = 0; i < number_nodes; i++) {
        node_pointers[i] = &nodes[i];
        probabilities.emplace_back(RandomAdapter::get_random_percentage<double>(mt));
    }

    const auto total_probability = std::reduce(probabilities.begin(), probabilities.end(), 0.0);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<BarnesHutCell>(mt);

    for (auto it = 0; it < number_nodes; it++) {
        const auto random_number = RandomAdapter::get_random_double<double>(total_probability + eps, (total_probability + eps + 1) * 2, mt);
        const auto* chosen_node = Kernel<BarnesHutCell>::pick_target(node_pointers, probabilities, random_number);

        ASSERT_EQ(chosen_node, node_pointers[number_nodes - 1]);
    }
}

TEST_F(KernelTest, testPickTargetEmpty2) {
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<BarnesHutCell>(mt);

    auto* result = Kernel<BarnesHutCell>::pick_target({ MPIRank::root_rank(), neuron_id }, position, {}, element_type, signal_type);

    ASSERT_EQ(result, nullptr);
}

TEST_F(KernelTest, testPickTargetException) {
    const auto number_nodes = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    std::vector<OctreeNode<FastMultipoleMethodsCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodsCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodsCell>*> node_pointers{ number_nodes, nullptr };

    for (auto i = 0; i < number_nodes; i++) {
        nodes[i].set_cell_neuron_id(TaggedIdAdapter::get_random_neuron_id(1000, 1000, mt));
        nodes[i].set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position());

        const auto& target_excitatory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_excitatory_dendrite_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_dendrite_position = SimulationAdapter::get_random_position(mt);

        const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);

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

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodsCell>(mt);

    using TT = Kernel<FastMultipoleMethodsCell>;
    ASSERT_THROW(auto* result = TT::pick_target({ MPIRank::root_rank(), neuron_id }, position, node_pointers, element_type, signal_type);, RelearnException);
}

TEST_F(KernelTest, testPickTargetRandom2) {
    const auto number_nodes = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(1000, mt);

    const auto& position = SimulationAdapter::get_random_position(mt);

    const auto element_type = NeuronTypesAdapter::get_random_element_type(mt);
    const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    std::vector<OctreeNode<FastMultipoleMethodsCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodsCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodsCell>*> node_pointers{ number_nodes, nullptr };

    for (auto i = 0; i < number_nodes; i++) {
        nodes[i].set_cell_neuron_id(TaggedIdAdapter::get_random_neuron_id(1000, 1000, mt));
        nodes[i].set_cell_size(SimulationAdapter::get_minimum_position(), SimulationAdapter::get_maximum_position());

        const auto& target_excitatory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_axon_position = SimulationAdapter::get_random_position(mt);
        const auto& target_excitatory_dendrite_position = SimulationAdapter::get_random_position(mt);
        const auto& target_inhibitory_dendrite_position = SimulationAdapter::get_random_position(mt);

        const auto& number_vacant_excitatory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_inhibitory_axons = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_excitatory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);
        const auto& number_vacant_inhibitory_dendrites = RandomAdapter::RandomAdapter::get_random_integer<RelearnTypes::counter_type>(0, 15, mt);

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto& debug_kernel_string = KernelAdapter::set_random_kernel<FastMultipoleMethodsCell>(mt);

    for (auto i = 0; i < number_nodes; i++) {
        auto* result = Kernel<FastMultipoleMethodsCell>::
            pick_target({ MPIRank::root_rank(), neuron_id }, position, node_pointers, element_type, signal_type);

        if (result == nullptr) {
            for (auto* ptr : node_pointers) {
                const auto attraction = Kernel<FastMultipoleMethodsCell>::calculate_attractiveness_to_connect({ MPIRank::root_rank(), neuron_id }, position, ptr, element_type, signal_type);
                ASSERT_EQ(attraction, 0.0);
            }

            continue;
        }

        auto pos = std::find(node_pointers.begin(), node_pointers.end(), result);
        ASSERT_NE(pos, node_pointers.end());
    }
}
