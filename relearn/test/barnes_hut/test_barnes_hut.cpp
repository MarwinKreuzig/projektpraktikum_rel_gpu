/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_barnes_hut.h"

#include "mpi/mpi_rank_adapter.h"
#include "neurons/neurons_adapter.h"
#include "neurons/neuron_types_adapter.h"
#include "simulation/simulation_adapter.h"
#include "synaptic_elements/synaptic_elements_adapter.h"
#include "tagged_id/tagged_id_adapter.h"

#include "algorithm/Algorithms.h"
#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/Cells.h"
#include "structure/Cell.h"
#include "structure/Octree.h"
#include "util/Vec3.h"

#include <memory>
#include <stack>
#include <tuple>
#include <vector>

TEST_F(BarnesHutTest, testBarnesHutGetterSetter) {
    using additional_cell_attributes = BarnesHutCell;

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    auto octree = std::make_shared<OctreeImplementation<additional_cell_attributes>>(min, max, 0);

    ASSERT_NO_THROW(BarnesHut algorithm(octree););

    BarnesHut algorithm(octree);

    ASSERT_EQ(algorithm.get_acceptance_criterion(), Constants::bh_default_theta);

    const auto random_acceptance_criterion = RandomAdapter::get_random_double<double>(0.0, Constants::bh_max_theta, mt);

    ASSERT_NO_THROW(algorithm.set_acceptance_criterion(random_acceptance_criterion));
    ASSERT_EQ(algorithm.get_acceptance_criterion(), random_acceptance_criterion);

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutTest, testBarnesHutACException) {
    using additional_cell_attributes = BarnesHutCell;

    const auto minimum = Vec3d{ 0.0, 0.0, 0.0 };
    const auto maximum = Vec3d{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankAdapter::get_random_mpi_rank(1024, mt);
    const auto level = RandomAdapter::get_random_integer<std::uint16_t>(0, 24, mt);
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> node{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto source_position = Vec3d{ 15.0, 15.0, 15.0 };

    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     nullptr, ElementType::Dendrite, searched_signal_type, Constants::bh_default_theta),
        RelearnException);

    const auto too_small_acceptance_criterion = RandomAdapter::get_random_double<double>(-1000.0, 0.0, mt);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     &node, ElementType::Dendrite, searched_signal_type, 0.0),
        RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     &node, ElementType::Dendrite, searched_signal_type, too_small_acceptance_criterion),
        RelearnException);

    const auto too_large_acceptance_criterion = RandomAdapter::get_random_double<double>(Constants::bh_max_theta + eps, 1000.0, mt);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     &node, ElementType::Dendrite, searched_signal_type, too_large_acceptance_criterion),
        RelearnException);

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutTest, testBarnesHutACLeaf) {
    using additional_cell_attributes = BarnesHutCell;

    const auto minimum = Vec3d{ 0.0, 0.0, 0.0 };
    const auto maximum = Vec3d{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankAdapter::get_random_mpi_rank(1024, mt);
    const auto level = RandomAdapter::get_random_integer<std::uint16_t>(0, 24, mt);
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> node{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);

    for (auto it = 0; it < 1000; it++) {
        const auto& position = SimulationAdapter::get_random_position(mt);
        const auto number_free_elements = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(1, 1000, mt);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_dendrites(number_free_elements, 0);
        } else {
            node.set_cell_number_dendrites(0, number_free_elements);
        }

        const auto accept = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(accept, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_dendrites(0, number_free_elements);
        } else {
            node.set_cell_number_dendrites(number_free_elements, 0);
        }

        const auto discard = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(discard, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Discard);
    }
}

TEST_F(BarnesHutTest, testBarnesHutACParent) {
    using additional_cell_attributes = BarnesHutCell;

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);
    const auto& scaled_minimum = minimum / 10.0;
    const auto& scaled_maximum = maximum / 10.0;

    const auto rank = MPIRankAdapter::get_random_mpi_rank(1024, mt);
    const auto level = RandomAdapter::get_random_integer<std::uint16_t>(0, 24, mt);
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationAdapter::get_random_position_in_box(scaled_minimum, scaled_maximum, mt);

    OctreeNode<additional_cell_attributes> node{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);
    node.set_parent();

    node.set_cell_size(scaled_minimum, scaled_maximum);
    node.set_cell_neuron_position(node_position);

    const auto& cell_dimenstions = scaled_maximum - scaled_minimum;
    const auto& maximum_cell_dimension = cell_dimenstions.get_maximum();

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        const auto distance = (node_position - position).calculate_2_norm();
        const auto quotient = maximum_cell_dimension / distance;

        const auto number_free_elements = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(1, 1000, mt);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_dendrites(number_free_elements, 0);
        } else {
            node.set_cell_number_dendrites(0, number_free_elements);
        }

        const auto status = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        if (acceptance_criterion > quotient) {
            ASSERT_EQ(status, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept);
        } else {
            ASSERT_EQ(status, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand);
        }

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_dendrites(0, number_free_elements);
        } else {
            node.set_cell_number_dendrites(number_free_elements, 0);
        }

        const auto discard = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(discard, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Discard);
    }
}

TEST_F(BarnesHutTest, testUpdateFunctor) {
    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto& axons = SynapticElementsAdapter::create_axons(number_neurons, mt);
    const auto& excitatory_dendrites = SynapticElementsAdapter::create_dendrites(number_neurons, SignalType::Excitatory, mt);
    const auto& inhibitory_dendrites = SynapticElementsAdapter::create_dendrites(number_neurons, SignalType::Inhibitory, mt);

    std::vector<std::tuple<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons, mt);

    auto octree = std::make_shared<OctreeImplementation<BarnesHutCell>>(min, max, 0);

    std::map<NeuronID::value_type, Vec3d> positions{};
    for (const auto& [position, id] : neurons_to_place) {
        octree->insert(position, id);
        positions[id.get_neuron_id()] = position;
    }

    octree->initializes_leaf_nodes(number_neurons);

    BarnesHut barnes_hut(octree);
    barnes_hut.set_synaptic_elements(axons, excitatory_dendrites, inhibitory_dendrites);

    const auto update_status = NeuronTypesAdapter::get_update_status(number_neurons, mt);

    ASSERT_NO_THROW(barnes_hut.update_octree(update_status));

    std::stack<OctreeNode<BarnesHutCell>*> stack{};
    stack.push(octree->get_root());

    while (!stack.empty()) {
        auto* node = stack.top();
        stack.pop();

        const auto& cell = node->get_cell();

        if (node->is_leaf()) {
            const auto id = cell.get_neuron_id();
            const auto local_id = id.get_neuron_id();

            ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
            ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());

            const auto& golden_position = positions[local_id];

            ASSERT_EQ(cell.get_excitatory_dendrites_position().value(), golden_position);
            ASSERT_EQ(cell.get_inhibitory_dendrites_position().value(), golden_position);

            if (update_status[local_id] == UpdateStatus::Disabled) {
                ASSERT_EQ(cell.get_number_excitatory_dendrites(), 0);
                ASSERT_EQ(cell.get_number_inhibitory_dendrites(), 0);
            } else {
                const auto& golden_excitatory_dendrites = excitatory_dendrites->get_free_elements(id);
                const auto& golden_inhibitory_dendrites = inhibitory_dendrites->get_free_elements(id);

                ASSERT_EQ(cell.get_number_excitatory_dendrites(), golden_excitatory_dendrites);
                ASSERT_EQ(cell.get_number_inhibitory_dendrites(), golden_inhibitory_dendrites);
            }
        } else {
            auto total_number_excitatory_dendrites = 0;
            auto total_number_inhibitory_dendrites = 0;

            Vec3d excitatory_dendrites_position = { 0, 0, 0 };
            Vec3d inhibitory_dendrites_position = { 0, 0, 0 };

            for (auto* child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                const auto& child_cell = child->get_cell();

                const auto number_excitatory_dendrites = child_cell.get_number_excitatory_dendrites();
                const auto number_inhibitory_dendrites = child_cell.get_number_inhibitory_dendrites();

                total_number_excitatory_dendrites += number_excitatory_dendrites;
                total_number_inhibitory_dendrites += number_inhibitory_dendrites;

                if (number_excitatory_dendrites != 0) {
                    const auto& opt = child_cell.get_excitatory_dendrites_position();
                    ASSERT_TRUE(opt.has_value());
                    const auto& position = opt.value();

                    excitatory_dendrites_position += (position * number_excitatory_dendrites);
                }

                if (number_inhibitory_dendrites != 0) {
                    const auto& opt = child_cell.get_inhibitory_dendrites_position();
                    ASSERT_TRUE(opt.has_value());
                    const auto& position = opt.value();

                    inhibitory_dendrites_position += (position * number_inhibitory_dendrites);
                }

                stack.push(child);
            }

            ASSERT_EQ(total_number_excitatory_dendrites, cell.get_number_excitatory_dendrites());
            ASSERT_EQ(total_number_inhibitory_dendrites, cell.get_number_inhibitory_dendrites());

            if (total_number_excitatory_dendrites == 0) {
                ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
            } else {
                const auto& opt = cell.get_excitatory_dendrites_position();
                ASSERT_TRUE(opt.has_value());
                const auto& position = opt.value();

                const auto& diff = (excitatory_dendrites_position / total_number_excitatory_dendrites) - position;
                const auto& norm = diff.calculate_2_norm();

                ASSERT_NEAR(norm, 0.0, eps);
            }

            if (total_number_inhibitory_dendrites == 0) {
                ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
            } else {
                const auto& opt = cell.get_inhibitory_dendrites_position();
                ASSERT_TRUE(opt.has_value());
                const auto& position = opt.value();

                const auto& diff = (inhibitory_dendrites_position / total_number_inhibitory_dendrites) - position;
                const auto& norm = diff.calculate_2_norm();

                ASSERT_NEAR(norm, 0.0, eps);
            }
        }
    }

    make_mpi_mem_available<BarnesHutCell>();
}

TEST_F(BarnesHutInvertedTest, testBarnesHutInvertedGetterSetter) {
    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(mt);
    auto octree = std::make_shared<OctreeImplementation<BarnesHutInvertedCell>>(min, max, 0);

    ASSERT_NO_THROW(BarnesHutInverted algorithm(octree););

    BarnesHutInverted algorithm(octree);

    ASSERT_EQ(algorithm.get_acceptance_criterion(), Constants::bh_default_theta);

    const auto random_acceptance_criterion = RandomAdapter::get_random_double<double>(0.0, Constants::bh_max_theta, mt);

    ASSERT_NO_THROW(algorithm.set_acceptance_criterion(random_acceptance_criterion));
    ASSERT_EQ(algorithm.get_acceptance_criterion(), random_acceptance_criterion);

    make_mpi_mem_available<BarnesHutInvertedCell>();
}

TEST_F(BarnesHutInvertedTest, testBarnesHutInvertedACException) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto minimum = Vec3d{ 0.0, 0.0, 0.0 };
    const auto maximum = Vec3d{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankAdapter::get_random_mpi_rank(1024, mt);
    const auto level = RandomAdapter::get_random_integer<std::uint16_t>(0, 24, mt);
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> node{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto source_position = Vec3d{ 15.0, 15.0, 15.0 };

    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     nullptr, ElementType::Axon, searched_signal_type, Constants::bh_default_theta),
        RelearnException);

    const auto too_small_acceptance_criterion = RandomAdapter::get_random_double<double>(-1000.0, 0.0, mt);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     &node, ElementType::Axon, searched_signal_type, 0.0),
        RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     &node, ElementType::Axon, searched_signal_type, too_small_acceptance_criterion),
        RelearnException);

    const auto too_large_acceptance_criterion = RandomAdapter::get_random_double<double>(Constants::bh_max_theta + eps, 1000.0, mt);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(source_position,
                     &node, ElementType::Axon, searched_signal_type, too_large_acceptance_criterion),
        RelearnException);

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutInvertedTest, testBarnesHutInvertedACLeaf) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto minimum = Vec3d{ 0.0, 0.0, 0.0 };
    const auto maximum = Vec3d{ 10.0, 10.0, 10.0 };

    const auto rank = MPIRankAdapter::get_random_mpi_rank(1024, mt);
    const auto level = RandomAdapter::get_random_integer<std::uint16_t>(0, 24, mt);
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> node{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);

    node.set_cell_size(minimum, maximum);
    node.set_cell_neuron_position(node_position);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);

    for (auto it = 0; it < 1000; it++) {
        const auto& position = SimulationAdapter::get_random_position(mt);
        const auto number_free_elements = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(1, 1000, mt);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_axons(number_free_elements, 0);
        } else {
            node.set_cell_number_axons(0, number_free_elements);
        }

        const auto accept = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(accept, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_axons(0, number_free_elements);
        } else {
            node.set_cell_number_axons(number_free_elements, 0);
        }

        const auto discard = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(discard, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Discard);
    }
}

TEST_F(BarnesHutInvertedTest, testBarnesHutACParent) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);
    const auto& scaled_minimum = minimum / 10.0;
    const auto& scaled_maximum = maximum / 10.0;

    const auto rank = MPIRankAdapter::get_random_mpi_rank(1024, mt);
    const auto level = RandomAdapter::get_random_integer<std::uint16_t>(0, 24, mt);
    const auto& neuron_id = TaggedIdAdapter::get_random_neuron_id(10000, mt);
    const auto& node_position = SimulationAdapter::get_random_position_in_box(scaled_minimum, scaled_maximum, mt);

    OctreeNode<additional_cell_attributes> node{};

    node.set_rank(rank);
    node.set_cell_neuron_id(neuron_id);
    node.set_level(level);
    node.set_parent();

    node.set_cell_size(scaled_minimum, scaled_maximum);
    node.set_cell_neuron_position(node_position);

    const auto& cell_dimenstions = scaled_maximum - scaled_minimum;
    const auto& maximum_cell_dimension = cell_dimenstions.get_maximum();

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        const auto distance = (node_position - position).calculate_2_norm();
        const auto quotient = maximum_cell_dimension / distance;

        const auto number_free_elements = RandomAdapter::get_random_integer<RelearnTypes::counter_type>(1, 1000, mt);

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_axons(number_free_elements, 0);
        } else {
            node.set_cell_number_axons(0, number_free_elements);
        }

        const auto status = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);

        if (acceptance_criterion > quotient) {
            ASSERT_EQ(status, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept);
        } else {
            ASSERT_EQ(status, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand);
        }

        if (searched_signal_type == SignalType::Excitatory) {
            node.set_cell_number_axons(0, number_free_elements);
        } else {
            node.set_cell_number_axons(number_free_elements, 0);
        }

        const auto discard = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, &node, ElementType::Axon, searched_signal_type, acceptance_criterion);
        ASSERT_EQ(discard, BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Discard);
    }
}
