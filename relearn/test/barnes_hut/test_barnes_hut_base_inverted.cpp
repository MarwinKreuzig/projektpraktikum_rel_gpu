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
#include "node_cache/node_cache_adapter.h"
#include "octree/octree_adapter.h"
#include "simulation/simulation_adapter.h"
#include "tagged_id/tagged_id_adapter.h"

#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/Cells.h"
#include "structure/Cell.h"
#include "structure/OctreeNode.h"
#include "util/Vec3.h"

#include <stack>
#include <tuple>
#include <vector>

TEST_F(BarnesHutInvertedBaseTest, testACException) {
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

TEST_F(BarnesHutInvertedBaseTest, testACLeafAxons) {
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

TEST_F(BarnesHutInvertedBaseTest, testACParentAxon) {
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

    const auto& cell_dimensions = scaled_maximum - scaled_minimum;
    const auto& maximum_cell_dimension = cell_dimensions.get_maximum();

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

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsider) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, false);
        std::ranges::sort(found_nodes);

        std::vector<OctreeNode<additional_cell_attributes>*> golden_nodes{};
        golden_nodes.reserve(number_neurons);

        std::stack<OctreeNode<additional_cell_attributes>*> stack{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Axon, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderDistributedTree) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    NodeCacheAdapter::set_node_cache_testing_purposes();

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto branching_level = SimulationAdapter::get_small_refinement_level(mt) + 1;

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, false);
        std::ranges::sort(found_nodes);

        std::vector<OctreeNode<additional_cell_attributes>*> golden_nodes{};
        golden_nodes.reserve(number_neurons);

        std::stack<OctreeNode<additional_cell_attributes>*> stack{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Axon, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderEarlyReturn) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, true);
        std::ranges::sort(found_nodes);

        std::vector<OctreeNode<additional_cell_attributes>*> golden_nodes{};
        golden_nodes.reserve(number_neurons);

        std::stack<OctreeNode<additional_cell_attributes>*> stack{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Axon, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderEarlyReturnDistributedTree) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    NodeCacheAdapter::set_node_cache_testing_purposes();

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto branching_level = SimulationAdapter::get_small_refinement_level(mt) + 1;

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, true);
        std::ranges::sort(found_nodes);

        std::vector<OctreeNode<additional_cell_attributes>*> golden_nodes{};
        golden_nodes.reserve(number_neurons);

        std::stack<OctreeNode<additional_cell_attributes>*> stack{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Axon, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand && current->get_mpi_rank() != MPIRank(0)) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderNoAxons) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_tree_no_axons<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes_axon = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, false);
        ASSERT_TRUE(found_nodes_axon.empty());
    }

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderNoElements) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_tree_no_synaptic_elements<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes_axon = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, false);
        ASSERT_TRUE(found_nodes_axon.empty());
    }

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutInvertedBaseTest, testNodesToConsiderNoDendrites) {
    using additional_cell_attributes = BarnesHutInvertedCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_tree_no_dendrites<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, acceptance_criterion, false);
        std::ranges::sort(found_nodes);

        std::vector<OctreeNode<additional_cell_attributes>*> golden_nodes{};
        golden_nodes.reserve(number_neurons);

        std::stack<OctreeNode<additional_cell_attributes>*> stack{};
        for (auto* child : root.get_children()) {
            if (child != nullptr) {
                stack.push(child);
            }
        }

        while (!stack.empty()) {
            auto* current = stack.top();
            stack.pop();

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Axon, searched_signal_type, acceptance_criterion);
            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Accept) {
                golden_nodes.emplace_back(current);
                continue;
            }

            if (ac == BarnesHutBase<additional_cell_attributes>::AcceptanceStatus::Expand) {
                for (auto* child : current->get_children()) {
                    if (child != nullptr) {
                        stack.push(child);
                    }
                }
            }
        }

        std::ranges::sort(golden_nodes);

        ASSERT_EQ(found_nodes, golden_nodes);
    }

    make_mpi_mem_available<additional_cell_attributes>();
}

