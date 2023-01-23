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

TEST_F(BarnesHutBaseTest, testACException) {
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

TEST_F(BarnesHutBaseTest, testACLeafDendrites) {
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

TEST_F(BarnesHutBaseTest, testACParentAxon) {
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

TEST_F(BarnesHutBaseTest, testACParentDendrite) {
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

TEST_F(BarnesHutBaseTest, testNodesToConsider) {
    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, false);
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

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
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

TEST_F(BarnesHutBaseTest, testNodesToConsiderDistributedTree) {
    using additional_cell_attributes = BarnesHutCell;

    NodeCacheAdapter::set_node_cache_testing_purposes();

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto branching_level = SimulationAdapter::get_small_refinement_level(mt) + 1;

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, false);
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

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
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

TEST_F(BarnesHutBaseTest, testNodesToConsiderEarlyReturn) {
    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, true);
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

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
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

TEST_F(BarnesHutBaseTest, testNodesToConsiderEarlyReturnDistributedTree) {
    using additional_cell_attributes = BarnesHutCell;

    NodeCacheAdapter::set_node_cache_testing_purposes();

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto branching_level = SimulationAdapter::get_small_refinement_level(mt) + 1;

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);
    OctreeAdapter::mark_node_as_distributed(&root, branching_level);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, true);
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

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
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

TEST_F(BarnesHutBaseTest, testNodesToConsiderNoAxons) {
    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_tree_no_axons<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, false);
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

            const auto ac = BarnesHutBase<additional_cell_attributes>::test_acceptance_criterion(position, current, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
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

TEST_F(BarnesHutBaseTest, testNodesToConsiderNoElements) {
    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_tree_no_synaptic_elements<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes_dendrite = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, false);
        ASSERT_TRUE(found_nodes_dendrite.empty());
    }

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutBaseTest, testNodesToConsiderNoDendrites) {
    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_tree_no_dendrites<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (auto it = 0; it < 1000; it++) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_nodes_dendrite = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion, false);
        ASSERT_TRUE(found_nodes_dendrite.empty());
    }

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutBaseTest, testNodesToConsiderException) {
    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt);
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);
    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    const Vec3d position{ 0.0 };
    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto too_small_acceptance_criterion = RandomAdapter::get_random_double<double>(-1000.0, 0.0, mt);
    const auto too_large_acceptance_criterion = RandomAdapter::get_random_double<double>(Constants::bh_max_theta + eps, 10000.0, mt);

    ASSERT_TRUE(BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, nullptr, ElementType::Dendrite, searched_signal_type, Constants::bh_default_theta).empty());
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Dendrite, searched_signal_type, 0.0);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Dendrite, searched_signal_type, Constants::bh_max_theta + eps);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Dendrite, searched_signal_type, too_small_acceptance_criterion), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Dendrite, searched_signal_type, too_large_acceptance_criterion), RelearnException);

    ASSERT_TRUE(BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, nullptr, ElementType::Axon, searched_signal_type, Constants::bh_default_theta).empty());
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, 0.0);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, Constants::bh_max_theta + eps);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, too_small_acceptance_criterion), RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::get_nodes_to_consider(position, &root, ElementType::Axon, searched_signal_type, too_large_acceptance_criterion), RelearnException);

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronException) {
    using additional_cell_attributes = BarnesHutCell;

    const NeuronID neuron_id(1000000);
    const Vec3d position{ 0.0 };

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto too_small_acceptance_criterion = RandomAdapter::get_random_double<double>(-1000.0, 0.0, mt);
    const auto too_large_acceptance_criterion = RandomAdapter::get_random_double<double>(Constants::bh_max_theta + eps, 10000.0, mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), neuron_id }, position, nullptr, ElementType::Dendrite, searched_signal_type, Constants::bh_default_theta);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Dendrite, searched_signal_type, 0.0);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Dendrite, searched_signal_type, Constants::bh_max_theta + eps);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Dendrite, searched_signal_type, too_small_acceptance_criterion);, RelearnException);
    ASSERT_THROW(auto val = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Dendrite, searched_signal_type, too_large_acceptance_criterion);, RelearnException);

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronNoChoice) {
    using additional_cell_attributes = BarnesHutCell;

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto root_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> root{};
    root.set_level(0);
    root.set_rank(MPIRank::root_rank());
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_number_excitatory_dendrites(2);
    root.set_cell_number_inhibitory_dendrites(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    for (auto it = 0; it < 1000; it++) {
        const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto first_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(0) }, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_FALSE(first_target_opt.has_value());

        auto second_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(1) }, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(second_target_opt.has_value());

        auto [second_rank, second_id] = second_target_opt.value();
        ASSERT_EQ(second_rank, MPIRank::root_rank());
        ASSERT_EQ(second_id, NeuronID(0));
    }

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronOneChoice) {
    using additional_cell_attributes = BarnesHutCell;

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto root_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> root{};
    root.set_level(0);
    root.set_rank(MPIRank::root_rank());
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(2));

    const auto first_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);
    auto _1 = root.insert(first_position, NeuronID(0));

    const auto second_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);
    auto _2 = root.insert(second_position, NeuronID(1));

    auto* first_node = OctreeAdapter::find_node<additional_cell_attributes>({ MPIRank::root_rank(), NeuronID(0) }, &root);
    auto* second_node = OctreeAdapter::find_node<additional_cell_attributes>({ MPIRank::root_rank(), NeuronID(1) }, &root);

    first_node->set_cell_number_excitatory_dendrites(1);
    first_node->set_cell_number_inhibitory_dendrites(1);
    second_node->set_cell_number_excitatory_dendrites(2);
    second_node->set_cell_number_inhibitory_dendrites(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
    const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
    const auto& position = SimulationAdapter::get_random_position(mt);

    auto first_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(0) }, first_position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
    ASSERT_TRUE(first_target_opt.has_value());

    auto [first_rank, first_id] = first_target_opt.value();
    ASSERT_EQ(first_rank, MPIRank(0));
    ASSERT_EQ(first_id, NeuronID(1));

    auto second_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(1) }, second_position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
    ASSERT_TRUE(second_target_opt.has_value());

    auto [second_rank, second_id] = second_target_opt.value();
    ASSERT_EQ(second_rank, MPIRank(0));
    ASSERT_EQ(second_id, NeuronID(0));

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronFullChoice) {
    using additional_cell_attributes = BarnesHutCell;

    const auto number_neurons = TaggedIdAdapter::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    auto root = OctreeAdapter::get_standard_tree<additional_cell_attributes>(number_neurons, minimum, maximum, mt);

    const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);

    for (const auto neuron_id : NeuronID::range(number_neurons)) {
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto found_target = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), neuron_id }, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);

        ASSERT_TRUE(found_target.has_value());
        auto [target_rank, target_id] = found_target.value();

        ASSERT_EQ(target_rank, MPIRank::root_rank());
        ASSERT_NE(neuron_id, target_id);
    }

    make_mpi_mem_available<additional_cell_attributes>();
}

TEST_F(BarnesHutBaseTest, testFindTargetNeuronNoChoiceDistributed) {
    using additional_cell_attributes = BarnesHutCell;

    NodeCacheAdapter::set_node_cache_testing_purposes();

    const auto& [minimum, maximum] = SimulationAdapter::get_random_simulation_box_size(mt);

    const auto root_position = SimulationAdapter::get_random_position_in_box(minimum, maximum, mt);

    OctreeNode<additional_cell_attributes> root{};
    root.set_level(0);
    root.set_rank(MPIRank(1));
    root.set_cell_size(minimum, maximum);
    root.set_cell_neuron_position(root_position);
    root.set_cell_neuron_id(NeuronID(0));
    root.set_cell_number_excitatory_dendrites(2);
    root.set_cell_number_inhibitory_dendrites(2);

    OctreeNodeUpdater<additional_cell_attributes>::update_tree(&root);

    for (auto it = 0; it < 1000; it++) {
        const auto searched_signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
        const auto acceptance_criterion = RandomAdapter::get_random_double<double>(eps, Constants::bh_max_theta, mt);
        const auto& position = SimulationAdapter::get_random_position(mt);

        auto first_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(0) }, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(first_target_opt.has_value());

        auto [first_rank, first_id] = first_target_opt.value();
        ASSERT_EQ(first_rank, MPIRank(1));
        ASSERT_EQ(first_id, NeuronID(0));

        auto second_target_opt = BarnesHutBase<additional_cell_attributes>::find_target_neuron({ MPIRank::root_rank(), NeuronID(1) }, position, &root, ElementType::Dendrite, searched_signal_type, acceptance_criterion);
        ASSERT_TRUE(second_target_opt.has_value());

        auto [second_rank, second_id] = second_target_opt.value();
        ASSERT_EQ(second_rank, MPIRank(1));
        ASSERT_EQ(second_id, NeuronID(0));
    }

    make_mpi_mem_available<additional_cell_attributes>();
}
