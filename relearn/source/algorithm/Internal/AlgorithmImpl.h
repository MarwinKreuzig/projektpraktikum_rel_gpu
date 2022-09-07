#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm/Algorithm.h"
#include "neurons/UpdateStatus.h"
#include "structure/Octree.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <memory>
#include <vector>

template <typename AdditionalCellAttributes>
class AlgorithmImpl : public Algorithm {
public:
    explicit AlgorithmImpl(const std::shared_ptr<OctreeImplementation<AdditionalCellAttributes>>& octree)
        : global_tree(octree) {
        RelearnException::check(octree != nullptr, "AlgorithmImpl::AlgorithmImpl: octree was null");
    }

    void update_octree(const std::vector<UpdateStatus>& disable_flags) override {
        // Update my leaf nodes
        Timers::start(TimerRegion::UPDATE_LEAF_NODES);
        update_leaf_nodes(disable_flags);
        Timers::stop_and_add(TimerRegion::UPDATE_LEAF_NODES);

        // Update the octree
        global_tree->synchronize_tree();
    }

    /**
     * @brief Updates all leaf nodes in the octree by the algorithm
     * @param disable_flags Flags that indicate if a neuron id disabled or enabled. If disabled, it won't be updated
     * @exception Throws a RelearnException if the number of flags is different than the number of leaf nodes, or if there is an internal error
     */
    void update_leaf_nodes(const std::vector<UpdateStatus>& disable_flags) override {
        const std::vector<double>& dendrites_excitatory_counts = excitatory_dendrites->get_grown_elements();
        const std::vector<unsigned int>& dendrites_excitatory_connected_counts = excitatory_dendrites->get_connected_elements();

        const std::vector<double>& dendrites_inhibitory_counts = inhibitory_dendrites->get_grown_elements();
        const std::vector<unsigned int>& dendrites_inhibitory_connected_counts = inhibitory_dendrites->get_connected_elements();

        const std::vector<double>& axons_counts = axons->get_grown_elements();
        const std::vector<unsigned int>& axons_connected_counts = axons->get_connected_elements();

        const auto& leaf_nodes = global_tree->get_leaf_nodes();
        const auto num_leaf_nodes = leaf_nodes.size();
        const auto num_disable_flags = disable_flags.size();
        const auto num_dendrites_excitatory_counts = dendrites_excitatory_counts.size();
        const auto num_dendrites_excitatory_connected_counts = dendrites_excitatory_connected_counts.size();
        const auto num_dendrites_inhibitory_counts = dendrites_inhibitory_counts.size();
        const auto num_dendrites_inhibitory_connected_counts = dendrites_inhibitory_connected_counts.size();

        const auto all_same_size = num_leaf_nodes == num_disable_flags
            && num_leaf_nodes == num_dendrites_excitatory_counts
            && num_leaf_nodes == num_dendrites_excitatory_connected_counts
            && num_leaf_nodes == num_dendrites_inhibitory_counts
            && num_leaf_nodes == num_dendrites_inhibitory_connected_counts;

        RelearnException::check(all_same_size, "AlgorithmImpl::update_leaf_nodes: The vectors were of different sizes");

        using counter_type = AdditionalCellAttributes::counter_type;

        for (const auto& neuron_id : NeuronID::range(num_leaf_nodes)) {
            const auto local_neuron_id = neuron_id.get_neuron_id();

            auto* node = leaf_nodes[local_neuron_id];
            RelearnException::check(node != nullptr, "AlgorithmImpl::update_leaf_nodes: node was nullptr: {}", neuron_id);

            const auto& cell = node->get_cell();
            const auto other_neuron_id = cell.get_neuron_id();
            RelearnException::check(neuron_id == other_neuron_id, "AlgorithmImpl::update_leaf_nodes: The nodes are not in order");

            if (disable_flags[local_neuron_id] == UpdateStatus::Disabled) {
                if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                    node->set_cell_number_excitatory_dendrites(0);
                }

                if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                    node->set_cell_number_inhibitory_dendrites(0);
                }

                if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                    node->set_cell_number_excitatory_axons(0);
                }

                if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                    node->set_cell_number_inhibitory_axons(0);
                }
                continue;
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_dendrite) {
                const auto number_vacant_excitatory_dendrites = excitatory_dendrites->get_free_elements(neuron_id);
                node->set_cell_number_excitatory_dendrites(number_vacant_excitatory_dendrites);
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_dendrite) {
                const auto number_vacant_inhibitory_dendrites = inhibitory_dendrites->get_free_elements(neuron_id);
                node->set_cell_number_inhibitory_dendrites(number_vacant_inhibitory_dendrites);
            }

            if constexpr (AdditionalCellAttributes::has_excitatory_axon) {
                const auto signal_type = axons->get_signal_type(neuron_id);

                if (signal_type == SignalType::Excitatory) {
                    const auto number_vacant_axons = axons->get_free_elements(neuron_id);
                    node->set_cell_number_excitatory_axons(number_vacant_axons);
                } else {
                    node->set_cell_number_excitatory_axons(0);
                }
            }

            if constexpr (AdditionalCellAttributes::has_inhibitory_axon) {
                const auto signal_type = axons->get_signal_type(neuron_id);

                if (signal_type == SignalType::Inhibitory) {
                    const auto number_vacant_axons = axons->get_free_elements(neuron_id);
                    node->set_cell_number_inhibitory_axons(number_vacant_axons);
                } else {
                    node->set_cell_number_inhibitory_axons(0);
                }
            }
        }
    }

protected:
    constexpr const std::shared_ptr<OctreeImplementation<AdditionalCellAttributes>>& get_octree() const noexcept {
        return global_tree;
    }

private:
    std::shared_ptr<OctreeImplementation<AdditionalCellAttributes>> global_tree{};
};