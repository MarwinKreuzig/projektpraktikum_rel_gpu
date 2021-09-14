/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include "Algorithm.h"
#include "FastMultipoleMethodsCell.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"

#include <array>
#include <memory>
#include <vector>

template <typename T>
class OctreeImplementation;

class FastMultipoleMethods : public Algorithm {
public:
    using AdditionalCellAttributes = FastMultipoleMethodsCell;

    FastMultipoleMethods(const std::shared_ptr<OctreeImplementation<FastMultipoleMethods>>& octree)
        : global_tree(octree) {
        RelearnException::check(octree != nullptr, "In FastMultipoleMethods::FastMultipoleMethods, the octree was null");
    }

    MapSynapseCreationRequests find_target_neurons(size_t num_neurons, const std::vector<char>& disable_flags,
        const std::unique_ptr<NeuronsExtraInfo>& extra_infos, const std::unique_ptr<SynapticElements>& axons) override;

    void update_leaf_nodes(const std::vector<char>& disable_flags,
        const std::vector<double>& dendrites_excitatory_counts, const std::vector<unsigned int>& dendrites_excitatory_connected_counts,
        const std::vector<double>& dendrites_inhibitory_counts, const std::vector<unsigned int>& dendrites_inhibitory_connected_counts) override;

    /**
     * @brief Updates the passed node with the values of its children according to the algorithm
     * @param node The node to update, must not be nullptr
     * @exception Throws a RelearnException if node is nullptr
     */
    static void update_functor(OctreeNode<FastMultipoleMethodsCell>* node) {
        RelearnException::check(node != nullptr, "In FunctorUpdateNode, node is nullptr");

        // NOLINTNEXTLINE
        if (!node->is_parent()) {
            return;
        }

        // I'm inner node, i.e., I have a super neuron
        Vec3d my_position_dendrites_excitatory = { 0., 0., 0. };
        Vec3d my_position_dendrites_inhibitory = { 0., 0., 0. };

        // Sum of number of dendrites of all my children
        auto my_number_dendrites_excitatory = 0;
        auto my_number_dendrites_inhibitory = 0;

        // For all my children
        for (const auto& child : node->get_children()) {
            if (child == nullptr) {
                continue;
            }

            // Sum up number of dendrites
            const auto child_number_dendrites_excitatory = child->get_cell().get_number_excitatory_dendrites();
            const auto child_number_dendrites_inhibitory = child->get_cell().get_number_inhibitory_dendrites();

            my_number_dendrites_excitatory += child_number_dendrites_excitatory;
            my_number_dendrites_inhibitory += child_number_dendrites_inhibitory;

            // Average the position by using the number of dendrites as weights
            std::optional<Vec3d> child_position_dendrites_excitatory = child->get_cell().get_excitatory_dendrites_position();
            std::optional<Vec3d> child_position_dendrites_inhibitory = child->get_cell().get_inhibitory_dendrites_position();

            /**
			 * We can use position if it's valid or if corresponding num of dendrites is 0 
			 */
            RelearnException::check(child_position_dendrites_excitatory.has_value() || (0 == child_number_dendrites_excitatory), "temp position exc was bad");
            RelearnException::check(child_position_dendrites_inhibitory.has_value() || (0 == child_number_dendrites_inhibitory), "temp position inh was bad");

            if (child_position_dendrites_excitatory.has_value()) {
                const auto scaled_position = child_position_dendrites_excitatory.value() * static_cast<double>(child_number_dendrites_excitatory);
                my_position_dendrites_excitatory += scaled_position;
            }

            if (child_position_dendrites_inhibitory.has_value()) {
                const auto scaled_position = child_position_dendrites_inhibitory.value() * static_cast<double>(child_number_dendrites_inhibitory);
                my_position_dendrites_inhibitory += scaled_position;
            }
        }

        node->set_cell_num_dendrites(my_number_dendrites_excitatory, my_number_dendrites_inhibitory);

        /**
		 * For calculating the new weighted position, make sure that we don't
		 * divide by 0. This happens if the my number of dendrites is 0.
		 */
        if (0 == my_number_dendrites_excitatory) {
            node->set_cell_neuron_pos_exc({});
        } else {
            const auto scaled_position = my_position_dendrites_excitatory / my_number_dendrites_excitatory;
            node->set_cell_neuron_pos_exc(std::optional<Vec3d>{ scaled_position });
        }

        if (0 == my_number_dendrites_inhibitory) {
            node->set_cell_neuron_pos_inh({});
        } else {
            const auto scaled_position = my_position_dendrites_inhibitory / my_number_dendrites_inhibitory;
            node->set_cell_neuron_pos_inh(std::optional<Vec3d>{ scaled_position });
        }
    }

private:
    std::vector<double> calc_attractiveness_to_connect_FMM(const OctreeNode<FastMultipoleMethodsCell>* source, const std::array<const OctreeNode<FastMultipoleMethodsCell>*, 8>& interaction_list,
        const SignalType dendrite_type_needed) const;

    unsigned int do_random_experiment(const OctreeNode<FastMultipoleMethodsCell>* source, const std::vector<double>& attractiveness) const;

    std::vector<double> calc_attractiveness_to_connect_FMM(const OctreeNode<FastMultipoleMethodsCell>* source,
        const std::array<const OctreeNode<FastMultipoleMethodsCell>*, 8>& interaction_list, const SignalType dendrite_type_needed);

    void make_creation_request_for(SignalType needed, MapSynapseCreationRequests& request,
        std::stack<std::pair<OctreeNode<FastMultipoleMethodsCell>*, std::array<const OctreeNode<FastMultipoleMethodsCell>*, 8>>>& nodes_with_axons);

    std::shared_ptr<OctreeImplementation<FastMultipoleMethods>> global_tree{};
};
