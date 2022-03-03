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

#include "BarnesHutInvertedCell.h"
#include "Types.h"
#include "algorithm/Algorithm.h"
#include "neurons/SignalType.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

class NeuronsExtraInfo;
template <typename T>
class OctreeImplementation;
class SynapticElements;

/**
 * This class represents the implementation and adaptation of the Barnes Hut algorithm. The parameters can be set on the fly.
 * It is strongly tied to Octree, which might perform MPI communication via NodeCache::download_children()
 */
class BarnesHutInverted : public Algorithm {
public:
    /**
     * This enum indicates for an OctreeNode what the acceptance status is
     * It can be:
     * - Discard (no axons there)
     * - Expand (would be too much approximation, need to expand)
     * - Accept (can use the node for the algorithm)
     */
    enum class AcceptanceStatus : char {
        Discard = 0,
        Expand = 1,
        Accept = 2,
    };

    using AdditionalCellAttributes = BarnesHutInvertedCell;

    using position_type = BarnesHutInvertedCell::position_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit BarnesHutInverted(const std::shared_ptr<OctreeImplementation<BarnesHutInverted>>& octree)
        : global_tree(octree) {
        RelearnException::check(octree != nullptr, "BarnesHutInverted::BarnesHutInverted: octree was null");
    }

    /**
     * @brief Sets acceptance criterion for cells in the tree
     * @param acceptance_criterion The acceptance criterion, >= 0.0
     * @exception Throws a RelearnException if acceptance_criterion < 0.0
     */
    void set_acceptance_criterion(const double acceptance_criterion) {
        RelearnException::check(acceptance_criterion > 0.0, "BarnesHutInverted::set_acceptance_criterion: acceptance_criterion was less than or equal to 0 ({})", acceptance_criterion);
        this->acceptance_criterion = acceptance_criterion;
    }

    /**
     * @brief Returns the currently used acceptance criterion
     * @return The currently used acceptance criterion
     */
    [[nodiscard]] double get_acceptance_criterion() const noexcept {
        return acceptance_criterion;
    }

    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons
     * @param number_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so (== 0), the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @param axons The axon model that is used
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    [[nodiscard]] CommunicationMap<SynapseCreationRequest> find_target_neurons(size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
        const std::unique_ptr<NeuronsExtraInfo>& extra_infos) override;

    /**
     * @brief Updates all leaf nodes in the octree by the algorithm
     * @param disable_flags Flags that indicate if a neuron id disabled (0) or enabled (otherwise)
     * @param axons The model for the axons
     * @param excitatory_axons The model for the excitatory axons
     * @param inhibitory_axons The model for the inhibitory axons
     * @exception Throws a RelearnException if the vectors have different sizes or the leaf nodes are not in order of their neuron id
     */
    void update_leaf_nodes(const std::vector<UpdateStatus>& disable_flags) override;

    /**
     * @brief Updates the passed node with the values of its children according to the algorithm
     * @param node The node to update, must not be nullptr
     * @exception Throws a RelearnException if node is nullptr
     */
    static void update_functor(OctreeNode<BarnesHutInvertedCell>* node) {
        RelearnException::check(node != nullptr, "BarnesHutInverted::update_functor: node is nullptr");

        // NOLINTNEXTLINE
        if (!node->is_parent()) {
            return;
        }

        using position_type = BarnesHutInvertedCell::position_type;
        using counter_type = BarnesHutInvertedCell::counter_type;

        // I'm inner node, i.e., I have a super neuron
        position_type my_position_axons_excitatory = { 0., 0., 0. };
        position_type my_position_axons_inhibitory = { 0., 0., 0. };

        // Sum of number of axons of all my children
        counter_type my_number_axons_excitatory = 0;
        counter_type my_number_axons_inhibitory = 0;

        // For all my children
        for (const auto& child : node->get_children()) {
            if (child == nullptr) {
                continue;
            }

            const auto& child_cell = child->get_cell();

            // Sum up number of axons
            const auto child_number_axons_excitatory = child_cell.get_number_excitatory_axons();
            const auto child_number_axons_inhibitory = child_cell.get_number_inhibitory_axons();

            my_number_axons_excitatory += child_number_axons_excitatory;
            my_number_axons_inhibitory += child_number_axons_inhibitory;

            // Average the position by using the number of axons as weights
            std::optional<position_type> opt_child_position_axons_excitatory = child_cell.get_excitatory_axons_position();
            std::optional<position_type> opt_child_position_axons_inhibitory = child_cell.get_inhibitory_axons_position();

            // We can use position if it's valid or if corresponding num of axons is 0
            RelearnException::check(opt_child_position_axons_excitatory.has_value() || (0 == child_number_axons_excitatory), "BarnesHutInverted::update_functor: The child had excitatory axons, but no position. ID: {}", child->get_cell_neuron_id());
            RelearnException::check(opt_child_position_axons_inhibitory.has_value() || (0 == child_number_axons_inhibitory), "BarnesHutInverted::update_functor: The child had inhibitory axons, but no position. ID: {}", child->get_cell_neuron_id());

            if (opt_child_position_axons_excitatory.has_value()) {
                const auto& child_position_axons_excitatory = opt_child_position_axons_excitatory.value();

                const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                const auto is_in_box = child_position_axons_excitatory.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                RelearnException::check(is_in_box, "BarnesHutInverted::update_functor: The excitatory child is not in its cell");

                const auto& scaled_position = child_position_axons_excitatory * static_cast<double>(child_number_axons_excitatory);
                my_position_axons_excitatory += scaled_position;
            }

            if (opt_child_position_axons_inhibitory.has_value()) {
                const auto& child_position_axons_inhibitory = opt_child_position_axons_inhibitory.value();

                const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                const auto is_in_box = child_position_axons_inhibitory.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                RelearnException::check(is_in_box, "BarnesHutInverted::update_functor: The inhibitory child is not in its cell");

                const auto& scaled_position = child_position_axons_inhibitory * static_cast<double>(child_number_axons_inhibitory);
                my_position_axons_inhibitory += scaled_position;
            }
        }

        node->set_cell_number_axons(my_number_axons_excitatory, my_number_axons_inhibitory);

        /**
         * For calculating the new weighted position, make sure that we don't
         * divide by 0. This happens if the my number of axons is 0.
         */
        if (0 == my_number_axons_excitatory) {
            node->set_cell_excitatory_axons_position({});
        } else {
            const auto scaled_position = my_position_axons_excitatory / my_number_axons_excitatory;
            node->set_cell_excitatory_axons_position(std::optional<position_type>{ scaled_position });
        }

        if (0 == my_number_axons_inhibitory) {
            node->set_cell_inhibitory_axons_position({});
        } else {
            const auto scaled_position = my_position_axons_inhibitory / my_number_axons_inhibitory;
            node->set_cell_inhibitory_axons_position(std::optional<position_type>{ scaled_position });
        }
    }

private:
    /**
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::download_children()
     * @param initiator_neuron_id The neuron's id that wants to connect. Is used to disallow autapses (connections to itself)
     * @param dendrite_position The neuron's position that wants to connect. Is used in probability computations
     * @param axon_type_needed The signal type that is searched.
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorihtm found a matching neuron, it's id and MPI rank are returned.
     */
    [[nodiscard]] std::optional<RankNeuronId> find_target_neuron(const NeuronID& src_neuron_id, const position_type& dendrite_position, SignalType axon_type_needed);

    [[nodiscard]] double
    calc_attractiveness_to_connect(
        const NeuronID& src_neuron_id,
        const position_type& dendrite_position,
        const OctreeNode<BarnesHutInvertedCell>& node_with_axon,
        SignalType axon_type_needed) const;

    [[nodiscard]] std::pair<double, std::vector<double>> create_interval(
        const NeuronID& src_neuron_id,
        const position_type& dendrite_position,
        SignalType axon_type_needed,
        const std::vector<OctreeNode<BarnesHutInvertedCell>*>& vector) const;

    [[nodiscard]] AcceptanceStatus acceptance_criterion_test(
        const position_type& dendrite_position,
        const OctreeNode<BarnesHutInvertedCell>* node_with_axon,
        SignalType axon_type_needed) const;

    [[nodiscard]] std::vector<OctreeNode<BarnesHutInvertedCell>*> get_nodes_for_interval(
        const position_type& dendrite_position,
        OctreeNode<BarnesHutInvertedCell>* root,
        SignalType axon_type_needed);

    double acceptance_criterion{ default_theta }; // Acceptance criterion

    std::shared_ptr<OctreeImplementation<BarnesHutInverted>> global_tree{};

public:
    constexpr static double default_theta{ 0.3 };

    constexpr static double max_theta{ 0.5 };
};
