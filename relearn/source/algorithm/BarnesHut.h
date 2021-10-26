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

#include "../neurons/SignalType.h"
#include "../neurons/helper/RankNeuronId.h"
#include "../neurons/helper/SynapseCreationRequests.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"
#include "Algorithm.h"
#include "BarnesHutCell.h"

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
class BarnesHut : public Algorithm {
public:
    using AdditionalCellAttributes = BarnesHutCell;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit BarnesHut(const std::shared_ptr<OctreeImplementation<BarnesHut>>& octree)
        : global_tree(octree) {
        RelearnException::check(octree != nullptr, "BarnesHut::BarnesHut: octree was null");
    }

    /**
     * @brief Sets acceptance criterion for cells in the tree
     * @param acceptance_criterion The acceptance criterion, >= 0.0
     * @exception Throws a RelearnException if acceptance_criterion < 0.0
     */
    void set_acceptance_criterion(const double acceptance_criterion) {
        RelearnException::check(acceptance_criterion > 0.0, "BarnesHut::set_acceptance_criterion: acceptance_criterion was less than or equal to 0 ({})", acceptance_criterion);
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
     * @param num_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so (== 0), the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @param axons The axon model that is used
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    [[nodiscard]] MapSynapseCreationRequests find_target_neurons(size_t num_neurons, const std::vector<char>& disable_flags,
        const std::unique_ptr<NeuronsExtraInfo>& extra_infos, const std::unique_ptr<SynapticElements>& axons) override;

    /**
     * @brief Updates all leaf nodes in the octree by the algorithm
     * @param disable_flags Flags that indicate if a neuron id disabled (0) or enabled (otherwise)
     * @param axons The model for the axons
     * @param excitatory_dendrites The model for the excitatory dendrites
     * @param inhibitory_dendrites The model for the inhibitory dendrites
     * @exception Throws a RelearnException if the vectors have different sizes or the leaf nodes are not in order of their neuron id
     */
    void update_leaf_nodes(const std::vector<char>& disable_flags, const std::unique_ptr<SynapticElements>& axons,
        const std::unique_ptr<SynapticElements>& excitatory_dendrites, const std::unique_ptr<SynapticElements>& inhibitory_dendrites) override;

    /**
     * @brief Updates the passed node with the values of its children according to the algorithm
     * @param node The node to update, must not be nullptr
     * @exception Throws a RelearnException if node is nullptr
     */
    static void update_functor(OctreeNode<BarnesHutCell>* node) {
        RelearnException::check(node != nullptr, "BarnesHut::update_functor: node is nullptr");

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

            const auto& child_cell = child->get_cell();

            // Sum up number of dendrites
            const auto child_number_dendrites_excitatory = child_cell.get_number_excitatory_dendrites();
            const auto child_number_dendrites_inhibitory = child_cell.get_number_inhibitory_dendrites();

            my_number_dendrites_excitatory += child_number_dendrites_excitatory;
            my_number_dendrites_inhibitory += child_number_dendrites_inhibitory;

            // Average the position by using the number of dendrites as weights
            std::optional<Vec3d> opt_child_position_dendrites_excitatory = child_cell.get_excitatory_dendrites_position();
            std::optional<Vec3d> opt_child_position_dendrites_inhibitory = child_cell.get_inhibitory_dendrites_position();

            /**
			 * We can use position if it's valid or if corresponding num of dendrites is 0 
			 */
            RelearnException::check(opt_child_position_dendrites_excitatory.has_value() || (0 == child_number_dendrites_excitatory), "BarnesHut::update_functor: The child had excitatory dendrites, but no position. ID: {}", child->get_cell_neuron_id());
            RelearnException::check(opt_child_position_dendrites_inhibitory.has_value() || (0 == child_number_dendrites_inhibitory), "BarnesHut::update_functor: The child had inhibitory dendrites, but no position. ID: {}", child->get_cell_neuron_id());

            if (opt_child_position_dendrites_excitatory.has_value()) {
                const auto& child_position_dendrites_excitatory = opt_child_position_dendrites_excitatory.value();

                const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                const auto is_in_box = child_position_dendrites_excitatory.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                RelearnException::check(is_in_box, "BarnesHut::update_functor: The excitatory child is not in its cell");

                const auto& scaled_position = child_position_dendrites_excitatory * static_cast<double>(child_number_dendrites_excitatory);
                my_position_dendrites_excitatory += scaled_position;
            }

            if (opt_child_position_dendrites_inhibitory.has_value()) {
                const auto& child_position_dendrites_inhibitory = opt_child_position_dendrites_inhibitory.value();

                const auto& [child_cell_xyz_min, child_cell_xyz_max] = child_cell.get_size();
                const auto is_in_box = child_position_dendrites_inhibitory.check_in_box(child_cell_xyz_min, child_cell_xyz_max);

                RelearnException::check(is_in_box, "BarnesHut::update_functor: The inhibitory child is not in its cell");

                const auto& scaled_position = child_position_dendrites_inhibitory * static_cast<double>(child_number_dendrites_inhibitory);
                my_position_dendrites_inhibitory += scaled_position;
            }
        }

        node->set_cell_number_dendrites(my_number_dendrites_excitatory, my_number_dendrites_inhibitory);

        /**
		 * For calculating the new weighted position, make sure that we don't
		 * divide by 0. This happens if the my number of dendrites is 0.
		 */
        if (0 == my_number_dendrites_excitatory) {
            node->set_cell_excitatory_dendrites_position({});
        } else {
            const auto scaled_position = my_position_dendrites_excitatory / my_number_dendrites_excitatory;
            node->set_cell_excitatory_dendrites_position(std::optional<Vec3d>{ scaled_position });
        }

        if (0 == my_number_dendrites_inhibitory) {
            node->set_cell_inhibitory_dendrites_position({});
        } else {
            const auto scaled_position = my_position_dendrites_inhibitory / my_number_dendrites_inhibitory;
            node->set_cell_inhibitory_dendrites_position(std::optional<Vec3d>{ scaled_position });
        }
    }

private:
    /**
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::download_children()
     * @param src_neuron_id The neuron's id that wants to connect. Is used to disallow autapses (connections to itself)
     * @param axon_pos_xyz The neuron's position that wants to connect. Is used in probability computations
     * @param dendrite_type_needed The signal type that is searched.
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorihtm found a matching neuron, it's id and MPI rank are returned.
     */
    [[nodiscard]] std::optional<RankNeuronId> find_target_neuron(size_t src_neuron_id, const Vec3d& axon_pos_xyz, SignalType dendrite_type_needed);

    [[nodiscard]] double
    calc_attractiveness_to_connect(
        size_t src_neuron_id,
        const Vec3d& axon_pos_xyz,
        const OctreeNode<BarnesHutCell>& node_with_dendrite,
        SignalType dendrite_type_needed) const;

    [[nodiscard]] std::pair<double, std::vector<double>> create_interval(
        size_t src_neuron_id,
        const Vec3d& axon_pos_xyz,
        SignalType dendrite_type_needed,
        const std::vector<OctreeNode<BarnesHutCell>*>& vector) const;

    [[nodiscard]] std::tuple<bool, bool> acceptance_criterion_test(
        const Vec3d& axon_pos_xyz,
        const OctreeNode<BarnesHutCell>* node_with_dendrite,
        SignalType dendrite_type_needed) const;

    [[nodiscard]] std::vector<OctreeNode<BarnesHutCell>*> get_nodes_for_interval(
        const Vec3d& axon_pos_xyz,
        OctreeNode<BarnesHutCell>* root,
        SignalType dendrite_type_needed);

    double acceptance_criterion{ default_theta }; // Acceptance criterion

    std::shared_ptr<OctreeImplementation<BarnesHut>> global_tree{};

public:
    constexpr static double default_theta{ 0.3 };

    constexpr static double max_theta{ 0.5 };
};
