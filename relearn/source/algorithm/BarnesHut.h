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
#include "BarnesHutCell.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"

#include <tuple>

template <typename T>
class OctreeImplementation;

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
    BarnesHut(const std::shared_ptr<OctreeImplementation<BarnesHut>>& octree)
        : global_tree(octree) {
        RelearnException::check(octree != nullptr, "In BarnesHut::BarnesHut, the octree was null");
    }

    /**
     * @brief Sets acceptance criterion for cells in the tree
     * @param acceptance_criterion The acceptance criterion, >= 0.0
     * @exception Throws a RelearnException if acceptance_criterion < 0.0
     */
    void set_acceptance_criterion(double acceptance_criterion) {
        RelearnException::check(acceptance_criterion >= 0.0, "In BarnesHut::set_acceptance_criterion, acceptance_criterion was less than 0");
        this->acceptance_criterion = acceptance_criterion;

        if (acceptance_criterion == 0.0) {
            naive_method = true;
        } else {
            naive_method = false;
        }
    }

    /**
     * @brief Sets probability parameter used to determine the probability for a cell of being selected
     * @param sigma The probability parameter, >= 0.0
     * @exception Throws a RelearnExeption if sigma < 0.0
     */
    void set_probability_parameter(double sigma) {
        RelearnException::check(sigma > 0.0, "In BarnesHut::set_probability_parameter, sigma was not greater than 0");
        this->sigma = sigma;
    }

    /**
     * @brief Returns a boolean indicating if the naive version is used (acceptance_criterion == 0.0)
     * @return True iff the naive version is used
     */
    [[nodiscard]] bool is_naive_method_used() const noexcept {
        return naive_method;
    }

    /**
     * @brief Returns the currently used probability parameter
     * @return The currently used probability parameter
     */
    [[nodiscard]] double get_probabilty_parameter() const noexcept {
        return sigma;
    }

    /**
     * @brief Returns the currently used acceptance criterion
     * @return The currently used acceptance criterion
     */
    [[nodiscard]] double get_acceptance_criterion() const noexcept {
        return acceptance_criterion;
    }

    /**
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::download_children()
     * @param src_neuron_id The neuron's id that wants to connect. Is used to disallow autapses (connections to itself)
     * @param axon_pos_xyz The neuron's position that wants to connect. Is used in probability computations
     * @param dendrite_type_needed The signal type that is searched.
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorihtm found a matching neuron, it's id and MPI rank are returned.
     */
    [[nodiscard]] std::optional<RankNeuronId> find_target_neuron(size_t src_neuron_id, const Vec3d& axon_pos_xyz, SignalType dendrite_type_needed) override;

    /**
     * @brief Updates all leaf nodes in the octree by the algorithm
     * @param disable_flags Flags that indicate if a neuron id disabled (0) or enabled (otherwise)
     * @param dendrites_excitatory_counts The number of total excitatory dendrites, accessed via operator[] with the neuron ids
     * @param dendrites_excitatory_connected_counts The number of connected excitatory dendrites, accessed via operator[] with the neuron ids
     * @param dendrites_inhibitory_counts The number of total inhibitory dendrites, accessed via operator[] with the neuron ids
     * @param dendrites_inhibitory_connected_counts The number of connected inhibitory dendrites, accessed via operator[] with the neuron ids
     * @exception Throws a RelearnException if the vectors have different sizes or the leaf nodes are not in order of their neuron id
     */
    void update_leaf_nodes(const std::vector<char>& disable_flags,
        const std::vector<double>& dendrites_excitatory_counts, const std::vector<unsigned int>& dendrites_excitatory_connected_counts,
        const std::vector<double>& dendrites_inhibitory_counts, const std::vector<unsigned int>& dendrites_inhibitory_connected_counts) override;

    /**
     * @brief Updates the passed node with the values of its children according to the algorithm
     * @param node The node to update, must not be nullptr
     * @exception Throws a RelearnException if node is nullptr
     */
    static void update_functor(OctreeNode<BarnesHutCell>* node) {
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
    [[nodiscard]] double
    calc_attractiveness_to_connect(
        size_t src_neuron_id,
        const Vec3d& axon_pos_xyz,
        const OctreeNode<BarnesHutCell>& node_with_dendrite,
        SignalType dendrite_type_needed) const;

    [[nodiscard]] std::vector<double> create_interval(size_t src_neuron_id, const Vec3d& axon_pos_xyz, SignalType dendrite_type_needed, const std::vector<OctreeNode<BarnesHutCell>*>& vector) const;

    [[nodiscard]] std::tuple<bool, bool> acceptance_criterion_test(const Vec3d& axon_pos_xyz,
        const OctreeNode<BarnesHutCell>* const node_with_dendrite,
        SignalType dendrite_type_needed) const;

    [[nodiscard]] std::vector<OctreeNode<BarnesHutCell>*> get_nodes_for_interval(
        const Vec3d& axon_pos_xyz,
        OctreeNode<BarnesHutCell>* root,
        SignalType dendrite_type_needed);

    double acceptance_criterion{ default_theta }; // Acceptance criterion
    double sigma{ default_sigma }; // Probability parameter
    bool naive_method{ default_theta == 0.0 }; // If true, expand every cell regardless of whether dendrites are available or not

    std::shared_ptr<OctreeImplementation<BarnesHut>> global_tree{};

public:
    constexpr static double default_theta{ 0.3 };
    constexpr static double default_sigma{ 750.0 };

    constexpr static double max_theta{ 0.5 };
};
