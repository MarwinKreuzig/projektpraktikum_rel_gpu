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

#include "../neurons/helper/RankNeuronId.h"
#include "../neurons/SignalType.h"

#include "../util/RelearnException.h"
#include "../util/Vec3.h"

#include <optional>
#include <tuple>
#include <vector>

class Octree;
class OctreeNode;

/**
 * This class represents the implementation and adaptation of the Barnes Hut algorithm. The parameters can be set on the fly.
 * It is strongly tied to Octree, which might perform MPI communication via Octree::downloadChildren()
 */
class BarnesHut {
public:
    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    BarnesHut(const std::shared_ptr<Octree>& octree)
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
     *      Might perform MPI communication via Octree::downloadChildren()
     * @param src_neuron_id The neuron's id that wants to connect. Is used to disallow autapses (connections to itself)
     * @param axon_pos_xyz The neuorn's position that wants to connect. Is used in probability computations
     * @param dendrite_type_needed The signal type that is searched.
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorihtm found a matching neuron, it's id and MPI rank are returned.
     */
    [[nodiscard]] std::optional<RankNeuronId> find_target_neuron(size_t src_neuron_id, const Vec3d& axon_pos_xyz, SignalType dendrite_type_needed);

private:
    [[nodiscard]] double calc_attractiveness_to_connect(
        size_t src_neuron_id,
        const Vec3d& axon_pos_xyz,
        const OctreeNode& node_with_dendrite,
        SignalType dendrite_type_needed) const;

    [[nodiscard]] std::vector<double> create_interval(size_t src_neuron_id, const Vec3d& axon_pos_xyz, SignalType dendrite_type_needed, const std::vector<OctreeNode*>& vector) const;

    [[nodiscard]] std::tuple<bool, bool> acceptance_criterion_test(const Vec3d& axon_pos_xyz,
        const OctreeNode* const node_with_dendrite,
        SignalType dendrite_type_needed,
        bool naive_method) const;

    [[nodiscard]] std::vector<OctreeNode*> get_nodes_for_interval(
        const Vec3d& axon_pos_xyz,
        OctreeNode* root,
        SignalType dendrite_type_needed,
        bool naive_method);

    double acceptance_criterion{ default_theta }; // Acceptance criterion
    double sigma{ default_sigma }; // Probability parameter
    bool naive_method{ default_theta == 0.0 }; // If true, expand every cell regardless of whether dendrites are available or not

    std::shared_ptr<Octree> global_tree;

public:
    constexpr static double default_theta{ 0.3 };
    constexpr static double default_sigma{ 750.0 };

    constexpr static double max_theta{ 0.5 };
};
