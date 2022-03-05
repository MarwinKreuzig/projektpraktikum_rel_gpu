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

#include "BarnesHutCell.h"
#include "BarnesHutBase.h"
#include "Types.h"
#include "algorithm/ExchangingAlgorithm.h"
#include "mpi/CommunicationMap.h"
#include "neurons/ElementType.h"
#include "neurons/SignalType.h"
#include "neurons/UpdateStatus.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"

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
class BarnesHutInverted : public BarnesHutBase<BarnesHutInvertedCell>, public BackwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse> {
public:
    using AdditionalCellAttributes = BarnesHutInvertedCell;
    using position_type = typename RelearnTypes::position_type;
    using counter_type = typename RelearnTypes::counter_type;

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
        if (node->is_child()) {
            return;
        }

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

protected:
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
     * @brief Finds target neurons for a specified source neuron
     * @param source_neuron_id The source neuron's id
     * @param source_position The source neuron's position
     * @param number_vacant_elements The number of vacant elements of the source neuron
     * @param root Where the source neuron should start to search for targets
     * @param element_type The element type the source neuron searches
     * @param signal_type The signal type the source neuron searches
     * @return A vector of pairs with (a) the target mpi rank and (b) the request for that rank
     */
    [[nodiscard]] std::vector<std::pair<int, SynapseCreationRequest>> find_target_neurons(const NeuronID& source_neuron_id, const position_type& source_position, const counter_type& number_vacant_elements,
        OctreeNode<AdditionalCellAttributes>* root, const ElementType element_type, const SignalType signal_type, const double sigma);

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param number_neurons The number of local neurons
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses from the local rank
     */
    [[nodiscard]] std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<LocalSynapses, DistantOutSynapses>>
    process_requests(size_t number_neurons, const CommunicationMap<SynapseCreationRequest>& RequestType) override;

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses to this MPI rank from other MPI ranks
     */
    [[nodiscard]] DistantInSynapses process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
        const CommunicationMap<SynapseCreationResponse>& creation_responses) override;

private:
    std::shared_ptr<OctreeImplementation<BarnesHutInverted>> global_tree{};
};
