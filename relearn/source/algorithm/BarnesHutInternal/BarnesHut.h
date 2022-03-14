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
#include "algorithm/Connector.h"
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
 * It is strongly tied to Octree, and might perform MPI communication via NodeCache::download_children()
 */
class BarnesHut : public BarnesHutBase<BarnesHutCell>, public ForwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse> {
public:
    using AdditionalCellAttributes = BarnesHutCell;
    using position_type = typename RelearnTypes::position_type;
    using counter_type = typename RelearnTypes::counter_type;

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
     * @brief Updates all leaf nodes in the octree by the algorithm
     * @param disable_flags Flags that indicate if a neuron id disabled (0) or enabled (otherwise)
     * @param axons The model for the axons
     * @param excitatory_dendrites The model for the excitatory dendrites
     * @param inhibitory_dendrites The model for the inhibitory dendrites
     * @exception Throws a RelearnException if the vectors have different sizes or the leaf nodes are not in order of their neuron id
     */
    void update_leaf_nodes(const std::vector<UpdateStatus>& disable_flags) override;

    /**
     * @brief Updates the passed node with the values of its children according to the algorithm
     * @param node The node to update, must not be nullptr
     * @exception Throws a RelearnException if node is nullptr
     */
    static void update_functor(OctreeNode<BarnesHutCell>* node) {
        RelearnException::check(node != nullptr, "BarnesHut::update_functor: node is nullptr");

        // NOLINTNEXTLINE
        if (node->is_child()) {
            return;
        }

        // I'm inner node, i.e., I have a super neuron
        position_type my_position_dendrites_excitatory = { 0., 0., 0. };
        position_type my_position_dendrites_inhibitory = { 0., 0., 0. };

        // Sum of number of dendrites of all my children
        counter_type my_number_dendrites_excitatory = 0;
        counter_type my_number_dendrites_inhibitory = 0;

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
            std::optional<position_type> opt_child_position_dendrites_excitatory = child_cell.get_excitatory_dendrites_position();
            std::optional<position_type> opt_child_position_dendrites_inhibitory = child_cell.get_inhibitory_dendrites_position();

            // We can use position if it's valid or if corresponding num of axons is 0
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
            node->set_cell_excitatory_dendrites_position(std::optional<position_type>{ scaled_position });
        }

        if (0 == my_number_dendrites_inhibitory) {
            node->set_cell_inhibitory_dendrites_position({});
        } else {
            const auto scaled_position = my_position_dendrites_inhibitory / my_number_dendrites_inhibitory;
            node->set_cell_inhibitory_dendrites_position(std::optional<position_type>{ scaled_position });
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
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<LocalSynapses, DistantInSynapses>> 
        process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests) override {
        return ForwardConnector::process_requests(creation_requests, excitatory_dendrites, inhibitory_dendrites);
    }

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] DistantOutSynapses process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
        const CommunicationMap<SynapseCreationResponse>& creation_responses) override {
        return ForwardConnector::process_responses(creation_requests, creation_responses, axons);
    }

 private:
    std::shared_ptr<OctreeImplementation<BarnesHut>> global_tree{};
};
