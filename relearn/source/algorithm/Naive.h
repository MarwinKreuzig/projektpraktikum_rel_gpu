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

#include "Algorithm.h"
#include "NaiveCell.h"
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
class Naive : public Algorithm {
public:
    using AdditionalCellAttributes = NaiveCell;
    using position_type = AdditionalCellAttributes::position_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit Naive(const std::shared_ptr<OctreeImplementation<Naive>>& octree)
        : global_tree(octree) {
        RelearnException::check(octree != nullptr, "BarnesHut::BarnesHut: octree was null");
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
    static void update_functor(OctreeNode<NaiveCell>* node) {
        RelearnException::check(node != nullptr, "Naive::update_functor: node is nullptr");

        using counter_type = NaiveCell::counter_type;

        counter_type my_number_dendrites_excitatory = 0;
        counter_type my_number_dendrites_inhibitory = 0;

        for (const auto& child : node->get_children()) {
            if (child == nullptr) {
                continue;
            }

            // Sum up number of dendrites
            const auto child_number_dendrites_excitatory = child->get_cell().get_number_excitatory_dendrites();
            const auto child_number_dendrites_inhibitory = child->get_cell().get_number_inhibitory_dendrites();

            my_number_dendrites_excitatory += child_number_dendrites_excitatory;
            my_number_dendrites_inhibitory += child_number_dendrites_inhibitory;
        }

        node->set_cell_number_dendrites(my_number_dendrites_excitatory, my_number_dendrites_inhibitory);
    }

private:
    /**
     * @brief Returns an optional RankNeuronId that the algorithm determined for the given source neuron. No actual request is made.
     *      Might perform MPI communication via NodeCache::download_children()
     * @param initiator_neuron_id The neuron's id that wants to connect. Is used to disallow autapses (connections to itself)
     * @param axon_position The neuron's position that wants to connect. Is used in probability computations
     * @param dendrite_type_needed The signal type that is searched.
     * @return If the algorithm didn't find a matching neuron, the return value is empty.
     *      If the algorihtm found a matching neuron, it's id and MPI rank are returned.
     */
    [[nodiscard]] std::optional<RankNeuronId> find_target_neuron(const NeuronID& src_neuron_id, const position_type& axon_position, SignalType dendrite_type_needed);

    [[nodiscard]] double
    calc_attractiveness_to_connect(
        const NeuronID& src_neuron_id,
        const position_type& axon_position,
        const OctreeNode<NaiveCell>& node_with_dendrite,
        SignalType dendrite_type_needed) const;

    [[nodiscard]] std::vector<double> create_interval(
        const NeuronID& src_neuron_id,
        const position_type& axon_position,
        SignalType dendrite_type_needed,
        const std::vector<OctreeNode<NaiveCell>*>& vector) const;

    [[nodiscard]] static std::tuple<bool, bool> acceptance_criterion_test(
        const position_type& axon_position,
        const OctreeNode<NaiveCell>* node_with_dendrite,
        SignalType dendrite_type_needed);

    [[nodiscard]] static std::vector<OctreeNode<NaiveCell>*> get_nodes_for_interval(
        const position_type& axon_position,
        OctreeNode<NaiveCell>* root,
        SignalType dendrite_type_needed);

    std::shared_ptr<OctreeImplementation<Naive>> global_tree{};
};
