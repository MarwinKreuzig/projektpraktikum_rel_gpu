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

#include "Config.h"
#include "FastMultipoleMethodsCell.h"
#include "FastMultipoleMethodsBase.h"
#include "Types.h"
#include "algorithm/Connector.h"
#include "algorithm/ExchangingAlgorithm.h"
#include "neurons/UpdateStatus.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/OctreeNode.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Stack.h"
#include "util/Utility.h"
#include "util/Stack.h"
#include <array>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

template <typename T>
class OctreeImplementation;

/**
 * This class represents the implementation and adaptation of fast multipole methods. The parameters can be set on the fly.
 * It is strongly tied to Octree, and might perform MPI communication via NodeCache::download_children()
 */
class FastMultipoleMethods : public ForwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse> {
    friend class FMMTest;
    std::shared_ptr<OctreeImplementation<FastMultipoleMethods>> global_tree{};

public:
    using AdditionalCellAttributes = FastMultipoleMethodsCell;
    using interaction_list_type = std::vector<OctreeNode<AdditionalCellAttributes>*>;
    using position_type = typename Cell<AdditionalCellAttributes>::position_type;
    using counter_type = typename Cell<AdditionalCellAttributes>::counter_type;
    using stack_entry = FastMultipoleMethodsBase<AdditionalCellAttributes>::stack_entry;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit FastMultipoleMethods(const std::shared_ptr<OctreeImplementation<FastMultipoleMethods>>& octree)
        : global_tree(octree) {
        RelearnException::check(octree != nullptr, "FastMultipoleMethods::FastMultipoleMethods: octree was null");
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
    static void update_functor(OctreeNode<FastMultipoleMethodsCell>* node) {
        RelearnException::check(node != nullptr, "FastMultipoleMethods::update_functor: node is nullptr");

        // NOLINTNEXTLINE
        if (node->is_child()) {
            return;
        }

        using position_type = FastMultipoleMethodsCell::position_type;
        using counter_type = FastMultipoleMethodsCell::counter_type;

        // I'm inner node, i.e., I have a super neuron
        position_type my_position_dendrites_excitatory = { 0., 0., 0. };
        position_type my_position_dendrites_inhibitory = { 0., 0., 0. };

        position_type my_position_axons_excitatory = { 0., 0., 0. };
        position_type my_position_axons_inhibitory = { 0., 0., 0. };

        // Sum of number of dendrites of all my children
        counter_type my_number_dendrites_excitatory = 0;
        counter_type my_number_dendrites_inhibitory = 0;

        counter_type my_number_axons_excitatory = 0;
        counter_type my_number_axons_inhibitory = 0;

        // For all my children
        for (const auto& child : node->get_children()) {
            if (child == nullptr) {
                continue;
            }

            // Sum up number of dendrites
            const auto child_number_dendrites_excitatory = child->get_cell().get_number_excitatory_dendrites();
            const auto child_number_dendrites_inhibitory = child->get_cell().get_number_inhibitory_dendrites();

            const auto child_number_axons_excitatory = child->get_cell().get_number_excitatory_axons();
            const auto child_number_axons_inhibitory = child->get_cell().get_number_inhibitory_axons();

            my_number_dendrites_excitatory += child_number_dendrites_excitatory;
            my_number_dendrites_inhibitory += child_number_dendrites_inhibitory;

            my_number_axons_excitatory += child_number_axons_excitatory;
            my_number_axons_inhibitory += child_number_axons_inhibitory;

            // Average the position by using the number of dendrites as weights
            std::optional<position_type> child_position_dendrites_excitatory = child->get_cell().get_excitatory_dendrites_position();
            std::optional<position_type> child_position_dendrites_inhibitory = child->get_cell().get_inhibitory_dendrites_position();

            std::optional<position_type> child_position_axons_excitatory = child->get_cell().get_excitatory_axons_position();
            std::optional<position_type> child_position_axons_inhibitory = child->get_cell().get_inhibitory_axons_position();

            /**
             * We can use position if it's valid or if corresponding num of dendrites is 0
             */
            RelearnException::check(child_position_dendrites_excitatory.has_value() || (0 == child_number_dendrites_excitatory), "FastMultipoleMethods::update_functor: The child had excitatory dendrites, but no position. ID: {}", child->get_cell_neuron_id());
            RelearnException::check(child_position_dendrites_inhibitory.has_value() || (0 == child_number_dendrites_inhibitory), "FastMultipoleMethods::update_functor: The child had inhibitory dendrites, but no position. ID: {}", child->get_cell_neuron_id());

            RelearnException::check(child_position_axons_excitatory.has_value() || (0 == child_number_axons_excitatory), "FastMultipoleMethods::update_functor: The child had excitatory axons, but no position. ID: {}", child->get_cell_neuron_id());
            RelearnException::check(child_position_axons_inhibitory.has_value() || (0 == child_number_axons_inhibitory), "FastMultipoleMethods::update_functor: The child had inhibitory axons, but no position. ID: {}", child->get_cell_neuron_id());

            if (child_position_dendrites_excitatory.has_value()) {
                const auto scaled_position = child_position_dendrites_excitatory.value() * static_cast<double>(child_number_dendrites_excitatory);
                my_position_dendrites_excitatory += scaled_position;
            }

            if (child_position_dendrites_inhibitory.has_value()) {
                const auto scaled_position = child_position_dendrites_inhibitory.value() * static_cast<double>(child_number_dendrites_inhibitory);
                my_position_dendrites_inhibitory += scaled_position;
            }

            if (child_position_axons_excitatory.has_value()) {
                const auto scaled_position = child_position_axons_excitatory.value() * static_cast<double>(child_number_axons_excitatory);
                my_position_axons_excitatory += scaled_position;
            }

            if (child_position_axons_inhibitory.has_value()) {
                const auto scaled_position = child_position_axons_inhibitory.value() * static_cast<double>(child_number_axons_inhibitory);
                my_position_axons_inhibitory += scaled_position;
            }
        }

        node->set_cell_number_dendrites(my_number_dendrites_excitatory, my_number_dendrites_inhibitory);
        node->set_cell_number_axons(my_number_axons_excitatory, my_number_axons_inhibitory);

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
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons.
     * @param number_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so (== 0), the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @param axons The axon model that is used
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank.
     */
    CommunicationMap<SynapseCreationRequest> find_target_neurons(size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
        const std::unique_ptr<NeuronsExtraInfo>& extra_infos) override;

    /**
     * @brief Outputs the calculation results for the different calculation methods and also which would be the calculation method used by the simulation. Used for debugging.
     * @param source Node with vacant axons.
     * @param target Node with vacant dendrites.
     * @param needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     */
    void print_calculations(OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, SignalType needed);

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
    /**
     * @brief Appends pairs of neurons to a SynapseCreationRequest which are suitable for a synapse formation.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory)
     * @param request SynapseCreationRequest which should be extended. This must be created before the method is called.
     * @exception Can throw a RelearnException.
     */
    void make_creation_request_for(const SignalType signal_type_needed, CommunicationMap<SynapseCreationRequest>& request);

    /**
     * @brief Creates an initialized stack for the make_creation_request_for method. Source nodes and target nodes are paired based on their level in the octree.
     * It also depends on the level_offset specified in the config file.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @return Returns the initalised stack.
     */
    Stack<stack_entry> init_stack(const SignalType signal_type_needed);

    /**
     * @brief Takes the top node pair from the stack and unpacks them as many times as specified in the config file.
     * This serves to give the neurons more freedom of choice. After that, the resulting pairs are put back on the stack.
     * When unpacking == 0 the stack is not changed.
     * @param stack Stack on which node pairs are located and on which is worked on.
     */
    void unpack_node_pair(Stack<stack_entry>& stack);

    /**
     * @brief Aligns the level of source and target node and thereby creates the associated interaction list.
     * This ist due to the reason that only the target parent is pushed to the stack to reduce the size.
     * @param source_node Node with vacant axons and the desired level.
     * @param target_parent Node with vacant dendrite and smaller level.
     * @param signal_type  Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @return The corresponding interaction list suitable for one source node.
     */
    interaction_list_type align_interaction_list(OctreeNode<AdditionalCellAttributes>* source_node, OctreeNode<AdditionalCellAttributes>* target_parent, const SignalType signal_type);

    /**
     * @brief Creates a list of possible targets for a source node, which is a leaf,
     * such that the number of axons in source is at least as large as the number of all dendrites in the targets.
     * @param source Node with vacant axons. Must be a leaf node.
     * @param interaction_list List of all possible targets.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @return Returns selected targets, which were chosen according to probability and together have more dendrites than there are axons.
     */
    std::vector<OctreeNode<AdditionalCellAttributes>*> make_target_list(OctreeNode<AdditionalCellAttributes>* source_node, interaction_list_type interaction_list, const SignalType signal_type_needed);

    /**
     * @brief If a target is a leaf node but the source is not, a pair of a selected source child and the target must be pushed back on the stack.
     * How many pairs are made depends on how many dendrites the target has and how many axons the individual sources children have.
     *
     * @param target_node Node with vacant dendrites. Must be a leaf node.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @param stack Reference to the stack on which the pairs must be pushed back.
     * @param source_children Refernce on the children of the source node.
     */
    void make_stack_entries_for_leaf(OctreeNode<AdditionalCellAttributes>* target_node, const SignalType signal_type_needed, Stack<stack_entry>& stack, const std::array<OctreeNode<FastMultipoleMethods::AdditionalCellAttributes>*, 8UL>& source_children);

    /**
     * @brief Calculates the attraction between a single source neuron and all target neurons in the interaction list.
     * @param source Node with vacant axons.
     * @param interaction_list List of Nodes with vacant dendrites.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Can throw a RelearnException
     * @return Returns a vector with the calculated forces of attraction. This contains as many elements as the interaction list.
     */
    std::vector<double> calc_attractiveness_to_connect(OctreeNode<FastMultipoleMethodsCell>* source, const interaction_list_type& interaction_list, SignalType signal_type_needed);

    /**
     * @brief Checks which calculation type is suitable for a given source and target node.
     * @param source Node with vacant axons.
     * @param target Node with vacant dendrites.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory)
     * @return CalculationType
     */
    static CalculationType check_calculation_requirements(OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, SignalType signal_type_needed);

    /**
     * @brief Calculates the taylor coefficients for a pair of nodes. The calculation of coefficients and series
     * expansion is executed separately.
     *
     * @param source Node with vacant axons.
     * @param target_center Position of the target node.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @return Returns an array of the taylor coefficients.
     */
    static std::vector<double> calc_taylor_coefficients(const OctreeNode<FastMultipoleMethodsCell>* source, const position_type& target_center, const SignalType& signal_type_needed);

    /**
     * @brief Calculates the force of attraction between two nodes of the octree using a Taylor series expansion.
     * @param source Node with vacant axons.
     * @param target Node with vacant dendrites.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Can throw a RelearnException.
     * @return Returns the attraction force.
     */
    double calc_taylor(const OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, const SignalType signal_type_needed);
    /**
     * @brief Calculates the force of attraction between two sets of neurons by using the kernel
     * presented by Butz and van Oooyen.
     * @param sources Vector of pairs with 3D position and number of vacant axons.
     * @param targets Vector of pairs with 3D position and number of vacant dendrites.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @return Returns the total attraction of the neurons.
     */

    static double calc_direct_gauss(OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, SignalType signal_type_needed);

    /**
     * @brief Calculates the hermite coefficients for a source node. The calculation of coefficients and series
     * expansion is executed separately, since the coefficients can be reused.
     * @param source Node with vacant axons.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Can throw a RelearnException.
     * @returns Returns an array of the hermite coefficients.
     */
    static std::vector<double> calc_hermite_coefficients(const OctreeNode<FastMultipoleMethodsCell>* source, SignalType signal_type_needed);

    /**
     * @brief Calculates the force of attraction between two nodes of the octree using a Hermite series expansion.
     * @param source Node with vacant axons.
     * @param target Node with vacant dendrites.
     * @param coefficients_buffer Memory location where the coefficients are stored.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Can throw a RelearnException.
     * @return Retunrs the attraction force.
     */
    static double calc_hermite(const OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target,
        const std::vector<double>& coefficients_buffer, SignalType signal_type_needed);
};
