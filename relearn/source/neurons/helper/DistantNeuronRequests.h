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

#include "Types.h"
#include "neurons/ElementType.h"
#include "neurons/SignalType.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "util/TaggedID.h"

#include <cstdint>
#include <utility>

/**
 * One DistantNeuronRequest always consists of a source neuron and its position, which initiates the request,
 * the signal type which it looks for, and an identifier for the target neuron.
 * This identifier changes based on the height of the target neuron in the octree.
 * There are 3 distinct cases (with descending priority):
 * (a) The target is on the branch node level
 * (b) The target is a leaf node
 * (c) The target is a virtual node
 */
class DistantNeuronRequest {
public:
    enum class TargetNeuronType : char {
        BranchNode = 0,
        Leaf = 1,
        VirtualNode = 2
    };

    constexpr DistantNeuronRequest() = default;

    /**
     * @brief Constructs a new request with the arguments. A request can be built for three different use cases:
     *      (a) The target node is a branch node -- target_neuron_type should be TargetNeuronType::BranchNode and target_neuron_identifier the index of it when considering all branch nodes
     *      (b) The target node is a leaf node -- target_neuron_type should be TargetNeuronType::Leaf and target_neuron_identifier the index of it in the local neurons
     *      (c) The target node is a virtual node -- target_neuron_type should be TargetNeuronType::VirtualNode and target_neuron_identifier the RMA offset
     * @param source_id The RankNeuronId of the source
     * @param source_position The position of the source
     * @param target_neuron_identifier The identifier of the target node
     * @param target_neuron_type The type of the target node
     * @param signal_type The signal type
     */
    constexpr DistantNeuronRequest(const NeuronID& source_id, const RelearnTypes::position_type& source_position,
        const NeuronID::value_type target_neuron_identifier, const TargetNeuronType target_neuron_type, const SignalType signal_type) noexcept
        : source_id(source_id)
        , source_position(source_position)
        , target_neuron_identifier(target_neuron_identifier)
        , target_neuron_type(target_neuron_type)
        , signal_type(signal_type) { }

    /**
     * @brief Returns the source of the request
     * @return The source
     */
    [[nodiscard]] constexpr const NeuronID& get_source_id() const noexcept {
        return source_id;
    }

    /**
     * @brief Returns the position of the source of the request
     * @return The source position
     */
    [[nodiscard]] constexpr RelearnTypes::position_type get_source_position() const noexcept {
        return source_position;
    }

    /**
     * @brief Returns the id of the target node, if it is a branch node.
     * @exception Throws a RelearnException if the target node type is not TargetNeuronType::BranchNode
     * @return The branch node id
     */
    [[nodiscard]] constexpr NeuronID::value_type get_branch_node_id() const {
        RelearnException::check(target_neuron_type == TargetNeuronType::BranchNode, "");
        return target_neuron_identifier;
    }

    /**
     * @brief Returns the id of the target node, if it is a leaf node.
     * @exception Throws a RelearnException if the target node type is not TargetNeuronType::Leaf
     * @return The leaf node id
     */
    [[nodiscard]] constexpr NeuronID::value_type get_leaf_node_id() const {
        RelearnException::check(target_neuron_type == TargetNeuronType::Leaf, "");
        return target_neuron_identifier;
    }

    /**
     * @brief Returns the RMA offset of the target node, if it is a virtual node.
     * @exception Throws a RelearnException if the target node type is not TargetNeuronType::VirtualNode
     * @return The RMA offset
     */
    [[nodiscard]] constexpr NeuronID::value_type get_rma_offset() const {
        RelearnException::check(target_neuron_type == TargetNeuronType::VirtualNode, "");
        return target_neuron_identifier;
    }

    /**
     * @brief Returns the type of target neuron
     * @return The type of the target neuron
     */
    [[nodiscard]] constexpr TargetNeuronType get_target_neuron_type() const noexcept {
        return target_neuron_type;
    }

    /**
     * @brief Returns the signal type of the request
     * @return The signal type
     */
    [[nodiscard]] constexpr SignalType get_signal_type() const noexcept {
        return signal_type;
    }

private:
    NeuronID source_id{};
    RelearnTypes::position_type source_position{};
    NeuronID::value_type target_neuron_identifier{};
    TargetNeuronType target_neuron_type{};
    SignalType signal_type{};
};

/**
 * The response for a DistantNeuronRequest consists of the source of the response and a SynapseCreationResponse
 */
class DistantNeuronResponse {
public:
    constexpr DistantNeuronResponse() = default;

    /**
     * @brief Constructs a new response with the arguments
     * @param source The RankNeuronId of the source
     * @param creation_response The response if a synapse was succesfully created
     */
    constexpr DistantNeuronResponse(const NeuronID& source_id, const SynapseCreationResponse creation_response)
        : source_id(source_id)
        , creation_response(creation_response) { }

    /**
     * @brief Returns the source of the response
     * @return The source
     */
    [[nodiscard]] constexpr const NeuronID& get_source_id() const noexcept {
        return source_id;
    }

    /**
     * @brief Returns the creation response
     * @return The creation response
     */
    [[nodiscard]] constexpr SynapseCreationResponse get_creation_response() const noexcept {
        return creation_response;
    }

private:
    NeuronID source_id{};
    SynapseCreationResponse creation_response{};
};
