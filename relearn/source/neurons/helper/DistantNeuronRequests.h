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

#include <utility>

//template <typename AdditionalCellAttributes>
//struct NeuronRequestInformation {
//    RankNeuronId source_rank_neuron_id;
//    RankNeuronId target_rank_neuron_id;
//    RelearnTypes::position_type source_neuron_position;
//    OctreeNode<AdditionalCellAttributes>* target_node;
//    ElementType element_type;
//    SignalType signal_type;
//};
//
//struct NeuronResponseInformation {
//    SynapseCreationResponse response;
//    NeuronID target_id;
//};

/**
 * One DistantNeuronRequest always consists of a target neuron, a source neuron, the position of the source, the target node and a signal type
 */

template <typename AdditionalCellAttributes>
class DistantNeuronRequest {
    NeuronID target_id{};
    NeuronID source_id{};
    RelearnTypes::position_type source_position{};
    OctreeNode<AdditionalCellAttributes>* target_node{};
    SignalType signal_type{};

public:
    DistantNeuronRequest() = default;

    /**
     * @brief Constructs a new request with the arguments
     * @param target_id The RankNeuronId of the target 
     * @param source_id The RankNeuronId of the source 
     * @param source_position The position of the source
     * @param target_node The targeted node of the request
     * @param signal_type The signal type
     */
    DistantNeuronRequest(NeuronID target_id, NeuronID source_id, RelearnTypes::position_type source_position, OctreeNode<AdditionalCellAttributes>* target_node, SignalType signal_type)
        : target_id(target_id)
        , source_id(source_id)
        , source_position(source_position)
        , target_node(target_node)
        , signal_type(signal_type) {
        RelearnException::check(target_node != nullptr, "DistantNeuronRequest::DistantNeuronRequest: target_node was nullptr");
    }

    /**
     * @brief Returns the target of the request
     * @return The target
     */
    [[nodiscard]] NeuronID get_target_id() const noexcept {
        return target_id;
    }

    /**
     * @brief Returns the source of the request
     * @return The source
     */
    [[nodiscard]] NeuronID get_source_id() const noexcept {
        return source_id;
    }

    /**
     * @brief Returns the position of the source of the request
     * @return The source position
     */
    [[nodiscard]] RelearnTypes::position_type get_source_position() const noexcept {
        return source_position;
    }

    /**
     * @brief Returns the targeted node of the request
     * @return The target node
     */
    [[nodiscard]] OctreeNode<AdditionalCellAttributes>* get_target_node() const noexcept {
        return target_node;
    }

    /**
     * @brief Returns the signal type of the request
     * @return The signal type
     */
    [[nodiscard]] SignalType get_signal_type() const noexcept {
        return signal_type;
    }
};

/**
 * The response for a DistantNeuronRequest consists of the source of the response and a SynapseCreationResponse
 */

class DistantNeuronResponse {
    NeuronID source_id{};
    SynapseCreationResponse creation_response{};

public:
    DistantNeuronResponse() = default;

    /**
     * @brief Constructs a new response with the arguments
     * @param source The RankNeuronId of the source
     * @param creation_response The response if a synapse was succesfully created
     */
    DistantNeuronResponse(NeuronID source_id, SynapseCreationResponse creation_response)
        : source_id(source_id)
        , creation_response(creation_response) {

    };

    /**
     * @brief Returns the source of the response
     * @return The source
     */
    [[nodiscard]] NeuronID get_source_id() const noexcept {
        return source_id;
    }

    /**
     * @brief Returns the creation response
     * @return The creation response
     */
    [[nodiscard]] SynapseCreationResponse get_creation_response() const noexcept {
        return creation_response;
    }
};
