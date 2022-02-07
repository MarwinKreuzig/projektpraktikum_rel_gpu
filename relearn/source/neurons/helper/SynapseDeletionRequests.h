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

#include "../../Config.h"
#include "../../util/RelearnException.h"
#include "../ElementType.h"
#include "../SignalType.h"

#include <utility>

/**
 * This type represents a synapse deletion request.
 * It is exchanged via MPI ranks but does not perform MPI communication on its own.
 * A synapse is represented as 
 *      (initiator_neuron_id, initiator_element_type, signal_type) <---> (affected_neuron_id, !initiator_element_type, signal_type)
 */
class SynapseDeletionRequest {
    RelearnTypes::neuron_id initiator_neuron_id{};
    RelearnTypes::neuron_id affected_neuron_id{};
    ElementType initiator_element_type{ ElementType::AXON };
    SignalType signal_type{ SignalType::EXCITATORY };

public:
    /**
     * Creates a new object 
     */
    SynapseDeletionRequest() = default;

    /**
     * @brief Creates a new deletion request with the passed arguments
     * @param initiator_neuron The neuron that initiaed the request
     * @param affected_neuron The neuron that is affected by the request (the other end of the synapse)
     * @param element_type The element type that initiated the request
     * @param signal_type The signal type of the synapse
     * @exception Throws a RelearnException if any neuron id is invalid
     */
    SynapseDeletionRequest(RelearnTypes::neuron_id initiator_neuron, RelearnTypes::neuron_id affected_neuron, const ElementType element_type, const SignalType signal_type)
        : initiator_neuron_id(initiator_neuron)
        , affected_neuron_id(affected_neuron)
        , initiator_element_type(element_type)
        , signal_type(signal_type) {
        RelearnException::check(initiator_neuron < Constants::uninitialized, "SynapseDeletionRequest::SynapseDeletionRequest(): initiator_neuron neuron id was too large");
        RelearnException::check(affected_neuron < Constants::uninitialized, "SynapseDeletionRequest::SynapseDeletionRequest(): affected_neuron neuron id was too large");
    }

    SynapseDeletionRequest(const SynapseDeletionRequest& other) = default;
    SynapseDeletionRequest(SynapseDeletionRequest&& other) = default;

    SynapseDeletionRequest& operator=(const SynapseDeletionRequest& other) = default;
    SynapseDeletionRequest& operator=(SynapseDeletionRequest&& other) = default;

    ~SynapseDeletionRequest() = default;

    /**
     * @brief Returns the initiator neuron id, i.e., the neuron which started the deletion
     * @return The source neuron id
     */
    [[nodiscard]] RelearnTypes::neuron_id get_initiator_neuron_id() const noexcept {
        return initiator_neuron_id;
    }

    /**
     * @brief Returns the affected neuron id, i.e., the neuron that must be notified
     * @return The target neuron id
     */
    [[nodiscard]] RelearnTypes::neuron_id get_affected_neuron_id() const noexcept {
        return affected_neuron_id;
    }

    /**
     * @brief Returns the initiator element type
     * @return The affected element type
     */
    [[nodiscard]] ElementType get_initiator_element_type() const noexcept {
        return initiator_element_type;
    }

    /**
     * @brief Returns the synapses signal type
     * @return The signal type
     */
    [[nodiscard]] SignalType get_signal_type() const noexcept {
        return signal_type;
    }

    template <std::size_t Index>
    auto& get() & {
        if constexpr (Index == 0)
            return initiator_neuron_id;
        if constexpr (Index == 1)
            return affected_neuron_id;
        if constexpr (Index == 2)
            return initiator_element_type;
        if constexpr (Index == 3)
            return signal_type;
    }

    template <std::size_t Index>
    auto const& get() const& {
        if constexpr (Index == 0)
            return initiator_neuron_id;
        if constexpr (Index == 1)
            return affected_neuron_id;
        if constexpr (Index == 2)
            return initiator_element_type;
        if constexpr (Index == 3)
            return signal_type;
    }

    template <std::size_t Index>
    auto&& get() && {
        if constexpr (Index == 0)
            return std::move(initiator_neuron_id);
        if constexpr (Index == 1)
            return std::move(affected_neuron_id);
        if constexpr (Index == 2)
            return std::move(initiator_element_type);
        if constexpr (Index == 3)
            return std::move(signal_type);
    }
};

namespace std {
template <>
struct tuple_size<typename ::SynapseDeletionRequest> {
    static constexpr size_t value = 4;
};

template <>
struct tuple_element<0, typename ::SynapseDeletionRequest> {
    using type = RelearnTypes::neuron_id;
};

template <>
struct tuple_element<1, typename ::SynapseDeletionRequest> {
    using type = RelearnTypes::neuron_id;
};

template <>
struct tuple_element<2, typename ::SynapseDeletionRequest> {
    using type = ElementType;
};

template <>
struct tuple_element<3, typename ::SynapseDeletionRequest> {
    using type = SignalType;
};

} //namespace std
