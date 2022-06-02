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
#include "neurons/SignalType.h"
#include "util/TaggedID.h"

#include <utility>

/**
 * One SynapseCreationRequest always consists of a target neuron, a source neuron, and a signal type
 */

class SynapseCreationRequest {
    NeuronID target{};
    NeuronID source{};
    SignalType signal_type{};

public:
    SynapseCreationRequest() = default;

    /**
     * @brief Constructs a new reqest with the arguments
     * @param target The neuron target id of the request
     * @param source The neuron source id of the request
     * @param signal_type The signal type
     */
    SynapseCreationRequest(NeuronID target, NeuronID source, SignalType signal_type)
        : target(target)
        , source(source)
        , signal_type(signal_type) {
        RelearnException::check(target.is_local(), "SynapseCreationRequest::SynapseCreationRequest: Can only serve local ids (target): {}", target);
        RelearnException::check(source.is_local(), "SynapseCreationRequest::SynapseCreationRequest: Can only serve local ids (source): {}", source);
    }

    /**
     * @brief Returns the target of the request
     * @return The target
     */
    [[nodiscard]] NeuronID get_target() const noexcept {
        return target;
    }

    /**
     * @brief Returns the source of the request
     * @return The source
     */
    [[nodiscard]] NeuronID get_source() const noexcept {
        return source;
    }

    /**
     * @brief Returns the neuron type of the request
     * @return The neuron type
     */
    [[nodiscard]] SignalType get_signal_type() const noexcept {
        return signal_type;
    }

    template <std::size_t Index>
    [[nodiscard]] auto& get() & {
        if constexpr (Index == 0) {
            return target;
        }
        if constexpr (Index == 1) {
            return source;
        }
        if constexpr (Index == 2) {
            return signal_type;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] auto const& get() const& {
        if constexpr (Index == 0) {
            return target;
        }
        if constexpr (Index == 1) {
            return source;
        }
        if constexpr (Index == 2) {
            return signal_type;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] auto&& get() && {
        if constexpr (Index == 0) {
            return target;
        }
        if constexpr (Index == 1) {
            return source;
        }
        if constexpr (Index == 2) {
            return signal_type;
        }
    }
};

namespace std {
template <>
struct tuple_size<typename ::SynapseCreationRequest> {
    static constexpr size_t value = 3;
};

template <>
struct tuple_element<0, typename ::SynapseCreationRequest> {
    using type = NeuronID;
};

template <>
struct tuple_element<1, typename ::SynapseCreationRequest> {
    using type = NeuronID;
};

template <>
struct tuple_element<2, typename ::SynapseCreationRequest> {
    using type = SignalType;
};

} //namespace std

/**
 * The response for a SynapseCreationRequest can be that it failed or succeeded
 */
enum class SynapseCreationResponse : char {
    Failed = 0,
    Succeeded = 1,
};
