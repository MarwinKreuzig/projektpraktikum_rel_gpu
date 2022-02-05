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
#include "../../mpi/MPIWrapper.h"
#include "../../util/RelearnException.h"
#include "../ElementType.h"
#include "../SignalType.h"

#include <map>
#include <utility>
#include <vector>

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

/**
 * This type aggregates multiple SynapseDeletionRequest into one and facilitates MPI communication.
 * It does not perform MPI communication.
 */
class SynapseDeletionRequests {
public:
    SynapseDeletionRequests() = default;

    /**
     * @brief Returns the number of stored requests
     * @return The number of stored requests
     */
    [[nodiscard]] size_t size() const noexcept {
        return requests.size();
    }

    /**
     * @brief Resizes the internal buffer to accomodate size-many requests
     * @param size The number of requests to be stored
     */
    void resize(const size_t size) {
        requests.resize(size);
    }

    /**
     * @brief Appends the SynapseDeletionRequest to the end of the buffer
     * @param pending_deletion The new SynapseDeletionRequest that should be appended
     */
    void append(const SynapseDeletionRequest& pending_deletion) {
        requests.push_back(pending_deletion);
    }

    /**
     * @brief Returns the SynapseDeletionRequest with the requested index
     * @param request_index The index of the SynapseDeletionRequest
     * @exception Throws a RelearnException if request_index is larger than the number of stored SynapseDeletionRequest
     * @return The deletion reques
     */
    [[nodiscard]] SynapseDeletionRequest get_request(const size_t request_index) const {
        RelearnException::check(request_index < requests.size(), "SynapseDeletionRequests::get_request: Index is out of bounds");
        return requests[request_index];
    }

    /**
     * @brief Returns the raw pointer to the requests.
     *      Does not transfer ownership
     * @return A raw pointer to the requests
     */
    [[nodiscard]] SynapseDeletionRequest* get_requests() noexcept {
        return requests.data();
    }

    /**
     * @brief Returns the raw pointer to the requests.
     *      Does not transfer ownership
     * @return A raw pointer to the requests
     */
    [[nodiscard]] const SynapseDeletionRequest* get_requests() const noexcept {
        return requests.data();
    }

    /**
     * @brief Returns the size of the internal buffer in bytes
     * @return The size of the internal buffer in bytes
     */
    [[nodiscard]] size_t get_requests_size_in_bytes() const noexcept {
        return requests.size() * sizeof(SynapseDeletionRequest);
    }

private:
    std::vector<SynapseDeletionRequest> requests{}; // This vector is used as MPI communication buffer

public:
    static std::map<int, SynapseDeletionRequests> exchange_requests(const std::map<int, SynapseDeletionRequests>& synapse_deletion_requests_outgoing) {
        const auto number_ranks = MPIWrapper::get_num_ranks();

        std::vector<size_t> number_requests_for_ranks(number_ranks, 0);
        // Fill vector with my number of synapse deletion requests for every rank
        for (const auto& [rank, requests] : synapse_deletion_requests_outgoing) {
            RelearnException::check(rank < number_ranks, "SynapseDeletionRequests::exchange_requests: rank was too large: {} of {}", rank, number_ranks);
            const auto num_requests = requests.size();
            number_requests_for_ranks[rank] = num_requests;
        }

        std::vector<size_t> number_requests_from_ranks(number_ranks, 0);
        MPIWrapper::all_to_all(number_requests_for_ranks, number_requests_from_ranks);

        // Now I know how many requests I will get from every rank.
        std::map<int, SynapseDeletionRequests> incoming_requests{};
        for (auto rank = 0; rank < number_ranks; ++rank) {
            if (const auto num_requests = number_requests_from_ranks[rank]; 0 != num_requests) {
                incoming_requests[rank].resize(num_requests);
            }
        }

        std::vector<MPIWrapper::AsyncToken> mpi_requests(synapse_deletion_requests_outgoing.size() + incoming_requests.size());

        /**
	     * Send and receive actual synapse deletion requests
	     */
        auto mpi_requests_index = 0;

        // Receive actual synapse deletion requests
        for (auto& [rank, requests] : incoming_requests) {
            auto* buffer = requests.get_requests();
            const auto size_in_bytes = static_cast<int>(requests.get_requests_size_in_bytes());

            MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

            ++mpi_requests_index;
        }

        // Send actual synapse deletion requests
        for (const auto& [rank, requests] : synapse_deletion_requests_outgoing) {
            const auto* const buffer = requests.get_requests();
            const auto size_in_bytes = static_cast<int>(requests.get_requests_size_in_bytes());

            MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);
            ++mpi_requests_index;
        }

        // Wait for all sends and receives to complete
        MPIWrapper::wait_all_tokens(mpi_requests);

        return incoming_requests;
    }
};

/**
 * Map of (MPI rank; SynapseDeletionRequests)
 * The MPI rank specifies the corresponding process
 */
using MapSynapseDeletionRequests = std::map<int, SynapseDeletionRequests>;
