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

#include "../../Types.h"
#include "../../mpi/MPIWrapper.h"
#include "../../util/RelearnException.h"
#include "../SignalType.h"

#include <map>
#include <utility>
#include <vector>

/**
 * An object of type SynapseCreationRequests stores the requests from the current MPI rank to a dedicated other MPI rank.
 * It stores all requests flattened and can manage the responses. 
 */
class SynapseCreationRequests {
public:
    /**
     * One Request always consists of a target neuron, a source neuron, and a signal type
     */
    class Request {
        RelearnTypes::neuron_id target{};
        RelearnTypes::neuron_id source{};
        SignalType signal_type{};

    public:
        Request() = default;

        /**
         * @brief Constructs a new reqest with the arguments
         * @param target The neuron target id of the request
         * @param source The neuron source id of the request
         * @param signal_type The signal type
         */
        Request(RelearnTypes::neuron_id target, RelearnTypes::neuron_id source, SignalType signal_type)
            : target(target)
            , source(source)
            , signal_type(signal_type) { }

        /**
         * @brief Returns the target of the request
         * @return The target
         */
        [[nodiscard]] RelearnTypes::neuron_id get_target() const noexcept {
            return target;
        }

        /**
         * @brief Returns the source of the request
         * @return The source
         */
        [[nodiscard]] RelearnTypes::neuron_id get_source() const noexcept {
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
        auto& get() & {
            if constexpr (Index == 0)
                return target;
            if constexpr (Index == 1)
                return source;
            if constexpr (Index == 2)
                return signal_type;
        }

        template <std::size_t Index>
        auto const& get() const& {
            if constexpr (Index == 0)
                return target;
            if constexpr (Index == 1)
                return source;
            if constexpr (Index == 2)
                return signal_type;
        }

        template <std::size_t Index>
        auto&& get() && {
            if constexpr (Index == 0)
                return std::move(target);
            if constexpr (Index == 1)
                return std::move(source);
            if constexpr (Index == 2)
                return std::move(signal_type);
        }
    };

    /**
     * The response for a request can be that the request failed or succeeded
     */
    enum class Response : char {
        failed = 0,
        succeeded = 1,
    };

    /**
     * @brief Creates an object with zero requests and responses.
     */
    SynapseCreationRequests() = default;

    /**
     * @brief Returns the number of stored requests and responses
     * @return The number of requests and responses
     */
    [[nodiscard]] size_t size() const {
        RelearnException::check(requests.size() == responses.size(), "SynapseCreationRequests::size: requests ({}) and responses ({}) had different sizes", requests.size(), responses.size());
        return requests.size();
    }

    /**
     * @brief Resizes the object so that it can hold the specified number of requests, allocates the necessary amount of memory
     * @param size The number of requests and responses to the stored
     */
    void resize(const size_t size) {
        requests.resize(size);
        responses.resize(size);
    }

    /**
     * @brief Appends a pending request
     */
    void append(const Request& request) noexcept {
        requests.emplace_back(request);
        responses.resize(responses.size() + 1);
    }

    /**
     * @brief Returns the requested index as a three-tuple of the source' local neuron id, the targets local neuron id,
     *      and a enum that indicates whether it is an excitatory or inhibitory request
     * @param request_index The required request-index 
     * @exception Throws a RelearnException if the request_index exceeds the stored number of requests
     * @return A tuple consisting of the local neuron id of source and target, and a enum that
     *       indicates whether it is an excitatory or inhibitory request
     */
    [[nodiscard]] Request get_request(const size_t request_index) const {
        RelearnException::check(request_index < requests.size(), "SynapseCreationRequests::get_request: index out of bounds: {} vs {}", request_index, requests.size());
        return requests[request_index];
    }

    /**
     * @brief Sets the responce for the index-specified request
     * @param request_index The request index 
     * @param connected A flag that specifies if the request is accepted (1) or denied (0)
     * @exception Throws a RelearnException if the request_index exceeds the stored number of responses
     */
    void set_response(const size_t request_index, const Response connected) {
        RelearnException::check(request_index < responses.size(), "SynapseCreationRequests::set_response: index out of bounds: {} vs {}", request_index, responses.size());
        responses[request_index] = connected;
    }

    /**
     * @brief Gets the responce for the index-specified request
     * @param request_index The request index 
     * @exception Throws a RelearnException if the request_index exceeds the stored number of responses
     * @return A flag that specifies if the request is accepted (1) or denied (0)
     */
    [[nodiscard]] Response get_response(const size_t request_index) const {
        RelearnException::check(request_index < responses.size(), "SynapseCreationRequests::get_response: index out of bounds: {} vs {}", request_index, responses.size());
        return responses[request_index];
    }

    /**
     * @brief Gets a raw non-owning pointer for the encoded requests. The pointer is invalidated by append()
     * @return The pointer to the encoded requests
     */
    [[nodiscard]] Request* data() noexcept {
        return requests.data();
    }

    /**
     * @brief Gets a raw non-owning and non-mutable pointer for the encoded requests. The pointer is invalidated by append()
     * @return The pointer to the encoded requests
     */
    [[nodiscard]] const Request* data() const noexcept {
        return requests.data();
    }

    /**
     * @brief Gets a raw non-owning pointer for the stored responses. The pointer is invalidated by append()
     * @return The pointer to the encoded answers: (1) for true, (0) for false
     */
    [[nodiscard]] Response* get_responses() noexcept {
        return responses.data();
    }

    /**
     * @brief Gets a raw non-owning and non-mutable pointer for the stored responses. The pointer is invalidated by append()
     * @return The pointer to the encoded answers: (1) for true, (0) for false
     */
    [[nodiscard]] const Response* get_responses() const noexcept {
        return responses.data();
    }

    /**
     * @brief Gets the number of bytes that all stored requests take. The size is invalidated by append()
     * @return The number of bytes all stored requests take
     */
    [[nodiscard]] size_t get_requests_size_in_bytes() const noexcept {
        return requests.size() * sizeof(Request);
    }

    /**
     * @brief Gets the number of bytes all stored responses take. The size is invalidated by append()
     * @return The number of bytes all stored responses take
     */
    [[nodiscard]] size_t get_responses_size_in_bytes() const noexcept {
        return responses.size() * sizeof(char);
    }

private:
    std::vector<Request> requests{}; // This vector is used as MPI communication buffer
    std::vector<Response> responses{}; // This vector is used as MPI communication buffer

public:
    /**
     * @brief Exchanges all requests across all MPI ranks, that is,
     *      if MPI rank i had the requests r for MPI rank j,
     *      i calls this function with outgoing_requests[j] = r,
     *      then after it returns j has <return>[i] = r
     * @param outgoing_requests The outgoing synapse creation requests for other ranks
     * @return The incoming synapse creation requests from other ranks
    */
    [[nodiscard]] static std::map<int, SynapseCreationRequests> exchange_requests(const std::map<int, SynapseCreationRequests>& outgoing_requests) {
        const auto number_ranks = MPIWrapper::get_num_ranks();

        /**
	     * Send to every rank the number of requests it should prepare for from me.
	     * Likewise, receive the number of requests that I should prepare for from every rank.
	     */
        std::vector<size_t> number_requests_for_ranks(number_ranks, 0);
        // Fill vector with my number of synapse requests for every rank (including me)
        for (const auto& [rank, requests] : outgoing_requests) {
            RelearnException::check(rank < number_ranks, "SynapseCreationRequests::exchange_requests: rank was too large: {} of {}", rank, number_ranks);
            const auto num_requests = requests.size();
            number_requests_for_ranks[rank] = num_requests;
        }

        std::vector<size_t> number_requests_from_ranks(number_ranks, 0);
        MPIWrapper::all_to_all(number_requests_for_ranks, number_requests_from_ranks);

        // Now I know how many requests I will get from every rank.
        std::map<int, SynapseCreationRequests> incoming_requests{};
        for (auto rank = 0; rank < number_ranks; rank++) {
            if (const auto num_requests = number_requests_from_ranks[rank]; 0 != num_requests) {
                incoming_requests[rank].resize(num_requests);
            }
        }

        std::vector<MPIWrapper::AsyncToken> mpi_requests(outgoing_requests.size() + incoming_requests.size());

        /**
	     * Send and receive actual synapse requests
	     */
        auto mpi_requests_index = 0;

        // Receive actual synapse requests
        for (auto& [rank, requests] : incoming_requests) {
            auto* buffer = requests.data();
            const auto size_in_bytes = static_cast<int>(requests.get_requests_size_in_bytes());

            MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

            mpi_requests_index++;
        }

        // Send actual synapse requests
        for (const auto& [rank, requests] : outgoing_requests) {
            const auto* const buffer = requests.data();
            const auto size_in_bytes = static_cast<int>(requests.get_requests_size_in_bytes());

            MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

            mpi_requests_index++;
        }

        // Wait for all sends and receives to complete
        MPIWrapper::wait_all_tokens(mpi_requests);

        return incoming_requests;
    }

    /**
     * @brief Exchanges all responses across all MPI ranks, that is,
     *      if MPI rank i had the response r for MPI rank j,
     *      i calls this function with responses_from_me[j] = r,
     *      then after it returns j has responses_for_me[i] = r 
     * @param responses_from_me My local responses to previous creation requests
     * @param responses_for_me The responses for me. Must be the same that was used in exchange_requests beforehand
     */
    static void exchange_responses(const std::map<int, SynapseCreationRequests>& responses_from_me, std::map<int, SynapseCreationRequests>& responses_for_me) {
        auto mpi_requests_index = 0;
        std::vector<MPIWrapper::AsyncToken> mpi_requests(responses_for_me.size() + responses_from_me.size());

        for (auto& [rank, requests] : responses_for_me) {
            auto* buffer = requests.get_responses();
            const auto size_in_bytes = static_cast<int>(requests.size());

            MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

            mpi_requests_index++;
        }

        for (const auto& [rank, requests] : responses_from_me) {
            const auto* const buffer = requests.get_responses();
            const auto size_in_bytes = static_cast<int>(requests.size());

            MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

            mpi_requests_index++;
        }

        // Wait for all sends and receives to complete
        MPIWrapper::wait_all_tokens(mpi_requests);
    }
};

/**
 * Map of (MPI rank; SynapseCreationRequests)
 * The MPI rank specifies the corresponding process
 */
using MapSynapseCreationRequests = std::map<int, SynapseCreationRequests>;

namespace std {
template <>
struct tuple_size<typename ::SynapseCreationRequests::Request> {
    static constexpr size_t value = 3;
};

template <>
struct tuple_element<0, typename ::SynapseCreationRequests::Request> {
    using type = RelearnTypes::neuron_id;
};

template <>
struct tuple_element<1, typename ::SynapseCreationRequests::Request> {
    using type = RelearnTypes::neuron_id;
};

template <>
struct tuple_element<2, typename ::SynapseCreationRequests::Request> {
    using type = SignalType;
};

} //namespace std
