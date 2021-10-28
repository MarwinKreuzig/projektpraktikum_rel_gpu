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

#include "../../util/RelearnException.h"
#include "../SignalType.h"

#include <map>
#include <vector>

/**
 * An object of type SynapseCreationRequests stores the requests from the current MPI rank to a dedicated other MPI rank.
 * It stores all requests flattened and can manage the responses.
 * The class does not perform any communication or synchronization with other MPI ranks.
 */
class SynapseCreationRequests {
public:
    /**
     * @brief Creates an object with zero requests and responses.
     */
    SynapseCreationRequests() = default;

    /**
     * @brief Returns the number of stored requests and responses
     * @return The number of requests and responses
     */
    [[nodiscard]] size_t size() const noexcept {
        return num_requests;
    }

    /**
     * @brief Resizes the object so that it can hold the specified number of requests, allocates the necessary amount of memory
     * @param size The number of requests and responses to the stored
     */
    void resize(const size_t size) {
        num_requests = size;
        requests.resize(3 * size);
        responses.resize(size);
    }

    /**
     * @brief Appends a pending request, comprising of the source and target neuron ids and a flag that denotes the
     *      required dendrite type; 0 for excitatory and 1 for inhibitory
     * @param source_neuron_id The local neuron id of the requesting neuron
     * @param target_neuron_id The local (to the other rank) neuron id of the requested neuron
     * @param dendrite_type_needed The required type, coded with 0 for excitatory and 1 for inhibitory
     */
    void append(const size_t source_neuron_id, const size_t target_neuron_id, const size_t dendrite_type_needed) {
        num_requests++;

        requests.push_back(source_neuron_id);
        requests.push_back(target_neuron_id);
        requests.push_back(dendrite_type_needed);

        responses.resize(responses.size() + 1);
    }

    /**
     * @brief Appends a pending request, comprising of the source and target neuron ids and an enum that denotes the
     *      required dendrite type
     * @param source_neuron_id The local neuron id of the requesting neuron
     * @param target_neuron_id The local (to the other rank) neuron id of the requested neuron
     * @param dendrite_type_needed The required type as enum
     */
    void append(const size_t source_neuron_id, const size_t target_neuron_id, const SignalType dendrite_type_needed) {
        size_t dendrite_type_val = 0;

        if (dendrite_type_needed == SignalType::INHIBITORY) {
            dendrite_type_val = 1;
        }

        append(source_neuron_id, target_neuron_id, dendrite_type_val);
    }

    /**
     * @brief Returns the requested index as a three-tuple of the source' local neuron id, the targets local neuron id,
     *      and a enum that indicates whether it is an excitatory or inhibitory request
     * @param request_index The required request-index
     * @exception Throws a RelearnException if the request_index exceeds the stored number of requests
     * @return A tuple consisting of the local neuron id of source and target, and a enum that
     *       indicates whether it is an excitatory or inhibitory request
     */
    [[nodiscard]] std::tuple<size_t, size_t, SignalType> get_request(const size_t request_index) const {
        RelearnException::check(request_index < num_requests, "SynapseCreationRequests::get_request: index out of bounds: {} vs {}", request_index, num_requests);

        const size_t base_index = 3 * request_index;

        const size_t source_neuron_id = requests[base_index];
        const size_t target_neuron_id = requests[base_index + 1];
        const size_t dendrite_type_needed = requests[base_index + 2];

        const SignalType dendrite_type_needed_converted = (dendrite_type_needed == 0) ? SignalType::EXCITATORY : SignalType::INHIBITORY;

        return std::make_tuple(source_neuron_id, target_neuron_id, dendrite_type_needed_converted);
    }

    /**
     * @brief Sets the responce for the index-specified request
     * @param request_index The request index
     * @param connected A flag that specifies if the request is accepted (1) or denied (0)
     * @exception Throws a RelearnException if the request_index exceeds the stored number of responses
     */
    void set_response(const size_t request_index, const char connected) {
        RelearnException::check(request_index < num_requests, "SynapseCreationRequests::set_response: index out of bounds: {} vs {}", request_index, num_requests);

        responses[request_index] = connected;
    }

    /**
     * @brief Gets the responce for the index-specified request
     * @param request_index The request index
     * @exception Throws a RelearnException if the request_index exceeds the stored number of responses
     * @return A flag that specifies if the request is accepted (1) or denied (0)
     */
    [[nodiscard]] char get_response(const size_t request_index) const {
        RelearnException::check(request_index < num_requests, "SynapseCreationRequests::get_response: index out of bounds: {} vs {}", request_index, num_requests);
        return responses[request_index];
    }

    /**
     * @brief Gets a raw non-owning pointer for the encoded requests. The pointer is invalidated by append()
     * @return The pointer to the encoded requests
     */
    [[nodiscard]] size_t* get_requests() noexcept {
        return requests.data();
    }

    /**
     * @brief Gets a raw non-owning and non-mutable pointer for the encoded requests. The pointer is invalidated by append()
     * @return The pointer to the encoded requests
     */
    [[nodiscard]] const size_t* get_requests() const noexcept {
        return requests.data();
    }

    /**
     * @brief Gets a raw non-owning and non-mutable pointer for the stored responses. The pointer is invalidated by append()
     * @return The pointer to the encoded answers: (1) for true, (0) for false
     */
    [[nodiscard]] const char* get_responses() const noexcept {
        return responses.data();
    }

    /**
     * @brief Gets the number of bytes that all stored requests take. The size is invalidated by append()
     * @return The number of bytes all stored requests take
     */
    [[nodiscard]] size_t get_requests_size_in_bytes() const noexcept {
        return requests.size() * sizeof(size_t);
    }

    /**
     * @brief Gets the number of bytes all stored responses take. The size is invalidated by append()
     * @return The number of bytes all stored responses take
     */
    [[nodiscard]] size_t get_responses_size_in_bytes() const noexcept {
        return responses.size() * sizeof(char);
    }

private:
    size_t num_requests{ 0 }; // Number of synapse creation requests
    std::vector<size_t> requests{}; // Each request to form a synapse is a 3-tuple: (source_neuron_id, target_neuron_id, dendrite_type_needed)
                                    // That is why requests.size() == 3*responses.size()
                                    // Note, a more memory-efficient implementation would use a smaller data type (not size_t) for dendrite_type_needed.
                                    // This vector is used as MPI communication buffer
    std::vector<char> responses{}; // Response if the corresponding request was accepted and thus the synapse was formed
                                   // responses[i] refers to requests[3*i,...,3*i+2]
                                   // This vector is used as MPI communication buffer
};

/**
 * Map of (MPI rank; SynapseCreationRequests)
 * The MPI rank specifies the corresponding process
 */
using MapSynapseCreationRequests = std::map<int, SynapseCreationRequests>;
