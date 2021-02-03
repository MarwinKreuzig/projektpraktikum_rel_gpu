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

#include "Cell.h"
#include "Config.h"

#include <map>
#include <vector>

/**
* Type for synapse creation requests which are used with MPI
*/
class SynapseCreationRequests {
public:
    SynapseCreationRequests() = default;

    [[nodiscard]] size_t size() const noexcept {
        return num_requests;
    }

    void resize(size_t size) {
        num_requests = size;
        requests.resize(3 * size);
        responses.resize(size);
    }

    void append(size_t source_neuron_id, size_t target_neuron_id, size_t dendrite_type_needed) {
        num_requests++;

        requests.push_back(source_neuron_id);
        requests.push_back(target_neuron_id);
        requests.push_back(dendrite_type_needed);

        responses.resize(responses.size() + 1);
    }

    void append(size_t source_neuron_id, size_t target_neuron_id, SignalType dendrite_type_needed) {
        size_t dendrite_type_val = 0;

        if (dendrite_type_needed == SignalType::INHIBITORY) {
            dendrite_type_val = 1;
        }

        append(source_neuron_id, target_neuron_id, dendrite_type_val);
    }

    [[nodiscard]] std::tuple<size_t, size_t, size_t> get_request(size_t request_index) const noexcept {
        const size_t base_index = 3 * request_index;

        const size_t source_neuron_id = requests[base_index];
        const size_t target_neuron_id = requests[base_index + 1];
        const size_t dendrite_type_needed = requests[base_index + 2];

        return std::make_tuple(source_neuron_id, target_neuron_id, dendrite_type_needed);
    }

    void set_response(size_t request_index, char connected) noexcept {
        responses[request_index] = connected;
    }

    [[nodiscard]] char get_response(size_t request_index) const noexcept {
        return responses[request_index];
    }

    [[nodiscard]] size_t* get_requests() noexcept {
        return requests.data();
    }

    [[nodiscard]] const size_t* get_requests() const noexcept {
        return requests.data();
    }

    [[nodiscard]] char* get_responses() noexcept {
        return responses.data();
    }

    [[nodiscard]] const char* get_responses() const noexcept {
        return responses.data();
    }

    [[nodiscard]] size_t get_requests_size_in_bytes() const noexcept {
        return requests.size() * sizeof(size_t);
    }

    [[nodiscard]] size_t get_responses_size_in_bytes() const noexcept {
        return responses.size() * sizeof(char);
    }

private:
    size_t num_requests{ 0 }; // Number of synapse creation requests
    std::vector<size_t> requests; // Each request to form a synapse is a 3-tuple: (source_neuron_id, target_neuron_id, dendrite_type_needed)
        // That is why requests.size() == 3*responses.size()
        // Note, a more memory-efficient implementation would use a smaller data type (not size_t) for dendrite_type_needed.
        // This vector is used as MPI communication buffer
    std::vector<char> responses; // Response if the corresponding request was accepted and thus the synapse was formed
        // responses[i] refers to requests[3*i,...,3*i+2]
        // This vector is used as MPI communication buffer
};

/**
 * Map of (MPI rank; SynapseCreationRequests)
 * The MPI rank specifies the corresponding process
 */
using MapSynapseCreationRequests = std::map<int, SynapseCreationRequests>;
