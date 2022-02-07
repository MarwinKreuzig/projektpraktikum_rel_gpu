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

#include "../util/RelearnException.h"

#include <map>
#include <vector>

template <typename RequestType>
class CommunicationMap {

public:
    using iterator = typename std::map<int, std::vector<RequestType>>::iterator;
    using const_iterator = typename std::map<int, std::vector<RequestType>>::const_iterator;

    CommunicationMap(const int my_rank, const int number_ranks)
        : my_rank(my_rank)
        , number_ranks(number_ranks) { }

    [[nodiscard]] bool contains(const int mpi_rank) const {
        RelearnException::check(mpi_rank < number_ranks, "CommunicationMap::contains: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        return requests.find(mpi_rank) != requests.end();
    }

    [[nodiscard]] size_t size() const noexcept {
        return requests.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return requests.empty();
    }

    void append(const int mpi_rank, const RequestType& request) {
        RelearnException::check(mpi_rank < number_ranks, "CommunicationMap::append: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        requests[mpi_rank].emplace_back(request);
    }

    [[nodiscard]] RequestType get_request(const int mpi_rank, const size_t request_index) const {
        RelearnException::check(mpi_rank < number_ranks, "CommunicationMap::get_request: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_request: There are no requests for rank {}", mpi_rank);

        const auto& requests_for_rank = requests.at(mpi_rank);
        RelearnException::check(request_index < requests_for_rank.size(), "CommunicationMap::get_request: index out of bounds: {} vs {}", request_index, requests_for_rank.size());

        return requests_for_rank[request_index];
    }

    [[nodiscard]] const std::vector<RequestType>& get_requests(const int mpi_rank) const {
        RelearnException::check(mpi_rank < number_ranks, "CommunicationMap::get_request: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_request: There are no requests for rank {}", mpi_rank);

        return requests.at(mpi_rank);
    }

    void resize(const int mpi_rank, const size_t size) {
        RelearnException::check(mpi_rank < number_ranks, "CommunicationMap::resize: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        requests[mpi_rank].resize(size);
    }

    [[nodiscard]] size_t size(const int mpi_rank) const {
        RelearnException::check(mpi_rank < number_ranks, "CommunicationMap::size: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        if (!contains(mpi_rank)) {
            return 0;
        }

        return requests.at(mpi_rank).size();
    }

    [[nodiscard]] size_t get_size_in_bytes(const int mpi_rank) const {
        RelearnException::check(mpi_rank < number_ranks, "CommunicationMap::get_data: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        if (!contains(mpi_rank)) {
            return 0;
        }

        return requests.at(mpi_rank).size() * sizeof(RequestType);
    }

    [[nodiscard]] RequestType* get_data(const int mpi_rank) {
        RelearnException::check(mpi_rank < number_ranks, "CommunicationMap::get_data: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_data: There are no requests for rank {}", mpi_rank);

        return requests.at(mpi_rank).data();
    }

    [[nodiscard]] const RequestType* get_data(const int mpi_rank) const {
        RelearnException::check(mpi_rank < number_ranks, "CommunicationMap::get_data const: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_data const: There are no requests for rank {}", mpi_rank);

        return requests.at(mpi_rank).data();
    }

    [[nodiscard]] std::vector<size_t> get_request_sizes() const noexcept {
        std::vector<size_t> number_requests(number_ranks, 0);

        for (const auto& [rank, requests_for_rank] : requests) {
            number_requests[rank] = requests_for_rank.size();
        }

        return number_requests;
    }

    [[nodiscard]] iterator begin() noexcept {
        return requests.begin();
    }

    [[nodiscard]] iterator end() noexcept {
        return requests.end();
    }

    [[nodiscard]] const_iterator begin() const noexcept {
        return requests.begin();
    }

    [[nodiscard]] const_iterator end() const noexcept {
        return requests.end();
    }

    [[nodiscard]] const_iterator cbegin() const noexcept {
        return requests.cbegin();
    }

    [[nodiscard]] const_iterator cend() const noexcept {
        return requests.cend();
    }

private:
    std::map<int, std::vector<RequestType>> requests{};
    int my_rank{};
    int number_ranks{};
};