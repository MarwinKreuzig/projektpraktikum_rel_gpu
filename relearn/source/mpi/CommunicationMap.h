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
#include <span>
#include <vector>

/**
 * This type accumulates multiple values that should be exchanged between different MPI ranks.
 * It does not perform MPI communication on its own.
 * 
 * @tparam RequestType The type of the values that should be exchanged
 */
template <typename RequestType>
class CommunicationMap {

public:
    using iterator = typename std::map<int, std::vector<RequestType>>::iterator;
    using const_iterator = typename std::map<int, std::vector<RequestType>>::const_iterator;

    /**
     * @brief Constructs a new communication map 
     * @param number_ranks The number of MPI ranks. Is used to check later one for correct usage
     * @exception Throws a RelearnException if number_ranks is smaller than 1
     */
    CommunicationMap(const int number_ranks)
        : number_ranks(number_ranks) {
        RelearnException::check(number_ranks > 0, "CommunicationMap::CommunicationMap: number_ranks is too small: {}", number_ranks);
    }

    /**
     * @brief Checks if there is data for the specified rank present
     * @param mpi_rank The MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks
     * @return True iff there is data for the MPI rank
     */
    [[nodiscard]] bool contains(const int mpi_rank) const {
        RelearnException::check(0 <= mpi_rank && mpi_rank < number_ranks, "CommunicationMap::contains: rank {} is larger than the number of ranks {} (or negative)", mpi_rank, number_ranks);
        return requests.find(mpi_rank) != requests.end();
    }

    /**
     * @brief Returns the number of data packages for MPI ranks
     * @return The number of ranks
     */
    [[nodiscard]] size_t size() const noexcept {
        return requests.size();
    }

    /**
     * @brief Checks if there is data at all
     * @return True iff there is some data
     */
    [[nodiscard]] bool empty() const noexcept {
        return requests.empty();
    }

    /**
     * @brief Appends the request to the data for the specified MPI rank
     * @param mpi_rank The MPI rank
     * @param request The data for the MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks
     */
    void append(const int mpi_rank, const RequestType& request) {
        RelearnException::check(0 <= mpi_rank && mpi_rank < number_ranks, "CommunicationMap::append: rank {} is larger than the number of ranks {} (or negative)", mpi_rank, number_ranks);
        requests[mpi_rank].emplace_back(request);
    }

    /**
     * @brief Returns the data for the specified rank and the specified index
     * @param mpi_rank The MPI rank
     * @param request_index The index of the data package
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks,
     *      if the index is too large, or if there is no data for the MPI rank at all
     * @return The data package
     */
    [[nodiscard]] RequestType get_request(const int mpi_rank, const size_t request_index) const {
        RelearnException::check(0 <= mpi_rank && mpi_rank < number_ranks, "CommunicationMap::get_request: rank {} is larger than the number of ranks {} (or negative)", mpi_rank, number_ranks);
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_request: There are no requests for rank {}", mpi_rank);

        const auto& requests_for_rank = requests.at(mpi_rank);
        RelearnException::check(request_index < requests_for_rank.size(), "CommunicationMap::get_request: index out of bounds: {} vs {}", request_index, requests_for_rank.size());

        return requests_for_rank[request_index];
    }

    /**
     * @brief Returns all data for the specified MPI rank
     * @param mpi_rank The MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks, 
     *      or if there is no data for the MPI rank at all
     * @return All data for the specified rank
     */
    [[nodiscard]] const std::vector<RequestType>& get_requests(const int mpi_rank) const {
        RelearnException::check(0 <= mpi_rank && mpi_rank < number_ranks, "CommunicationMap::get_request: rank {} is larger than the number of ranks {} (or negative)", mpi_rank, number_ranks);
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_request: There are no requests for rank {}", mpi_rank);

        return requests.at(mpi_rank);
    }

    /**
     * @brief Resized the buffer for the data packages for a specified MPI rank
     * @param mpi_rank The MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks
     */
    void resize(const int mpi_rank, const size_t size) {
        RelearnException::check(0 <= mpi_rank && mpi_rank < number_ranks, "CommunicationMap::resize: rank {} is larger than the number of ranks {} (or negative)", mpi_rank, number_ranks);
        requests[mpi_rank].resize(size);
    }

    /**
     * @brief Returns the number of packages for the specified MPI rank
     * @param mpi_rank The MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks
     * @return The number of packages for the specified MPI rank. Is 0 if there is no data present
     */
    [[nodiscard]] size_t size(const int mpi_rank) const {
        RelearnException::check(0 <= mpi_rank && mpi_rank < number_ranks, "CommunicationMap::size: rank {} is larger than the number of ranks {} (or negative)", mpi_rank, number_ranks);
        if (!contains(mpi_rank)) {
            return 0;
        }

        return requests.at(mpi_rank).size();
    }

    /**
     * @brief Returns the number of bytes for the packages for the specified MPI rank
     * @param mpi_rank The MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks
     * @return The number of bytes for the packages for the specified MPI rank. Is 0 if there is no data present
     */
    [[nodiscard]] size_t get_size_in_bytes(const int mpi_rank) const {
        RelearnException::check(0 <= mpi_rank && mpi_rank < number_ranks, "CommunicationMap::get_data: rank {} is larger than the number of ranks {} (or negative)", mpi_rank, number_ranks);
        if (!contains(mpi_rank)) {
            return 0;
        }

        return requests.at(mpi_rank).size() * sizeof(RequestType);
    }

    /**
     * @brief Returns a non-owning pointer to the buffer for the specified MPI rank. 
     *      The pointer is invalidated by calls to resize or append.
     * @param mpi_rank The MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks,
     *      or if there is no data for the specified rank
     * @return A non-owning pointer to the buffer
     */
    [[nodiscard]] RequestType* get_data(const int mpi_rank) {
        RelearnException::check(0 <= mpi_rank && mpi_rank < number_ranks, "CommunicationMap::get_data: rank {} is larger than the number of ranks {} (or negative)", mpi_rank, number_ranks);
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_data: There are no requests for rank {}", mpi_rank);

        return requests.at(mpi_rank).data();
    }

    /**
     * @brief Returns a non-owning pointer to the buffer for the specified MPI rank. 
     *      The pointer is invalidated by calls to resize or append.
     * @param mpi_rank The MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks,
     *      or if there is no data for the specified rank
     * @return A non-owning pointer to the buffer
     */
    [[nodiscard]] const RequestType* get_data(const int mpi_rank) const {
        RelearnException::check(0 <= mpi_rank && mpi_rank < number_ranks, "CommunicationMap::get_data const: rank {} is larger than the number of ranks {} (or negative)", mpi_rank, number_ranks);
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_data const: There are no requests for rank {}", mpi_rank);

        return requests.at(mpi_rank).data();
    }

    std::span<RequestType> get_span(const int mpi_rank) {
        return std::span<RequestType>{ requests.at(mpi_rank) };
    }

    std::span<const RequestType> get_span(const int mpi_rank) const {
        return std::span<const RequestType>{ requests.at(mpi_rank) };
    }

    /**
     * @brief Returns the number of requests for each MPI rank
     * @return Returns the number of requests for each MPI rank, i.e.,
     *      <return>[i] = k indicates that there are k requests for rank i
     */
    [[nodiscard]] std::vector<size_t> get_request_sizes() const noexcept {
        std::vector<size_t> number_requests(number_ranks, 0);

        for (const auto& [rank, requests_for_rank] : requests) {
            number_requests[rank] = requests_for_rank.size();
        }

        return number_requests;
    }

    /**
     * @brief Returns the begin-iterator
     * @return The begin-iterator
     */
    [[nodiscard]] iterator begin() noexcept {
        return requests.begin();
    }

    /**
     * @brief Returns the end-iterator
     * @return The end-iterator
     */
    [[nodiscard]] iterator end() noexcept {
        return requests.end();
    }

    /**
     * @brief Returns the begin-iterator
     * @return The begin-iterator
     */
    [[nodiscard]] const_iterator begin() const noexcept {
        return requests.begin();
    }

    /**
     * @brief Returns the end-iterator
     * @return The end-iterator
     */
    [[nodiscard]] const_iterator end() const noexcept {
        return requests.end();
    }

    /**
     * @brief Returns the begin-iterator
     * @return The begin-iterator
     */
    [[nodiscard]] const_iterator cbegin() const noexcept {
        return requests.cbegin();
    }

    /**
     * @brief Returns the end-iterator
     * @return The end-iterator
     */
    [[nodiscard]] const_iterator cend() const noexcept {
        return requests.cend();
    }

private:
    std::map<int, std::vector<RequestType>> requests{};
    int number_ranks{};
};