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

#include "util/MPIRank.h"
#include "util/RelearnException.h"

#include <functional>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/map.hpp>
#include <range/v3/view/transform.hpp>

/**
 * This type accumulates multiple values that should be exchanged between different MPI ranks.
 * It does not perform MPI communication on its own.
 *
 * @tparam RequestType The type of the values that should be exchanged
 */
template <typename RequestType>
class CommunicationMap {

public:
    using container_type = std::unordered_map<MPIRank, std::vector<RequestType>>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using size_type = typename container_type::size_type;
    using requests_size_type = typename std::vector<RequestType>::size_type;
    using sizes_type = std::unordered_map<MPIRank, requests_size_type>;

    /**
     * @brief Constructs a new communication map
     * @param number_ranks The number of MPI ranks. Is used to check later one for correct usage
     * @param size_hint The hint how many different ranks will have values stored in here. Does not need to match the final number.
     *      Can be ignored depending on the container_type.
     * @exception Throws a RelearnException if number_ranks is smaller than 1
     */
    explicit CommunicationMap(const size_t number_ranks, const size_type size_hint = 1)
        : number_ranks(number_ranks) {
        RelearnException::check(number_ranks > 0, "CommunicationMap::CommunicationMap: number_ranks is too small: {}", number_ranks);
        requests.reserve(size_hint);
    }

    /**
     * @brief Checks if there is data for the specified rank present
     * @param mpi_rank The MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks
     * @return True iff there is data for the MPI rank
     */
    [[nodiscard]] bool contains(const MPIRank mpi_rank) const {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::contains: mpi_rank is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::contains: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        return requests.find(mpi_rank) != requests.end();
    }

    /**
     * @brief Returns the number of data packages for MPI ranks
     * @return The number of ranks
     */
    [[nodiscard]] size_type size() const noexcept {
        return requests.size();
    }

    /**
     * @brief Returns the number of ranks that this map can hold
     * @return The number of ranks
     */
    [[nodiscard]] size_t get_number_ranks() const noexcept {
        return number_ranks;
    }

    /**
     * @brief Returns the total number of requests
     * @return The total number of requests
     */
    [[nodiscard]] requests_size_type get_total_number_requests() const noexcept {
        return ranges::accumulate(
            requests
                | ranges::views::values
                | ranges::views::transform(ranges::size),
            requests_size_type{ 0U });
    }

    /**
     * @brief Checks if there is data at all
     * @return True iff there is some data
     */
    [[nodiscard]] bool empty() const noexcept {
        return requests.empty();
    }

    /**
     * @brief Appends the request to the data for the specified MPI rank, inserts the requests for that rank if not yet present
     * @param mpi_rank The MPI rank
     * @param request The data for the MPI rank
     * @exception Throws a RelearnException
     */
    void append(const MPIRank mpi_rank, const RequestType& request) {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::append: mpi_rank is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::append: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        requests[mpi_rank].emplace_back(request);
    }

    /**
     * @brief Emplaces a newly created element in the communication map
     * @tparam ...ValueType The type for the constructor of the element
     * @param mpi_rank The MPI rank
     * @param ...Val The values for the constructor of the element
     * @exception Throws an exception if mpi_rank is negative or too large with respect to the number of ranks, if the memory allocation fails, or the constructor of the element throws
     * @return A reference to the newly created element
     */
    template <class... ValueType>
    constexpr decltype(auto) emplace_back(const MPIRank mpi_rank, ValueType&&... Val) {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::emplace_back: mpi_rank is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::emplace_back: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        return requests[mpi_rank].emplace_back(std::forward<ValueType>(Val)...);
    }

    /**
     * @brief Sets the request for the specified position
     * @param mpi_rank The MPI rank
     * @param request_index The index of the data package
     * @param request The data for the MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks,
     *      if the index is too large within the requests for that rank, or if there is no data for the MPI rank at all
     */
    void set_request(const MPIRank mpi_rank, const requests_size_type request_index, const RequestType& request) {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::set_request: mpi_rank is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::set_request: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        RelearnException::check(contains(mpi_rank), "CommunicationMap::set_request: Does not contain a buffer for rank {}", mpi_rank);
        RelearnException::check(request_index < size(mpi_rank), "CommunicationMap::set_request: The index was too large: {} vs {}", request_index, requests[mpi_rank].size());

        requests[mpi_rank][request_index] = request;
    }

    /**
     * @brief Returns the data for the specified rank and the specified index
     * @param mpi_rank The MPI rank
     * @param request_index The index of the data package
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks,
     *      if the index is too large within the requests for that rank, or if there is no data for the MPI rank at all
     * @return The data package
     */
    [[nodiscard]] RequestType get_request(const MPIRank mpi_rank, const requests_size_type request_index) const {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::get_request: mpi_rank is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::get_request: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
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
    [[nodiscard]] const std::vector<RequestType>& get_requests(const MPIRank mpi_rank) const {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::contains: get_requests is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::get_requests: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_requests: There are no requests for rank {}", mpi_rank);

        return requests.at(mpi_rank);
    }

    /**
     * @brief Returns all data for the specified MPI rank wrapped in an std::optional. If the MPI rank is not saved, returns the empty state.
     * @param mpi_rank The MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks
     * @return All data for the specified rank (might be empty)
     */
    [[nodiscard]] std::optional<std::reference_wrapper<const std::vector<RequestType>>> get_optional_request(const MPIRank mpi_rank) const {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::contains: get_optional_request is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::get_optional_request: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);

        const auto find_it = requests.find(mpi_rank);
        if (find_it == requests.end()) {
            return {};
        }

        const auto& reference = find_it->second;
        return {
            std::reference_wrapper{ reference }
        };
    }

    /**
     * @brief Resized the buffer for the data packages for a specified MPI rank, inserts the requests for that rank if not yet present
     * @param mpi_rank The MPI rank
     * @param size_for_rank The number of elements the buffer should be able to hold
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks
     */
    void resize(const MPIRank mpi_rank, const requests_size_type size_for_rank) {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::resize: mpi_rank is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::resize: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        requests[mpi_rank].resize(size_for_rank);
    }

    /**
     * @brief Resized the buffer for the data packages for all specified MPI ranks
     * @param sizes The sizes for the respective MPI ranks
     * @exception Throws a RelearnException if sizes.size() > number_ranks
     */
    void resize(std::vector<requests_size_type> sizes) {
        RelearnException::check(sizes.size() <= number_ranks, "CommunicationMap::resize: number of sizes {} is larger than the number of ranks {}", sizes.size(), number_ranks);

        for (auto mpi_rank = 0; mpi_rank < number_ranks; ++mpi_rank) {
            const auto size_for_ranks = sizes[mpi_rank];
            const auto rank = MPIRank{ mpi_rank };
            if (size_for_ranks == 0 && !contains(rank)) {
                // Don't want to insert an empty element into the container
                continue;
            }

            requests[rank].resize(size_for_ranks);
        }
    }

    /**
     * @brief Resized the buffer for the data packages for all specified MPI ranks
     * @param sizes The sizes for the respective MPI ranks (not specified ranks won't be added)
     * @exception Throws a RelearnException if sizes.size() > number_ranks
     */
    void resize(sizes_type sizes) {
        requests.clear();

        for (const auto& [mpi_rank, size_for_rank] : sizes) {
            RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::resize: The rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
            requests[mpi_rank].resize(size_for_rank);
        }
    }

    /**
     * @brief Clears the requests
     */
    void clear() {
        requests.clear();
    }

    /**
     * @brief Returns the number of packages for the specified MPI rank
     * @param mpi_rank The MPI rank
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks
     * @return The number of packages for the specified MPI rank. Is 0 if there is no data present
     */
    [[nodiscard]] requests_size_type size(const MPIRank mpi_rank) const {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::size: mpi_rank is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::size: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
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
    [[nodiscard]] requests_size_type get_size_in_bytes(const MPIRank mpi_rank) const {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::get_size_in_bytes: mpi_rank is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::get_size_in_bytes: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
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
    [[nodiscard]] RequestType* get_data(const MPIRank mpi_rank) {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::get_data: mpi_rank is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::get_data: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
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
    [[nodiscard]] const RequestType* get_data(const MPIRank mpi_rank) const {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::get_data const: mpi_rank is not initialized.");
        RelearnException::check(mpi_rank.get_rank() < number_ranks, "CommunicationMap::get_data const: rank {} is larger than the number of ranks {}", mpi_rank, number_ranks);
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_data const: There are no requests for rank {}", mpi_rank);

        return requests.at(mpi_rank).data();
    }

    /**
     * @brief Returns a span on the buffer for the specified rank
     * @param mpi_rank The MPI rank whose buffer should be queried
     * @exception Throws a RelearnException if mpi_rank is negative or too large with respect to the number of ranks
     * @exception Throws a RelearnException if mpi_rank is negative, the rank does not have saved requests, or the value is too large with respect to the number of ranks
     */
    [[nodiscard]] std::span<RequestType> get_span(const MPIRank mpi_rank) {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::get_span: mpi_rank is not initialized.");
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_span: There are no requests for rank {}", mpi_rank);
        return std::span<RequestType>{ requests.at(mpi_rank) };
    }

    /**
     * @brief Returns a constant span on the buffer for the specified rank
     * @param mpi_rank The MPI rank whose buffer should be queried
     * @exception Throws a RelearnException if mpi_rank is negative, the rank does not have saved requests, or the value is too large with respect to the number of ranks
     */
    [[nodiscard]] std::span<const RequestType> get_span(const MPIRank mpi_rank) const {
        RelearnException::check(mpi_rank.is_initialized(), "CommunicationMap::get_span const: mpi_rank is not initialized.");
        RelearnException::check(contains(mpi_rank), "CommunicationMap::get_span const: There are no requests for rank {}", mpi_rank);
        return std::span<const RequestType>{ requests.at(mpi_rank) };
    }

    /**
     * @brief Returns the number of requests for each MPI rank (includes those without requests with size 0)
     * @return Returns the number of requests for each MPI rank, i.e.,
     *      <return>[i] = k indicates that there are k requests for rank i
     */
    [[nodiscard]] std::vector<requests_size_type> get_request_sizes_vector() const noexcept {
        std::vector<requests_size_type> number_requests(number_ranks, 0);

        for (const auto& [rank, requests_for_rank] : requests) {
            number_requests[rank.get_rank()] = requests_for_rank.size();
        }

        return number_requests;
    }

    /**
     * @brief Returns the number of requests for each stored MPI rank (leaves out those that are not stored)
     * @return Returns the number of requests for the stored MPI rank, i.e.,
     *      <return>[i] = k indicates that there are k requests for rank i
     */
    [[nodiscard]] sizes_type get_request_sizes() const noexcept {
        return requests
            | ranges::views::transform([](const auto& rank_requests_pair) -> sizes_type::value_type {
                  return { rank_requests_pair.first, ranges::size(rank_requests_pair.second) };
              })
            | ranges::to<sizes_type>;
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
     * @brief Returns the constant begin-iterator
     * @return The begin-iterator
     */
    [[nodiscard]] const_iterator begin() const noexcept {
        return requests.begin();
    }

    /**
     * @brief Returns the constant end-iterator
     * @return The end-iterator
     */
    [[nodiscard]] const_iterator end() const noexcept {
        return requests.end();
    }

    /**
     * @brief Returns the constant begin-iterator
     * @return The begin-iterator
     */
    [[nodiscard]] const_iterator cbegin() const noexcept {
        return requests.cbegin();
    }

    /**
     * @brief Returns the constant end-iterator
     * @return The end-iterator
     */
    [[nodiscard]] const_iterator cend() const noexcept {
        return requests.cend();
    }

private:
    size_t number_ranks{};
    container_type requests{};
};