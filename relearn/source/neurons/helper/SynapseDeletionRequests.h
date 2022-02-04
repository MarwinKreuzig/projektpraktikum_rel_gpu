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
#include "RankNeuronId.h"

#include <map>
#include <utility>
#include <vector>

/**
 * This type represents a synapse deletion request.
 * It is exchanged via MPI ranks but does not perform MPI communication on its own.
 * A synapse is (src_neuron_id, axon, signal_type) ---synapse_id---> (tgt_neuron_id, dendrite, signal_type), the deletion if initiated from one side,
 * and the other side is saved as (affected_neuron_id, affected_element_type, signal_type)
 */
class PendingSynapseDeletion {
    RankNeuronId src_neuron_id{}; // Synapse source neuron id
    RankNeuronId tgt_neuron_id{}; // Synapse target neuron id
    RankNeuronId affected_neuron_id{}; // Neuron whose synaptic element should be set vacant
    ElementType affected_element_type{ ElementType::AXON }; // Type of the element (axon/dendrite) to be set vacant
    SignalType signal_type{ SignalType::EXCITATORY }; // Signal type (exc/inh) of the synapse
    unsigned int synapse_id{ 0 }; // Synapse id of the synapse to be deleted
    bool affected_element_already_deleted{ false }; // "True" if the element to be set vacant was already deleted by the neuron owning it
        // "False" if the element must be set vacant

public:
    /**
     * Creates a new object 
     */
    PendingSynapseDeletion() = default;

    /**
     * @brief Creates a new deletion request with the passed arguments
     * @param src The source neuron, i.e., the neuron which's axon is involved in the synapse
     * @param tgt The target neuron, i.e., the neuron which's dendrite is involved in the synapse
     * @param aff The affected neuron, i.e., the neuron that must be notified
     * @param elem The element type that is affected, i.e., from the neuron that must be notified
     * @param sign The signal type of the synapse
     * @param id The id of the synapse (in case the two neurons have multiple synapses)
     * @exception Throws a RelearnException if any RankNeuronId is invalid (negative MPI rank or too large neuron id)
     */
    PendingSynapseDeletion(const RankNeuronId& src, const RankNeuronId& tgt, const RankNeuronId& aff,
        const ElementType elem, const SignalType sign, const unsigned int id)
        : src_neuron_id(src)
        , tgt_neuron_id(tgt)
        , affected_neuron_id(aff)
        , affected_element_type(elem)
        , signal_type(sign)
        , synapse_id(id) {
        RelearnException::check(src.get_neuron_id() < Constants::uninitialized, "PendingSynapseDeletion::PendingSynapseDeletion(): src neuron id was too large");
        RelearnException::check(tgt.get_neuron_id() < Constants::uninitialized, "PendingSynapseDeletion::PendingSynapseDeletion(): tgt neuron id was too large");
        RelearnException::check(aff.get_neuron_id() < Constants::uninitialized, "PendingSynapseDeletion::PendingSynapseDeletion(): aff neuron id was too large");
        RelearnException::check(src.get_rank() >= 0, "PendingSynapseDeletion::PendingSynapseDeletion(): src MPI rank was negative");
        RelearnException::check(tgt.get_rank() >= 0, "PendingSynapseDeletion::PendingSynapseDeletion(): tgt MPI rank was negative");
        RelearnException::check(aff.get_rank() >= 0, "PendingSynapseDeletion::PendingSynapseDeletion(): aff MPI rank was negative");
    }

    PendingSynapseDeletion(const PendingSynapseDeletion& other) = default;
    PendingSynapseDeletion(PendingSynapseDeletion&& other) = default;

    PendingSynapseDeletion& operator=(const PendingSynapseDeletion& other) = default;
    PendingSynapseDeletion& operator=(PendingSynapseDeletion&& other) = default;

    ~PendingSynapseDeletion() = default;

    /**
     * @brief Returns the source neuron id, i.e., the neuron which's axon is involved in the deletion
     * @return The source neuron id
     */
    [[nodiscard]] const RankNeuronId& get_source_neuron_id() const noexcept {
        return src_neuron_id;
    }

    /**
     * @brief Returns the target neuron id, i.e., the neuron which's dendrite is involved in the deletion
     * @return The target neuron id
     */
    [[nodiscard]] const RankNeuronId& get_target_neuron_id() const noexcept {
        return tgt_neuron_id;
    }

    /**
     * @brief Returns the affected neuron id, i.e., the neuron that has to be notified of the deletion
     * @return The affected neuron id
     */
    [[nodiscard]] const RankNeuronId& get_affected_neuron_id() const noexcept {
        return affected_neuron_id;
    }

    /**
     * @brief Returns the affected element type, i.e., the affected neuron's type (axon or dendrite)
     * @return The affected element type
     */
    [[nodiscard]] ElementType get_affected_element_type() const noexcept {
        return affected_element_type;
    }

    /**
     * @brief Returns the synapse' signal type
     * @return The signal type
     */
    [[nodiscard]] SignalType get_signal_type() const noexcept {
        return signal_type;
    }

    /**
     * @brief Returns the synapse' id
     * @return The synapse' id
     */
    [[nodiscard]] unsigned int get_synapse_id() const noexcept {
        return synapse_id;
    }

    /**
     * @brief Returns the flag if the synapse is already deleted locally (in case both wanted to delete the same synapse)
     * @return True iff the synapse is already deleted
     */
    [[nodiscard]] bool get_affected_element_already_deleted() const noexcept {
        return affected_element_already_deleted;
    }

    /**
     * @brief Sets the flag that indicated if the synapse is already deleted locally (in case both wanted to delete the same synapse)
     */
    void set_affected_element_already_deleted() noexcept {
        affected_element_already_deleted = true;
    }

    /**
     * @brief Compares this and other by comparing the source neuron id, the target neuron id, and the synapse id
     * @param other The other deletion request
     * @return True iff both objects refer to the same synapse
     */
    [[nodiscard]] bool check_light_equality(const PendingSynapseDeletion& other) const noexcept {
        return check_light_equality(other.src_neuron_id, other.tgt_neuron_id, other.synapse_id);
    }

    /**
     * @brief Compares this and the passed components and checks if they refer to the same synapse
     * @param src The other source neuron id
     * @param tgt The other target neuron id
     * @param id The other synapse' id
     * @return True iff (src, tgt, id) refer to the same synapse as this
     */
    [[nodiscard]] bool check_light_equality(const RankNeuronId& src, const RankNeuronId& tgt, const unsigned int id) const noexcept {
        const bool src_neuron_id_eq = src == src_neuron_id;
        const bool tgt_neuron_id_eq = tgt == tgt_neuron_id;

        const bool id_eq = id == synapse_id;

        return src_neuron_id_eq && tgt_neuron_id_eq && id_eq;
    }

    template <std::size_t Index>
    auto& get() & {
        if constexpr (Index == 0)
            return src_neuron_id;
        if constexpr (Index == 1)
            return tgt_neuron_id;
        if constexpr (Index == 2)
            return affected_neuron_id;
        if constexpr (Index == 3)
            return affected_element_type;
        if constexpr (Index == 4)
            return signal_type;
        if constexpr (Index == 5)
            return synapse_id;
        if constexpr (Index == 6)
            return affected_element_already_deleted;
    }

    template <std::size_t Index>
    auto const& get() const& {
        if constexpr (Index == 0)
            return src_neuron_id;
        if constexpr (Index == 1)
            return tgt_neuron_id;
        if constexpr (Index == 2)
            return affected_neuron_id;
        if constexpr (Index == 3)
            return affected_element_type;
        if constexpr (Index == 4)
            return signal_type;
        if constexpr (Index == 5)
            return synapse_id;
        if constexpr (Index == 6)
            return affected_element_already_deleted;
    }

    template <std::size_t Index>
    auto&& get() && {
        if constexpr (Index == 0)
            return std::move(src_neuron_id);
        if constexpr (Index == 1)
            return std::move(tgt_neuron_id);
        if constexpr (Index == 2)
            return std::move(affected_neuron_id);
        if constexpr (Index == 3)
            return std::move(affected_element_type);
        if constexpr (Index == 4)
            return std::move(signal_type);
        if constexpr (Index == 5)
            return std::move(synapse_id);
        if constexpr (Index == 6)
            return std::move(affected_element_already_deleted);
    }
};
using PendingDeletionsV = std::vector<PendingSynapseDeletion>;

/**
 * This type aggregates multiple PendingSynapseDeletion into one and facilitates MPI communication.
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
     * @brief Appends the PendingSynapseDeletion to the end of the buffer
     * @param pending_deletion The new PendingSynapseDeletion that should be appended
     */
    void append(const PendingSynapseDeletion& pending_deletion) {
        requests.push_back(pending_deletion);
    }

    /**
     * @brief Returns the PendingSynapseDeletion with the requested index
     * @param request_index The index of the PendingSynapseDeletion
     * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
     * @return The deletion reques
     */
    [[nodiscard]] PendingSynapseDeletion get_request(const size_t request_index) const {
        RelearnException::check(request_index < requests.size(), "SynapseDeletionRequests::get_request: Index is out of bounds");
        return requests[request_index];
    }

    /**
     * @brief Returns the source neuron id of the PendingSynapseDeletion with the requested index
     * @param request_index The index of the PendingSynapseDeletion
     * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
     * @return The source neuron id
     */
    [[nodiscard]] size_t get_source_neuron_id(const size_t request_index) const {
        RelearnException::check(request_index < requests.size(), "SynapseDeletionRequests::get_source_neuron_id: Index is out of bounds");
        return requests[request_index].get_source_neuron_id().get_neuron_id();
    }

    /**
     * @brief Returns the target neuron id of the PendingSynapseDeletion with the requested index
     * @param request_index The index of the PendingSynapseDeletion
     * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
     * @return The target neuron id
     */
    [[nodiscard]] size_t get_target_neuron_id(const size_t request_index) const {
        RelearnException::check(request_index < requests.size(), "SynapseDeletionRequests::get_source_neuron_id: Index is out of bounds");
        return requests[request_index].get_target_neuron_id().get_neuron_id();
    }

    /**
     * @brief Returns the affected neuron id of the PendingSynapseDeletion with the requested index
     * @param request_index The index of the PendingSynapseDeletion
     * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
     * @return The affected neuron id
     */
    [[nodiscard]] size_t get_affected_neuron_id(const size_t request_index) const {
        RelearnException::check(request_index < requests.size(), "SynapseDeletionRequests::get_affected_neuron_id: Index is out of bounds");
        return requests[request_index].get_affected_neuron_id().get_neuron_id();
    }

    /**
     * @brief Returns the affected element type of the PendingSynapseDeletion with the requested index
     * @param request_index The index of the PendingSynapseDeletion
     * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
     * @return The element type
     */
    [[nodiscard]] ElementType get_affected_element_type(const size_t request_index) const {
        RelearnException::check(request_index < requests.size(), "SynapseDeletionRequests::get_affected_element_type: Index is out of bounds");
        return requests[request_index].get_affected_element_type();
    }

    /**
     * @brief Returns the synapse' signal type of the PendingSynapseDeletion with the requested index
     * @param request_index The index of the PendingSynapseDeletion
     * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
     * @return The synapse' signal type
     */
    [[nodiscard]] SignalType get_signal_type(const size_t request_index) const {
        RelearnException::check(request_index < requests.size(), "SynapseDeletionRequests::get_signal_type: Index is out of bounds");
        return requests[request_index].get_signal_type();
    }

    /**
     * @brief Returns the synapse' 9d of the PendingSynapseDeletion with the requested index
     * @param request_index The index of the PendingSynapseDeletion
     * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
     * @return The synapse' id
     */
    [[nodiscard]] unsigned int get_synapse_id(const size_t request_index) const {
        RelearnException::check(request_index < requests.size(), "SynapseDeletionRequests::get_synapse_id: Index is out of bounds");
        const auto synapse_id = requests[request_index].get_synapse_id();
        return synapse_id;
    }

    /**
     * @brief Returns the raw pointer to the requests.
     *      Does not transfer ownership
     * @return A raw pointer to the requests
     */
    [[nodiscard]] PendingSynapseDeletion* get_requests() noexcept {
        return requests.data();
    }

    /**
     * @brief Returns the raw pointer to the requests.
     *      Does not transfer ownership
     * @return A raw pointer to the requests
     */
    [[nodiscard]] const PendingSynapseDeletion* get_requests() const noexcept {
        return requests.data();
    }

    /**
     * @brief Returns the size of the internal buffer in bytes
     * @return The size of the internal buffer in bytes
     */
    [[nodiscard]] size_t get_requests_size_in_bytes() const noexcept {
        return requests.size() * sizeof(PendingSynapseDeletion);
    }

private:
    std::vector<PendingSynapseDeletion> requests{}; // This vector is used as MPI communication buffer

public:
    static std::map<int, SynapseDeletionRequests> exchange_requests(const PendingDeletionsV& pending_deletions) {
        /**
	     * - Go through list with pending synapse deletions and copy those into map "synapse_deletion_requests_outgoing"
	     *   where the other neuron affected by the deletion is not one of my neurons
	     * - Tell every rank how many deletion requests to receive from me
	     * - Prepare for corresponding number of deletion requests from every rank and receive them
	     * - Add received deletion requests to the list with pending deletions
	     * - Execute pending deletions
	     */

        /**
	     * Go through list with pending synapse deletions and copy those into
	     * map "synapse_deletion_requests_outgoing" where the other neuron
	     * affected by the deletion is not one of my neurons
	     */

        std::map<int, SynapseDeletionRequests> synapse_deletion_requests_incoming;
        std::map<int, SynapseDeletionRequests> synapse_deletion_requests_outgoing;
        // All pending deletion requests
        for (const auto& list_it : pending_deletions) {
            const auto target_rank = list_it.get_affected_neuron_id().get_rank();

            // Affected neuron of deletion request resides on different rank.
            // Thus the request needs to be communicated.
            if (target_rank != MPIWrapper::get_my_rank()) {
                synapse_deletion_requests_outgoing[target_rank].append(list_it);
            }
        }

        /**
	     * Send to every rank the number of deletion requests it should prepare for from me.
	     * Likewise, receive the number of deletion requests that I should prepare for from every rank.
	     */

        std::vector<size_t> num_synapse_deletion_requests_for_ranks(MPIWrapper::get_num_ranks(), 0);
        // Fill vector with my number of synapse deletion requests for every rank
        // Requests to myself are kept local and not sent to myself again.
        for (const auto& it : synapse_deletion_requests_outgoing) {
            auto rank = it.first;
            auto num_requests = it.second.size();

            num_synapse_deletion_requests_for_ranks[rank] = num_requests;
        }

        std::vector<size_t> num_synapse_deletion_requests_from_ranks(MPIWrapper::get_num_ranks(), Constants::uninitialized);
        // Send and receive the number of synapse deletion requests
        MPIWrapper::all_to_all(num_synapse_deletion_requests_for_ranks, num_synapse_deletion_requests_from_ranks);
        // Now I know how many requests I will get from every rank.
        // Allocate memory for all incoming synapse deletion requests.
        for (auto rank = 0; rank < MPIWrapper::get_num_ranks(); ++rank) {
            if (auto num_requests = num_synapse_deletion_requests_from_ranks[rank]; 0 != num_requests) {
                synapse_deletion_requests_incoming[rank].resize(num_requests);
            }
        }

        std::vector<MPIWrapper::AsyncToken> mpi_requests(synapse_deletion_requests_outgoing.size() + synapse_deletion_requests_incoming.size());

        /**
	     * Send and receive actual synapse deletion requests
	     */

        auto mpi_requests_index = 0;

        // Receive actual synapse deletion requests
        for (auto& it : synapse_deletion_requests_incoming) {
            const auto rank = it.first;
            auto* buffer = it.second.get_requests();
            const auto size_in_bytes = static_cast<int>(it.second.get_requests_size_in_bytes());

            MPIWrapper::async_receive(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

            ++mpi_requests_index;
        }

        // Send actual synapse deletion requests
        for (const auto& it : synapse_deletion_requests_outgoing) {
            const auto rank = it.first;
            const auto* const buffer = it.second.get_requests();
            const auto size_in_bytes = static_cast<int>(it.second.get_requests_size_in_bytes());

            MPIWrapper::async_send(buffer, size_in_bytes, rank, mpi_requests[mpi_requests_index]);

            ++mpi_requests_index;
        }

        // Wait for all sends and receives to complete
        MPIWrapper::wait_all_tokens(mpi_requests);

        return synapse_deletion_requests_incoming;
    }
};

/**
 * Map of (MPI rank; SynapseDeletionRequests)
 * The MPI rank specifies the corresponding process
 */
using MapSynapseDeletionRequests = std::map<int, SynapseDeletionRequests>;

namespace std {
template <>
struct tuple_size<typename ::PendingSynapseDeletion> {
    static constexpr size_t value = 7;
};

template <>
struct tuple_element<0, typename ::PendingSynapseDeletion> {
    using type = RankNeuronId;
};

template <>
struct tuple_element<1, typename ::PendingSynapseDeletion> {
    using type = RankNeuronId;
};

template <>
struct tuple_element<2, typename ::PendingSynapseDeletion> {
    using type = RankNeuronId;
};

template <>
struct tuple_element<3, typename ::PendingSynapseDeletion> {
    using type = ElementType;
};

template <>
struct tuple_element<4, typename ::PendingSynapseDeletion> {
    using type = SignalType;
};

template <>
struct tuple_element<5, typename ::PendingSynapseDeletion> {
    using type = unsigned int;
};

template <>
struct tuple_element<6, typename ::PendingSynapseDeletion> {
    using type = bool;
};

} //namespace std
