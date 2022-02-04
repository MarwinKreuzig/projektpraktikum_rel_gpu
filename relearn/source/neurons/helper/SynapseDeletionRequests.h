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
#include "RankNeuronId.h"

#include <map>
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
};

/**
 * Map of (MPI rank; SynapseDeletionRequests)
 * The MPI rank specifies the corresponding process
 */
using MapSynapseDeletionRequests = std::map<int, SynapseDeletionRequests>;
