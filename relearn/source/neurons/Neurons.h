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

#include "../Config.h"
#include "../algorithm/Algorithm.h"
#include "../util/RelearnException.h"
#include "../util/StatisticalMeasures.h"
#include "../util/Vec3.h"
#include "ElementType.h"
#include "NeuronsExtraInfo.h"
#include "SignalType.h"
#include "UpdateStatus.h"
#include "helper/RankNeuronId.h"
#include "helper/SynapseCreationRequests.h"
#include "models/NeuronModels.h"
#include "models/SynapticElements.h"

#include <array>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <tuple>
#include <vector>

class NetworkGraph;
class NeuronIdTranslator;
class NeuronMonitor;
class Octree;
class Partition;

/**
 * @brief This class gathers all information for the neurons and provides the primary interface for the simulation
 */
class Neurons {
    friend class NeuronMonitor;

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
            RelearnException::check(src.get_neuron_id().is_initialized, "PendingSynapseDeletion::PendingSynapseDeletion(): src neuron id not initialized");
            RelearnException::check(tgt.get_neuron_id().is_initialized, "PendingSynapseDeletion::PendingSynapseDeletion(): tgt neuron id not initialized");
            RelearnException::check(aff.get_neuron_id().is_initialized, "PendingSynapseDeletion::PendingSynapseDeletion(): aff neuron id not initialized");
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
     * This type is used as aggregate when it comes to selecting a synapse for deletion
     */
    class Synapse {
        RankNeuronId neuron_id{};
        unsigned int synapse_id{}; // Id of the synapse. Used to distinguish multiple synapses between the same neuron pair
    public:
        /**
         * @brief Creates a new object with the passed neuron id and synapse id
         * @param neuron_id The "other" neuron id
         * @param synapse_id The synapse' id, in case there are multiple synapses connecting two neurons
         * @exception Throws a RelearnException if the MPI rank is negative or neuron id is too large
         */
        Synapse(const RankNeuronId& neuron_id, const unsigned int synapse_id)
            : neuron_id(neuron_id)
            , synapse_id(synapse_id) {
            RelearnException::check(neuron_id.get_neuron_id().is_initialized, "Synapse::Synapse: neuron_id is not initialized");
            RelearnException::check(neuron_id.get_rank() >= 0, "Synapse::Synapse: neuron_id MPI rank was negative");
        }

        /**
         * @brief Returns the neuron id
         * @return The neuron id
         */
        [[nodiscard]] RankNeuronId get_neuron_id() const noexcept {
            return neuron_id;
        }

        /**
         * @brief Returns the synapse' id
         * @return The synapse' id
         */
        [[nodiscard]] unsigned int get_synapse_id() const noexcept {
            return synapse_id;
        }
    };

    /**
     * This type aggregates multiple PendingSynapseDeletion into one and facilitates MPI communication.
     * It does not perform MPI communication.
     */
    struct SynapseDeletionRequests {
        SynapseDeletionRequests() = default;

        /**
         * @brief Returns the number of stored requests
         * @return The number of stored requests
         */
        [[nodiscard]] size_t size() const noexcept {
            return num_requests;
        }

        /**
         * @brief Resizes the internal buffer to accomodate size-many requests
         * @param size The number of requests to be stored
         */
        void resize(const size_t size) {
            num_requests = size;
            requests.resize(Constants::num_items_per_request * size);
        }

        /**
         * @brief Appends the PendingSynapseDeletion to the end of the buffer
         * @param pending_deletion The new PendingSynapseDeletion that should be appended
         */
        void append(const PendingSynapseDeletion& pending_deletion) {
            num_requests++;

            const size_t affected_element_type_converted = pending_deletion.get_affected_element_type() == ElementType::AXON ? 0 : 1;
            const size_t signal_type_converted = pending_deletion.get_signal_type() == SignalType::EXCITATORY ? 0 : 1;

            requests.push_back(pending_deletion.get_source_neuron_id().get_neuron_id().id);
            requests.push_back(pending_deletion.get_target_neuron_id().get_neuron_id().id);
            requests.push_back(pending_deletion.get_affected_neuron_id().get_neuron_id().id);
            requests.push_back(affected_element_type_converted);
            requests.push_back(signal_type_converted);
            requests.push_back(pending_deletion.get_synapse_id());
        }

        /**
         * @brief Returns the source neuron id of the PendingSynapseDeletion with the requested index
         * @param request_index The index of the PendingSynapseDeletion
         * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
         * @return The source neuron id
         */
        [[nodiscard]] NeuronID get_source_neuron_id(const size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index;
            RelearnException::check(index < requests.size(), "SynapseDeletionRequests::get_source_neuron_id: Index is out of bounds");
            return NeuronID{ requests[index] };
        }

        /**
         * @brief Returns the target neuron id of the PendingSynapseDeletion with the requested index
         * @param request_index The index of the PendingSynapseDeletion
         * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
         * @return The target neuron id
         */
        [[nodiscard]] NeuronID get_target_neuron_id(const size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 1;
            RelearnException::check(index < requests.size(), "SynapseDeletionRequests::get_source_neuron_id: Index is out of bounds");
            return NeuronID{ requests[index] };
        }

        /**
         * @brief Returns the affected neuron id of the PendingSynapseDeletion with the requested index
         * @param request_index The index of the PendingSynapseDeletion
         * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
         * @return The affected neuron id
         */
        [[nodiscard]] NeuronID get_affected_neuron_id(const size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 2;
            RelearnException::check(index < requests.size(), "SynapseDeletionRequests::get_affected_neuron_id: Index is out of bounds");
            return NeuronID{ requests[index] };
        }

        /**
         * @brief Returns the affected element type of the PendingSynapseDeletion with the requested index
         * @param request_index The index of the PendingSynapseDeletion
         * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
         * @return The element type
         */
        [[nodiscard]] ElementType get_affected_element_type(const size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 3;
            RelearnException::check(index < requests.size(), "SynapseDeletionRequests::get_affected_element_type: Index is out of bounds");
            const auto affected_element_type_converted = requests[index] == 0 ? ElementType::AXON : ElementType::DENDRITE;
            return affected_element_type_converted;
        }

        /**
         * @brief Returns the synapse' signal type of the PendingSynapseDeletion with the requested index
         * @param request_index The index of the PendingSynapseDeletion
         * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
         * @return The synapse' signal type
         */
        [[nodiscard]] SignalType get_signal_type(const size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 4;
            RelearnException::check(index < requests.size(), "SynapseDeletionRequests::get_signal_type: Index is out of bounds");
            const auto affected_element_type_converted = requests[index] == 0 ? SignalType::EXCITATORY : SignalType::INHIBITORY;
            return affected_element_type_converted;
        }

        /**
         * @brief Returns the synapse' 9d of the PendingSynapseDeletion with the requested index
         * @param request_index The index of the PendingSynapseDeletion
         * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
         * @return The synapse' id
         */
        [[nodiscard]] unsigned int get_synapse_id(const size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 5;
            RelearnException::check(index < requests.size(), "SynapseDeletionRequests::get_synapse_id: Index is out of bounds");
            const auto synapse_id = static_cast<unsigned int>(requests[index]);
            return synapse_id;
        }

        /**
         * @brief Returns the raw pointer to the requests.
         *      Does not transfer ownership
         * @return A raw pointer to the requests
         */
        [[nodiscard]] size_t* get_requests() noexcept {
            return requests.data();
        }

        /**
         * @brief Returns the raw pointer to the requests.
         *      Does not transfer ownership
         * @return A raw pointer to the requests
         */
        [[nodiscard]] const size_t* get_requests() const noexcept {
            return requests.data();
        }

        /**
         * @brief Returns the size of the internal buffer in bytes
         * @return The size of the internal buffer in bytes
         */
        [[nodiscard]] size_t get_requests_size_in_bytes() const noexcept {
            return requests.size() * sizeof(size_t);
        }

    private:
        size_t num_requests{ 0 }; // Number of synapse deletion requests
        std::vector<size_t> requests{}; // Each request to delete a synapse is a 6-tuple:
                                        // (src_neuron_id, tgt_neuron_id, affected_neuron_id, affected_element_type, signal_type, synapse_id)
                                        // That is why requests.size() == 6*num_requests
                                        // Note, a more memory-efficient implementation would use a smaller data type (not size_t)
                                        // for affected_element_type, signal_type.
                                        // This vector is used as MPI communication buffer
    };

public:
    // Types
    using Axons = SynapticElements;
    using DendritesExcitatory = SynapticElements;
    using DendritesInhibitory = SynapticElements;

    /**
     * Map of (MPI rank; SynapseDeletionRequests)
     * The MPI rank specifies the corresponding process
     */
    using MapSynapseDeletionRequests = std::map<int, SynapseDeletionRequests>;

    /**
     * @brief Creates a new object with the passed Partition, NeuronModel, Axons, DendritesExc, and DendritesInh
     * @param partition The partition, is only used for printing, must not be empty
     * @param model_ptr The electrical model for the neurons, must not be empty
     * @param axons_ptr The model for the axons, must not be empty
     * @param dend_ex_ptr The model for the excitatory dendrites, must not be empty
     * @param dend_in_ptr The model for the inhibitory dendrites, must not be empty
     * @exception Throws a RelearnException if any of the pointers is empty
     */
    Neurons(std::shared_ptr<Partition> partition,
        std::unique_ptr<NeuronModel> model_ptr,
        std::unique_ptr<Axons> axons_ptr,
        std::unique_ptr<DendritesExcitatory> dend_ex_ptr,
        std::unique_ptr<DendritesInhibitory> dend_in_ptr)
        : partition(std::move(partition))
        , neuron_model(std::move(model_ptr))
        , axons(std::move(axons_ptr))
        , dendrites_exc(std::move(dend_ex_ptr))
        , dendrites_inh(std::move(dend_in_ptr)) {

        const bool all_filled = this->partition && neuron_model && axons && dendrites_exc && dendrites_inh;
        RelearnException::check(all_filled, "Neurons::Neurons: Neurons was constructed with some null arguments");
    }

    ~Neurons() = default;

    Neurons(const Neurons& other) = delete;
    Neurons(Neurons&& other) = default;

    Neurons& operator=(const Neurons& other) = delete;
    Neurons& operator=(Neurons&& other) = default;

    /**
     * @brief Initializes this class and all models with number_neurons, i.e.,
     *      (a) Initializes the electrical model
     *      (b) Initializes the extra infos
     *      (c) Initializes the synaptic models
     *      (d) Enables all neurons
     *      (e) Calculates if the neurons fired once to initialize the calcium values to beta or 0.0
     * @param number_neurons The number of local neurons
     * @param target_calcium_values The target calcium values for the local neurons
     * @param initial_calcium_values The initial calcium values for the local neurons
     * @exception Throws a RelearnException if target_calcium_values.size() != number_neurons, initial_calcium_values.size() != number_neurons, number_neurons == 0, or something unexpected happened
     */
    void init(size_t number_neurons, std::vector<double> target_calcium_values, std::vector<double> initial_calcium_values);

    /**
     * @brief Sets the octree in which the neurons are stored
     * @param octree The octree
     */
    void set_octree(std::shared_ptr<Octree> octree) noexcept {
        global_tree = std::move(octree);
    }

    /**
     * @brief Sets the algorithm that calculates to which neuron a neuron connects during the plasticity update
     * @param algorithm_ptr The pointer to the algorithm
     */
    void set_algorithm(std::shared_ptr<Algorithm> algorithm_ptr) noexcept {
        algorithm = std::move(algorithm_ptr);
    }

    /**
     * @brief Sets the network graph in which the synapses for the neurons are stored
     * @param octree The network graph
     */
    void set_network_graph(std::shared_ptr<NetworkGraph> network) noexcept {
        network_graph = std::move(network);
    }

    /**
     * @brief Sets the neuron id translator for the neurons are stored
     * @param neuron_id_translator The translator
     */
    void set_neuron_id_translator(std::shared_ptr<NeuronIdTranslator> neuron_id_translator) {
        translator = std::move(neuron_id_translator);
    }

    /**
     * @brief Returns the model parameters for the specified synaptic elements
     * @param element_type The element type
     * @param signal_type The signal type, only relevant if element_type == dendrites
     * @return The model parameters for the specified synaptic elements
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter(const ElementType element_type, const SignalType signal_type) {
        if (element_type == ElementType::AXON) {
            return axons->get_parameter();
        }

        if (signal_type == SignalType::EXCITATORY) {
            return dendrites_exc->get_parameter();
        }

        return dendrites_inh->get_parameter();
    }

    /**
     * @brief Returns the number of neurons in this object
     * @return The number of neurons in this object
     */
    [[nodiscard]] size_t get_num_neurons() const noexcept {
        return number_neurons;
    }

    /**
     * @brief Sets the area names in the extra infos
     * @param names The area names
     * @exception Throws the same RelearnException as NeuronsExtraInfo::set_area_names
     */
    void set_area_names(std::vector<std::string> names) {
        extra_info->set_area_names(std::move(names));
    }

    /**
     * @brief Sets the positions in the extra infos
     * @param names The positions
     * @exception Throws the same RelearnException as NeuronsExtraInfo::set_positions
     */
    void set_positions(std::vector<NeuronsExtraInfo::position_type> pos) {
        extra_info->set_positions(std::move(pos));
    }

    /**
     * @brief Returns a constant reference to the extra informations
     * @return The extra informations for the neurons
     */
    const std::unique_ptr<NeuronsExtraInfo>& get_extra_info() const noexcept {
        return extra_info;
    }

    /**
     * @brief Sets the signal types in the extra infos
     * @param names The signal types
     * @exception Throws the same RelearnException as NeuronsExtraInfo::set_signal_types
     */
    void set_signal_types(std::vector<SignalType> signal_types) {
        axons->set_signal_types(std::move(signal_types));
    }

    /**
     * @brief Returns a constant reference to the axon model
     *      The reference is never invalidated
     * @return A constant reference to the axon model
     */
    [[nodiscard]] const Axons& get_axons() const noexcept {
        return *axons;
    }

    /**
     * @brief Returns a constant reference to the excitatory dendrites model
     *      The reference is never invalidated
     * @return A constant reference to the excitatory dendrites model
     */
    [[nodiscard]] const DendritesExcitatory& get_dendrites_exc() const noexcept {
        return *dendrites_exc;
    }

    /**
     * @brief Returns a constant reference to the inhibitory dendrites model
     *      The reference is never invalidated
     * @return A constant reference to the inhibitory dendrites model
     */
    [[nodiscard]] const DendritesInhibitory& get_dendrites_inh() const noexcept {
        return *dendrites_inh;
    }

    /**
     * @brief Returns a constant reference to the disable flags for the neurons
     *      The reference is never invalidated
     * @return A constant reference to the disable flags
     */
    [[nodiscard]] const std::vector<UpdateStatus>& get_disable_flags() const noexcept {
        return disable_flags;
    }

    /**
     * @brief Initializes the synaptic elements with respect to the network graph, i.e.,
     *      adds the synapses from the network graph as connected counts to the synaptic elements models
     */
    void init_synaptic_elements();

    /**
     * @brief Disables all neurons with specified ids
     *      If a neuron is already disabled, nothing happens for that one
     *      Otherwise, also deletes all synapses from the disabled neurons
     * @exception Throws RelearnExceptions if something unexpected happens
     */
    size_t disable_neurons(const std::vector<NeuronID>& neuron_ids);

    /**
     * @brief Enables all neurons with specified ids
     *      If a neuron is already enabled, nothing happens for that one
     * @exception Throws RelearnExceptions if something unexpected happens
     */
    void enable_neurons(const std::vector<NeuronID>& neuron_ids);

    /**
     * @brief Creates creation_count many new neurons with default values
     *      (a) Creates neurons in the electrical model
     *      (b) Creates neurons in the extra infos
     *      (c) Creates neurons in the synaptic models
     *      (d) Enables all created neurons
     *      (e) Calculates if the neurons fired once to initialize the calcium values to beta or 0.0
     *      (f) Inserts the newly created neurons into the octree
     * @param creation_count The number of newly created neurons
     * @param new_target_calcium_values The target calcium values for the newly created neurons
     * @param new_initial_calcium_values The initial calcium values for the newly created neurons
     * @exception Throws a RelearnException if creation_count != new_target_calcium_values.size(), or if something unexpected happens
     */
    void create_neurons(size_t creation_count, const std::vector<double>& new_target_calcium_values, const std::vector<double>& new_initial_calcium_values);

    /**
     * @brief Calls update_electrical_activity from the electrical model with the stored network graph,
     *      and updates the calcium values afterwards
     * @exception Throws a RelearnException if something unexpected happens
     */
    void update_electrical_activity();

    /**
     * @brief Updates the delta of the synaptic elements for (1) axons, (2) excitatory dendrites, (3) inhibitory dendrites
     * @exception Throws a RelearnException if something unexpected happens
     */
    void update_number_synaptic_elements_delta() {
        axons->update_number_elements_delta(calcium, target_calcium, disable_flags);
        dendrites_exc->update_number_elements_delta(calcium, target_calcium, disable_flags);
        dendrites_inh->update_number_elements_delta(calcium, target_calcium, disable_flags);
    }

    /**
     * @brief Updates the plasticity by
     *      (1) Deleting superfluous synapses
     *      (2) Creating new synapses with the stored algorithm
     * @exception Throws a RelearnException if the network graph, the octree, or the algorithm is empty,
     *      or something unexpected happens
     * @return Returns a tuple with (1) the number of deleted synapses, and (2) the number of created synapses
     */
    [[nodiscard]] std::tuple<size_t, size_t> update_connectivity();

    /**
     * @brief Calculates the number vacant axons and dendrites (excitatory, inhibitory) and prints them to LogFiles::EventType::Sums
     *      Performs communication with MPI
     * @param step The current simulation step
     * @param sum_synapses_deleted The number of deleted synapses (locally)
     * @param sum_synapses_created The number of created synapses (locally)
     */
    void print_sums_of_synapses_and_elements_to_log_file_on_rank_0(size_t step, size_t sum_synapses_deleted, size_t sum_synapses_created);

    /**
     * @brief Prints the overview of the neurons to LogFiles::EventType::NeuronsOverview
     *      Performs communication with MPI
     * @param step The current simulation step
     */
    void print_neurons_overview_to_log_file_on_rank_0(size_t step);

    /**
     * @brief Prints the calcium statistics to LogFiles::EventType::Essentials
     *      Performs communication with MPI
     */
    void print_calcium_statistics_to_essentials();

    /**
     * @brief Prints the network graph to LogFiles::EventType::Network
     */
    void print_network_graph_to_log_file();

    /**
     * @brief Prints the neuron positions to LogFiles::EventType::Positions
     */
    void print_positions_to_log_file();

    /**
     * @brief Prints some overview to LogFiles::EventType::Cout
     */
    void print();

    /**
     * @brief Prints some algorithm overview to LogFiles::EventType::Cout
     */
    void print_info_for_algorithm();

    /**
     * @brief Prints the histogram of in edges for the local neurons at the current simulation step
     * @param current_step The current simulation step
     */
    void print_local_network_histogram(size_t current_step);

    /**
     * @brief Prints the histogram of out edges for the local neurons at the current simulation step
     * @param current_step The current simulation step
     */
    void print_calcium_values_to_file(size_t current_step);

    /**
     * @brief Performs debug checks on the synaptic element models if Config::do_debug_checks
     * @exception Throws a RelearnException if a check fails
     */
    void debug_check_counts();

    /**
     * @brief Returns a statistical measure for the selected attribute, considering all MPI ranks.
     *      Performs communication across MPI processes
     * @param attribute The selected attribute of the neurons
     * @return The statistical measure across all MPI processes. All MPI processes have the same return value
     */
    StatisticalMeasures get_statistics(NeuronAttribute attribute) const;

private:
    void update_calcium();

    [[nodiscard]] StatisticalMeasures global_statistics(const std::vector<double>& local_values, int root, const std::vector<UpdateStatus>& disable_flags) const;

    template <typename T>
    [[nodiscard]] StatisticalMeasures global_statistics_integral(const std::vector<T>& local_values, const int root, const std::vector<UpdateStatus>& disable_flags) const {
        std::vector<double> converted_values;
        converted_values.reserve(local_values.size());

        for (const auto& value : local_values) {
            converted_values.emplace_back(static_cast<double>(value));
        }

        return global_statistics(converted_values, root, disable_flags);
    }

    [[nodiscard]] size_t delete_synapses();

    [[nodiscard]] std::pair<PendingDeletionsV, std::vector<size_t>> delete_synapses_find_synapses(
        const SynapticElements& synaptic_elements, const std::pair<unsigned int, std::vector<unsigned int>>& to_delete, const PendingDeletionsV& other_pending_deletions);

    /**
     * Determines which synapses should be deleted.
     * The selected synapses connect with neuron "neuron_id" and the type of
     * those synapses is given by "signal_type".
     *
     * NOTE: The semantics of the function is not nice but used to postpone all updates
     * due to synapse deletion until all neurons have decided *independently* which synapse
     * to delete. This should reflect how it's done for a distributed memory implementation.
     */
    [[nodiscard]] std::vector<size_t> delete_synapses_find_synapses_on_neuron(
        const NeuronID& neuron_id,
        ElementType element_type,
        SignalType signal_type,
        unsigned int num_synapses_to_delete,
        PendingDeletionsV& pending_deletions,
        const PendingDeletionsV& other_pending_deletions);

    [[nodiscard]] static std::vector<Neurons::Synapse> delete_synapses_register_edges(const std::vector<std::pair<RankNeuronId, int>>& edges);

    [[nodiscard]] static MapSynapseDeletionRequests delete_synapses_exchange_requests(const PendingDeletionsV& pending_deletions);

    static void delete_synapses_process_requests(const MapSynapseDeletionRequests& synapse_deletion_requests_incoming, PendingDeletionsV& pending_deletions);

    [[nodiscard]] size_t delete_synapses_commit_deletions(const PendingDeletionsV& list);

    [[nodiscard]] size_t create_synapses();

    void create_synapses_update_octree();

    [[nodiscard]] static MapSynapseCreationRequests create_synapses_exchange_requests(const MapSynapseCreationRequests& synapse_creation_requests_outgoing);

    [[nodiscard]] std::pair<size_t, std::map<int, std::vector<char>>> create_synapses_process_requests(const MapSynapseCreationRequests& synapse_creation_requests_incoming);

    [[nodiscard]] static std::map<int, std::vector<char>> create_synapses_exchange_responses(const std::map<int, std::vector<char>>& synapse_creation_responses, const MapSynapseCreationRequests& synapse_creation_requests_outgoing);

    [[nodiscard]] size_t create_synapses_process_responses(const MapSynapseCreationRequests& synapse_creation_requests_outgoing, const std::map<int, std::vector<char>>& received_responses);

    static void print_pending_synapse_deletions(const PendingDeletionsV& list);

    size_t number_neurons = 0;

    std::shared_ptr<Partition> partition{};

    std::shared_ptr<Octree> global_tree{};
    std::shared_ptr<Algorithm> algorithm{};

    std::shared_ptr<NetworkGraph> network_graph{};
    std::shared_ptr<NeuronIdTranslator> translator{};

    std::unique_ptr<NeuronModel> neuron_model{};

    std::unique_ptr<Axons> axons{};
    std::unique_ptr<DendritesExcitatory> dendrites_exc{};
    std::unique_ptr<DendritesInhibitory> dendrites_inh{};

    std::vector<double> target_calcium{};
    std::vector<double> calcium{}; // Intracellular calcium concentration of every neuron

    std::vector<UpdateStatus> disable_flags{};

    std::unique_ptr<NeuronsExtraInfo> extra_info{ std::make_unique<NeuronsExtraInfo>() };
};
