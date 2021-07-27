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

#include "../algorithm/BarnesHut.h"
#include "../Config.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"
#include "ElementType.h"
#include "NeuronsExtraInfo.h"
#include "SignalType.h"
#include "helper/RankNeuronId.h"
#include "helper/SynapseCreationRequests.h"
#include "models/NeuronModels.h"
#include "models/SynapticElements.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

class NetworkGraph;
class NeuronMonitor;
class Octree;
class Partition;

// Types
using Axons = SynapticElements;
using DendritesExc = SynapticElements;
using DendritesInh = SynapticElements;

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
            ElementType elem, SignalType sign, unsigned int id)
            : src_neuron_id(src)
            , tgt_neuron_id(tgt)
            , affected_neuron_id(aff)
            , affected_element_type(elem)
            , signal_type(sign)
            , synapse_id(id) {
            RelearnException::check(src.get_neuron_id() < Constants::uninitialized, "PendingSynapseDeletion::PendingSynapseDeletion(), src neuron id was too large");
            RelearnException::check(tgt.get_neuron_id() < Constants::uninitialized, "PendingSynapseDeletion::PendingSynapseDeletion(), tgt neuron id was too large");
            RelearnException::check(aff.get_neuron_id() < Constants::uninitialized, "PendingSynapseDeletion::PendingSynapseDeletion(), aff neuron id was too large");
            RelearnException::check(src.get_rank() >= 0, "PendingSynapseDeletion::PendingSynapseDeletion(), src MPI rank was negative");
            RelearnException::check(tgt.get_rank() >= 0, "PendingSynapseDeletion::PendingSynapseDeletion(), tgt MPI rank was negative");
            RelearnException::check(aff.get_rank() >= 0, "PendingSynapseDeletion::PendingSynapseDeletion(), aff MPI rank was negative");
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
        [[nodiscard]] bool check_light_equality(const RankNeuronId& src, const RankNeuronId& tgt, unsigned int id) const noexcept {
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
        Synapse(RankNeuronId neuron_id, unsigned int synapse_id)
            : neuron_id(neuron_id)
            , synapse_id(synapse_id) {
            RelearnException::check(neuron_id.get_neuron_id() < Constants::uninitialized, "Synapse::Synapse(), neuron id was too large");
            RelearnException::check(neuron_id.get_rank() >= 0, "Synapse::Synapse(), neuron_id MPI rank was negative");
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
        void resize(size_t size) {
            num_requests = size;
            requests.resize(Constants::num_items_per_request * size);
        }

        /**
         * @brief Appends the PendingSynapseDeletion to the end of the buffer
         * @param pending_deletion The new PendingSynapseDeletion that should be appended
         */
        void append(const PendingSynapseDeletion& pending_deletion) {
            num_requests++;

            size_t affected_element_type_converted = pending_deletion.get_affected_element_type() == ElementType::AXON ? 0 : 1;
            size_t signal_type_converted = pending_deletion.get_signal_type() == SignalType::EXCITATORY ? 0 : 1;

            requests.push_back(pending_deletion.get_source_neuron_id().get_neuron_id());
            requests.push_back(pending_deletion.get_target_neuron_id().get_neuron_id());
            requests.push_back(pending_deletion.get_affected_neuron_id().get_neuron_id());
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
        [[nodiscard]] size_t get_source_neuron_id(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
            return requests[index];
        }

        /**
         * @brief Returns the target neuron id of the PendingSynapseDeletion with the requested index
         * @param request_index The index of the PendingSynapseDeletion
         * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
         * @return The target neuron id
         */
        [[nodiscard]] size_t get_target_neuron_id(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 1;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
            return requests[index];
        }

        /**
         * @brief Returns the affected neuron id of the PendingSynapseDeletion with the requested index
         * @param request_index The index of the PendingSynapseDeletion
         * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
         * @return The affected neuron id
         */
        [[nodiscard]] size_t get_affected_neuron_id(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 2;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
            return requests[index];
        }

        /**
         * @brief Returns the affected element type of the PendingSynapseDeletion with the requested index
         * @param request_index The index of the PendingSynapseDeletion
         * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
         * @return The element type
         */
        [[nodiscard]] ElementType get_affected_element_type(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 3;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
            const auto affected_element_type_converted = requests[index] == 0 ? ElementType::AXON : ElementType::DENDRITE;
            return affected_element_type_converted;
        }

        /**
         * @brief Returns the synapse' signal type of the PendingSynapseDeletion with the requested index
         * @param request_index The index of the PendingSynapseDeletion
         * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
         * @return The synapse' signal type
         */
        [[nodiscard]] SignalType get_signal_type(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 4;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
            const auto affected_element_type_converted = requests[index] == 0 ? SignalType::EXCITATORY : SignalType::INHIBITORY;
            return affected_element_type_converted;
        }

        /**
         * @brief Returns the synapse' 9d of the PendingSynapseDeletion with the requested index
         * @param request_index The index of the PendingSynapseDeletion
         * @exception Throws a RelearnException if request_index is larger than the number of stored PendingSynapseDeletion
         * @return The synapse' id
         */
        [[nodiscard]] unsigned int get_synapse_id(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 5;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
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

    /**
     * @brief This struct is used to aggregate different statistical parameters
     */
    struct StatisticalMeasures {
        double min{ 0.0 };
        double max{ 0.0 };
        double avg{ 0.0 };
        double var{ 0.0 };
        double std{ 0.0 };
    };

public:
    /**
	 * Map of (MPI rank; SynapseDeletionRequests)
	 * The MPI rank specifies the corresponding process
	 */
    using MapSynapseDeletionRequests = std::map<int, SynapseDeletionRequests>;

    Neurons(std::shared_ptr<Partition> partition,
        std::unique_ptr<NeuronModel> model,
        std::unique_ptr<Axons> axons_ptr,
        std::unique_ptr<DendritesExc> dend_ex_ptr,
        std::unique_ptr<DendritesInh> dend_in_ptr)
        : partition(std::move(partition))
        , neuron_model(std::move(model))
        , axons(std::move(axons_ptr))
        , dendrites_exc(std::move(dend_ex_ptr))
        , dendrites_inh(std::move(dend_in_ptr)) {

        const bool all_filled = this->partition && neuron_model && axons && dendrites_exc && dendrites_inh;
        RelearnException::check(all_filled, "Neurons was constructed with some null arguments");
    }

    ~Neurons() = default;

    Neurons(const Neurons& other) = delete;
    Neurons(Neurons&& other) = default;

    Neurons& operator=(const Neurons& other) = delete;
    Neurons& operator=(Neurons&& other) = default;

    void init(size_t number_neurons);

    void set_octree(std::shared_ptr<Octree> octree) {
        global_tree = std::move(octree);
    }

    void set_barnes_hut(std::shared_ptr<BarnesHut> barnes_hut) {
        barnes_hut_algorithm = std::move(barnes_hut);
    }

    void set_network_graph(std::shared_ptr<NetworkGraph> network) {
        network_graph = std::move(network);
    }

    [[nodiscard]] std::vector<ModelParameter> get_parameter(ElementType element_type, SignalType signal_type) {
        if (element_type == ElementType::AXON) {
            return axons->get_parameter();
        }

        if (signal_type == SignalType::EXCITATORY) {
            return dendrites_exc->get_parameter();
        }

        return dendrites_inh->get_parameter();
    }

    [[nodiscard]] size_t get_num_neurons() const noexcept {
        return num_neurons;
    }

    void set_area_names(std::vector<std::string> names) {
        extra_info->set_area_names(std::move(names));
    }

    void set_x_dims(std::vector<double> x_dims) {
        extra_info->set_x_dims(std::move(x_dims));
    }

    void set_y_dims(std::vector<double> y_dims) {
        extra_info->set_y_dims(std::move(y_dims));
    }

    void set_z_dims(std::vector<double> z_dims) {
        extra_info->set_z_dims(std::move(z_dims));
    }

    void set_signal_types(std::vector<SignalType> signal_types) {
        axons->set_signal_types(std::move(signal_types));
    }

    [[nodiscard]] const Axons& get_axons() const noexcept {
        return *axons;
    }

    [[nodiscard]] const DendritesExc& get_dendrites_exc() const noexcept {
        return *dendrites_exc;
    }

    [[nodiscard]] const DendritesInh& get_dendrites_inh() const noexcept {
        return *dendrites_inh;
    }

    [[nodiscard]] const std::vector<char>& get_disable_flags() const noexcept {
        return disable_flags;
    }

    void init_synaptic_elements();

    /**
     * Disables all neurons with specified ids
     * If a neuron is already disabled, nothing happens for that one
     */
    size_t disable_neurons(const std::vector<size_t>& neuron_ids);

    /**
     * Enables all neurons with specified ids
     * If a neuron is already enabled, nothing happens for that one
     */
    void enable_neurons(const std::vector<size_t>& neuron_ids);

    /**
     * Creates creation_count many new neurons with default values
     */
    void create_neurons(size_t creation_count);

    void update_electrical_activity();

    void update_number_synaptic_elements_delta() noexcept {
        axons->update_number_elements_delta(calcium, disable_flags);
        dendrites_exc->update_number_elements_delta(calcium, disable_flags);
        dendrites_inh->update_number_elements_delta(calcium, disable_flags);
    }

    [[nodiscard]] std::tuple<size_t, size_t> update_connectivity();

    void print_sums_of_synapses_and_elements_to_log_file_on_rank_0(size_t step, size_t sum_synapses_deleted, size_t sum_synapses_created);

    // Print global information about all neurons at rank 0
    void print_neurons_overview_to_log_file_on_rank_0(size_t step);

    void print_statistics_to_essentials();

    void print_network_graph_to_log_file();

    void print_positions_to_log_file();

    void print();

    void print_info_for_barnes_hut();

    void debug_check_counts();

private:
    void update_calcium();

    [[nodiscard]] StatisticalMeasures global_statistics(const std::vector<double>& local_values, int root, const std::vector<char>& disable_flags) const;

    template <typename T>
    [[nodiscard]] StatisticalMeasures global_statistics_integral(const std::vector<T>& local_values, int root, const std::vector<char>& disable_flags) const {
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
    [[nodiscard]] std::vector<size_t> delete_synapses_find_synapses_on_neuron(size_t neuron_id,
        ElementType element_type,
        SignalType signal_type,
        unsigned int num_synapses_to_delete,
        PendingDeletionsV& pending_deletions,
        const PendingDeletionsV& other_pending_deletions);

    [[nodiscard]] static std::vector<Neurons::Synapse> delete_synapses_register_edges(const std::vector<std::pair<std::pair<int, size_t>, int>>& edges);

    [[nodiscard]] static MapSynapseDeletionRequests delete_synapses_exchange_requests(const PendingDeletionsV& pending_deletions);

    static void delete_synapses_process_requests(const MapSynapseDeletionRequests& synapse_deletion_requests_incoming, PendingDeletionsV& pending_deletions);

    [[nodiscard]] size_t delete_synapses_commit_deletions(const PendingDeletionsV& list);

    [[nodiscard]] size_t create_synapses();

    void create_synapses_update_octree();

    [[nodiscard]] MapSynapseCreationRequests create_synapses_find_targets();

    [[nodiscard]] static MapSynapseCreationRequests create_synapses_exchange_requests(const MapSynapseCreationRequests& synapse_creation_requests_outgoing);

    [[nodiscard]] std::pair<size_t, std::map<int, std::vector<char>>> create_synapses_process_requests(const MapSynapseCreationRequests& synapse_creation_requests_incoming);

    [[nodiscard]] static std::map<int, std::vector<char>> create_synapses_exchange_responses(const std::map<int, std::vector<char>>& synapse_creation_responses, const MapSynapseCreationRequests& synapse_creation_requests_outgoing);

    [[nodiscard]] size_t create_synapses_process_responses(const MapSynapseCreationRequests& synapse_creation_requests_outgoing, const std::map<int, std::vector<char>>& received_responses);

    static void print_pending_synapse_deletions(const PendingDeletionsV& list);

    size_t num_neurons = 0; // Local number of neurons

    std::shared_ptr<Partition> partition{};

    std::shared_ptr<Octree> global_tree{};
    std::shared_ptr<BarnesHut> barnes_hut_algorithm{};

    std::shared_ptr<NetworkGraph> network_graph{};

    std::unique_ptr<NeuronModel> neuron_model{};

    std::unique_ptr<Axons> axons{};
    std::unique_ptr<DendritesExc> dendrites_exc{};
    std::unique_ptr<DendritesInh> dendrites_inh{};

    std::vector<double> calcium{}; // Intracellular calcium concentration of every neuron

    std::vector<char> disable_flags{};

    std::unique_ptr<NeuronsExtraInfo> extra_info{ std::make_unique<NeuronsExtraInfo>() };
};
