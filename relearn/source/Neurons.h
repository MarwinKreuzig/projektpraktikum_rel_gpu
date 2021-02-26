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

#include "Config.h"
#include "ElementType.h"
#include "NeuronModels.h"
#include "RankNeuronId.h"
#include "RelearnException.h"
#include "SignalType.h"
#include "SynapseCreationRequests.h"
#include "SynapticElements.h"
#include "Vec3.h"

#include <array>
#include <memory>
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

class NeuronsExtraInfo {
    size_t size{ 0 };

    std::vector<std::string> area_names;
    std::vector<double> x_dims;
    std::vector<double> y_dims;
    std::vector<double> z_dims;

public:
    void init(size_t number_neurons) noexcept {
        RelearnException::check(size == 0, "NeuronsExtraInfo initialized two times");
        size = number_neurons;
    }

    void set_area_names(std::vector<std::string> names) {
        RelearnException::check(area_names.empty(), "Area names are not empty");
        RelearnException::check(size == names.size(), "Size does not match area names count");
        area_names = std::move(names);
    }

    void set_x_dims(std::vector<double> dims) {
        RelearnException::check(x_dims.empty(), "X dimensions are not empty");
        RelearnException::check(size == dims.size(), "Size does not match area names count");
        x_dims = std::move(dims);
    }

    void set_y_dims(std::vector<double> dims) {
        RelearnException::check(y_dims.empty(), "Y dimensions are not empty");
        RelearnException::check(size == dims.size(), "Size does not match area names count");
        y_dims = std::move(dims);
    }

    void set_z_dims(std::vector<double> dims) {
        RelearnException::check(z_dims.empty(), "Z dimensions are not empty");
        RelearnException::check(size == dims.size(), "Size does not match area names count");
        z_dims = std::move(dims);
    }

    [[nodiscard]] const std::vector<double>& get_x_dims() const noexcept {
        return x_dims;
    }

    [[nodiscard]] const std::vector<double>& get_y_dims() const noexcept {
        return y_dims;
    }

    [[nodiscard]] const std::vector<double>& get_z_dims() const noexcept {
        return z_dims;
    }

    [[nodiscard]] Vec3d get_position(size_t idx) const {
        RelearnException::check(idx < size, "Idx must be smaller than size in Positions");
        return Vec3d{ x_dims[idx], y_dims[idx], z_dims[idx] };
    }

    [[nodiscard]] const std::vector<std::string>& get_area_names() const noexcept {
        return area_names;
    }

    [[nodiscard]] double get_x(size_t neuron_id) const {
        RelearnException::check(neuron_id < x_dims.size(), "neuron_id must be smaller than size in NeuronsExtraInfo");
        return x_dims[neuron_id];
    }

    [[nodiscard]] double get_y(size_t neuron_id) const {
        RelearnException::check(neuron_id < y_dims.size(), "neuron_id must be smaller than size in NeuronsExtraInfo");
        return y_dims[neuron_id];
    }

    [[nodiscard]] double get_z(size_t neuron_id) const {
        RelearnException::check(neuron_id < z_dims.size(), "neuron_id must be smaller than size in NeuronsExtraInfo");
        return z_dims[neuron_id];
    }

    [[nodiscard]] const std::string& get_area_name(size_t neuron_id) const {
        RelearnException::check(neuron_id < area_names.size(), "neuron_id must be smaller than size in NeuronsExtraInfo");
        return area_names[neuron_id];
    }
};

class Neurons {
    friend class NeuronMonitor;

    /**
	* Type for list element used to store pending synapse deletion
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
        PendingSynapseDeletion() = default;

        PendingSynapseDeletion(const RankNeuronId& src, const RankNeuronId& tgt, const RankNeuronId& aff,
            ElementType elem, SignalType sign, unsigned int id)
            : src_neuron_id(src)
            , tgt_neuron_id(tgt)
            , affected_neuron_id(aff)
            , affected_element_type(elem)
            , signal_type(sign)
            , synapse_id(id) {
        }

        PendingSynapseDeletion(const PendingSynapseDeletion& other) = default;
        PendingSynapseDeletion(PendingSynapseDeletion&& other) = default;

        PendingSynapseDeletion& operator=(const PendingSynapseDeletion& other) = default;
        PendingSynapseDeletion& operator=(PendingSynapseDeletion&& other) = default;

        ~PendingSynapseDeletion() = default;

        [[nodiscard]] const RankNeuronId& get_src_neuron_id() const noexcept {
            return src_neuron_id;
        }

        [[nodiscard]] const RankNeuronId& get_tgt_neuron_id() const noexcept {
            return tgt_neuron_id;
        }

        [[nodiscard]] const RankNeuronId& get_affected_neuron_id() const noexcept {
            return affected_neuron_id;
        }

        [[nodiscard]] ElementType get_affected_element_type() const noexcept {
            return affected_element_type;
        }

        [[nodiscard]] SignalType get_signal_type() const noexcept {
            return signal_type;
        }

        [[nodiscard]] unsigned int get_synapse_id() const noexcept {
            return synapse_id;
        }

        [[nodiscard]] bool get_affected_element_already_deleted() const noexcept {
            return affected_element_already_deleted;
        }

        void set_affected_element_already_deleted() noexcept {
            affected_element_already_deleted = true;
        }

        [[nodiscard]] bool check_light_equality(const PendingSynapseDeletion& other) const {
            const bool src_neuron_id_eq = other.src_neuron_id == src_neuron_id;
            const bool tgt_neuron_id_eq = other.tgt_neuron_id == tgt_neuron_id;

            const bool id_eq = other.synapse_id == synapse_id;

            return src_neuron_id_eq && tgt_neuron_id_eq && id_eq;
        }

        [[nodiscard]] bool check_light_equality(const RankNeuronId& src, const RankNeuronId& tgt, unsigned int id) const {
            const bool src_neuron_id_eq = src == src_neuron_id;
            const bool tgt_neuron_id_eq = tgt == tgt_neuron_id;

            const bool id_eq = id == synapse_id;

            return src_neuron_id_eq && tgt_neuron_id_eq && id_eq;
        }
    };
    using PendingDeletionsV = std::vector<PendingSynapseDeletion>;

    /**
	 * Type for list element used to represent a synapse for synapse selection
	 */
    class Synapse {
        RankNeuronId rank_neuron_id;
        unsigned int synapse_id; // Id of the synapse. Used to distinguish multiple synapses between the same neuron pair
    public:
        Synapse(RankNeuronId rank_neuron_id, unsigned int synapse_id) noexcept
            : rank_neuron_id(rank_neuron_id)
            , synapse_id(synapse_id) {
        }

        [[nodiscard]] RankNeuronId get_rank_neuron_id() const noexcept {
            return rank_neuron_id;
        }

        [[nodiscard]] unsigned int get_synapse_id() const noexcept {
            return synapse_id;
        }
    };

    /**
	 * Type for synapse deletion requests which are used with MPI
	 */
    struct SynapseDeletionRequests {
        SynapseDeletionRequests() = default;

        [[nodiscard]] size_t size() const noexcept {
            return num_requests;
        }

        void resize(size_t size) {
            num_requests = size;
            requests.resize(Constants::num_items_per_request * size);
        }

        void append(const PendingSynapseDeletion& pending_deletion) {
            num_requests++;

            size_t affected_element_type_converted = pending_deletion.get_affected_element_type() == ElementType::AXON ? 0 : 1;
            size_t signal_type_converted = pending_deletion.get_signal_type() == SignalType::EXCITATORY ? 0 : 1;

            requests.push_back(pending_deletion.get_src_neuron_id().get_neuron_id());
            requests.push_back(pending_deletion.get_tgt_neuron_id().get_neuron_id());
            requests.push_back(pending_deletion.get_affected_neuron_id().get_neuron_id());
            requests.push_back(affected_element_type_converted);
            requests.push_back(signal_type_converted);
            requests.push_back(pending_deletion.get_synapse_id());
        }

        [[nodiscard]] std::array<size_t, Constants::num_items_per_request> get_request(size_t request_index) const noexcept {
            const size_t base_index = Constants::num_items_per_request * request_index;

            std::array<size_t, Constants::num_items_per_request> arr{};

            for (size_t i = 0; i < Constants::num_items_per_request; i++) {
                arr[i] = requests[base_index + i];
            }

            return arr;
        }

        [[nodiscard]] size_t get_source_neuron_id(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
            return requests[index];
        }

        [[nodiscard]] size_t get_target_neuron_id(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 1;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
            return requests[index];
        }

        [[nodiscard]] size_t get_affected_neuron_id(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 2;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
            return requests[index];
        }

        [[nodiscard]] ElementType get_affected_element_type(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 3;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
            const auto affected_element_type_converted = requests[index] == 0 ? ElementType::AXON : ElementType::DENDRITE;
            return affected_element_type_converted;
        }

        [[nodiscard]] SignalType get_signal_type(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 4;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
            const auto affected_element_type_converted = requests[index] == 0 ? SignalType::EXCITATORY : SignalType::INHIBITORY;
            return affected_element_type_converted;
        }

        [[nodiscard]] unsigned int get_synapse_id(size_t request_index) const {
            const auto index = Constants::num_items_per_request * request_index + 5;
            RelearnException::check(index < requests.size(), "Index is out of bounds");
            const auto synapse_id = static_cast<unsigned int>(requests[index]);
            return synapse_id;
        }

        // Get pointer to data
        [[nodiscard]] size_t* get_requests() noexcept {
            return requests.data();
        }

        [[nodiscard]] const size_t* get_requests() const noexcept {
            return requests.data();
        }

        [[nodiscard]] size_t get_requests_size_in_bytes() const noexcept {
            return requests.size() * sizeof(size_t);
        }

    private:
        size_t num_requests{ 0 }; // Number of synapse deletion requests
        std::vector<size_t> requests; // Each request to delete a synapse is a 6-tuple:
            // (src_neuron_id, tgt_neuron_id, affected_neuron_id, affected_element_type, signal_type, synapse_id)
            // That is why requests.size() == 6*num_requests
            // Note, a more memory-efficient implementation would use a smaller data type (not size_t)
            // for affected_element_type, signal_type.
            // This vector is used as MPI communication buffer
    };

    struct StatisticalMeasures {
        double min;
        double max;
        double avg;
        double var;
        double std;
    };

public:
    /**
	 * Map of (MPI rank; SynapseDeletionRequests)
	 * The MPI rank specifies the corresponding process
	 */
    using MapSynapseDeletionRequests = std::map<int, SynapseDeletionRequests>;

    Neurons(const Partition& partition, std::unique_ptr<NeuronModels> model)
        : Neurons(partition, std::move(model),
            std::make_unique<Axons>(ElementType::AXON, SynapticElements::default_eta_Axons),
            std::make_unique<DendritesExc>(ElementType::DENDRITE, SynapticElements::default_eta_Dendrites_exc),
            std::make_unique<DendritesInh>(ElementType::DENDRITE, SynapticElements::default_eta_Dendrites_inh)) { }

    Neurons(const Partition& partition,
        std::unique_ptr<NeuronModels> model,
        std::unique_ptr<Axons> axons_ptr,
        std::unique_ptr<DendritesExc> dend_ex_ptr,
        std::unique_ptr<DendritesInh> dend_in_ptr)
        : partition(&partition)
        , neuron_model(std::move(model))
        , axons(std::move(*axons_ptr))
        , dendrites_exc(std::move(*dend_ex_ptr))
        , dendrites_inh(std::move(*dend_in_ptr)) {
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

    void set_network_graph(std::shared_ptr<NetworkGraph> network) {
        network_graph = std::move(network);
    }

    [[nodiscard]] std::vector<ModelParameter> get_parameter(ElementType element_type, SignalType signal_type) {
        if (element_type == ElementType::AXON) {
            return axons.get_parameter();
        }

        if (signal_type == SignalType::EXCITATORY) {
            return dendrites_exc.get_parameter();
        }

        return dendrites_inh.get_parameter();
    }

    void set_model(std::unique_ptr<NeuronModels> model) noexcept {
        neuron_model = std::move(model);
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
        axons.set_signal_types(std::move(signal_types));
    }

    [[nodiscard]] const Axons& get_axons() const noexcept {
        return axons;
    }

    [[nodiscard]] const DendritesExc& get_dendrites_exc() const noexcept {
        return dendrites_exc;
    }

    [[nodiscard]] const DendritesInh& get_dendrites_inh() const noexcept {
        return dendrites_inh;
    }

    void init_synaptic_elements();

    void update_electrical_activity();

    void update_number_synaptic_elements_delta() noexcept {
        axons.update_number_elements_delta(calcium);
        dendrites_exc.update_number_elements_delta(calcium);
        dendrites_inh.update_number_elements_delta(calcium);
    }

    [[nodiscard]] std::tuple<size_t, size_t> update_connectivity();

    void print_sums_of_synapses_and_elements_to_log_file_on_rank_0(size_t step, size_t sum_synapses_deleted, size_t sum_synapses_created);

    // Print global information about all neurons at rank 0
    void print_neurons_overview_to_log_file_on_rank_0(size_t step);

    void print_network_graph_to_log_file();

    void print_positions_to_log_file();

    void print();

    void print_info_for_barnes_hut();

    void debug_check_counts();

private:
    void update_calcium();

    [[nodiscard]] StatisticalMeasures global_statistics(const std::vector<double>& local_values, [[maybe_unused]] size_t num_local_values, size_t total_num_values, int root) const;

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

    const Partition* partition;

    std::shared_ptr<Octree> global_tree;
    std::shared_ptr<NetworkGraph> network_graph;

    std::unique_ptr<NeuronModels> neuron_model;

    Axons axons;
    DendritesExc dendrites_exc;
    DendritesInh dendrites_inh;

    std::vector<double> calcium; // Intracellular calcium concentration of every neuron

    std::unique_ptr<NeuronsExtraInfo> extra_info{ std::make_unique<NeuronsExtraInfo>() };
};
