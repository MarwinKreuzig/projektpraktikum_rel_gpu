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

#include "Config.h"
#include "ElementType.h"
#include "NeuronsExtraInfo.h"
#include "SignalType.h"
#include "UpdateStatus.h"
#include "algorithm/Algorithm.h"
#include "helper/RankNeuronId.h"
#include "helper/SynapseCreationRequests.h"
#include "helper/SynapseDeletionRequests.h"
#include "models/NeuronModels.h"
#include "models/SynapticElements.h"
#include "mpi/CommunicationMap.h"
#include "util/RelearnException.h"
#include "util/StatisticalMeasures.h"
#include "util/Vec3.h"

#include <array>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <tuple>
#include <vector>

class NetworkGraph;
class NeuronMonitor;
class Octree;
class Partition;

/**
 * This class gathers all information for the neurons and provides the primary interface for the simulation
 */
class Neurons {
    friend class NeuronMonitor;

public:
    using Axons = SynapticElements;
    using DendritesExcitatory = SynapticElements;
    using DendritesInhibitory = SynapticElements;

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
        std::shared_ptr<Axons> axons_ptr,
        std::shared_ptr<DendritesExcitatory> dend_ex_ptr,
        std::shared_ptr<DendritesInhibitory> dend_in_ptr)
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
     * @brief Returns the model parameters for the specified synaptic elements
     * @param element_type The element type
     * @param signal_type The signal type, only relevant if element_type == dendrites
     * @return The model parameters for the specified synaptic elements
     */
    [[nodiscard]] std::vector<ModelParameter> get_parameter(const ElementType element_type, const SignalType signal_type) {
        if (element_type == ElementType::Axon) {
            return axons->get_parameter();
        }

        if (signal_type == SignalType::Excitatory) {
            return dendrites_exc->get_parameter();
        }

        return dendrites_inh->get_parameter();
    }

    /**
     * @brief Returns the number of neurons in this object
     * @return The number of neurons in this object
     */
    [[nodiscard]] size_t get_number_neurons() const noexcept {
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
    [[nodiscard]] const std::unique_ptr<NeuronsExtraInfo>& get_extra_info() const noexcept {
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
    [[nodiscard]] std::tuple<uint64_t, uint64_t, uint64_t> update_connectivity();

    /**
     * @brief Calculates the number vacant axons and dendrites (excitatory, inhibitory) and prints them to LogFiles::EventType::Sums
     *      Performs communication with MPI
     * @param step The current simulation step
     * @param sum_synapses_deleted The number of deleted synapses (locally)
     * @param sum_synapses_created The number of created synapses (locally)
     */
    void print_sums_of_synapses_and_elements_to_log_file_on_rank_0(size_t step, uint64_t sum_axon_deleted, uint64_t sum_dendrites_deleted, uint64_t sum_synapses_created);

    /**
     * @brief Prints the overview of the neurons to LogFiles::EventType::NeuronsOverview
     *      Performs communication with MPI
     * @param step The current simulation step
     */
    void print_neurons_overview_to_log_file_on_rank_0(size_t step) const;

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
    [[nodiscard]] StatisticalMeasures get_statistics(NeuronAttribute attribute) const;

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

    [[nodiscard]] std::pair<uint64_t, uint64_t> delete_synapses();

    [[nodiscard]] CommunicationMap<SynapseDeletionRequest> delete_synapses_find_synapses(const SynapticElements& synaptic_elements, const std::pair<unsigned int, std::vector<unsigned int>>& to_delete);

    [[nodiscard]] std::vector<RankNeuronId> delete_synapses_find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, unsigned int num_synapses_to_delete);

    [[nodiscard]] size_t delete_synapses_commit_deletions(const CommunicationMap<SynapseDeletionRequest>& list);

    [[nodiscard]] size_t create_synapses();

    size_t number_neurons = 0;

    std::shared_ptr<Partition> partition{};

    std::shared_ptr<Octree> global_tree{};
    std::shared_ptr<Algorithm> algorithm{};

    std::shared_ptr<NetworkGraph> network_graph{};

    std::unique_ptr<NeuronModel> neuron_model{};

    std::shared_ptr<Axons> axons{};
    std::shared_ptr<DendritesExcitatory> dendrites_exc{};
    std::shared_ptr<DendritesInhibitory> dendrites_inh{};

    std::vector<double> target_calcium{};
    std::vector<double> calcium{}; // Intracellular calcium concentration of every neuron

    std::vector<UpdateStatus> disable_flags{};

    std::unique_ptr<NeuronsExtraInfo> extra_info{ std::make_unique<NeuronsExtraInfo>() };
};
