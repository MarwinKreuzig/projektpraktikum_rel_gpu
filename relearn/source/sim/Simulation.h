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

#include "../algorithm/Types.h"
#include "../util/StatisticalMeasures.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class Algorithm;
class NetworkGraph;
class NeuronModel;
class NeuronMonitor;
class NeuronToSubdomainAssignment;
class Neurons;
class Octree;
class Partition;
class SynapticElements;

/**
 * This class encapsulates all necessary attributes of a simulation.
 * The neuron model and the synaptic elements must be set before loading the neurons,
 * which in turn must happen before calling simulate.
 */
class Simulation {
public:
    /**
     * @brief Constructs a new object with the given partition.
     * @param partition The partition for this simulation
     */
    explicit Simulation(std::shared_ptr<Partition> partition);

    /**
     * @brief Registers a monitor for the given neuron id.
     *      Does not check for duplicates, etc.
     * @param neuron_id The local neuron id that should be monitored
     */
    void register_neuron_monitor(size_t neuron_id);

    /**
     * @brief Sets the acceptance criterion (theta) for the barnes hut algorithm
     * @param value The acceptance criterion (theta) in [0.0, BarnesHut::max_theta]
     * @exception Throws a RelearnException if value is not from [0.0, BarnesHut::max_theta]
     */
    void set_acceptance_criterion_for_barnes_hut(double value);

    /**
     * @brief Sets the neuron model used for the simulation
     * @param nm The neuron model
     */
    void set_neuron_model(std::unique_ptr<NeuronModel>&& nm) noexcept;

    /**
     * @brief Sets the synaptic elements model for the axons
     * @param se The synaptic elements model
     */
    void set_axons(std::unique_ptr<SynapticElements>&& se) noexcept;

    /**
     * @brief Sets the synaptic elements model for the excitatory dendrites
     * @param se The synaptic elements model
     */
    void set_dendrites_ex(std::unique_ptr<SynapticElements>&& se) noexcept;

    /**
     * @brief Sets the synaptic elements model for the inhibitory dendrites
     * @param se The synaptic elements model
     */
    void set_dendrites_in(std::unique_ptr<SynapticElements>&& se) noexcept;

    /**
     * @brief Sets the enable interrupts during the simulation.
     *      An enable interrupt is a pair of (1) the simulation set (2) all local ids that should be enabled
     * @param interrupts The enable interrupts
     */
    void set_enable_interrupts(std::vector<std::pair<size_t, std::vector<size_t>>> interrupts);

    /**
     * @brief Sets the disable interrupts during the simulation.
     *      An disable interrupt is a pair of (1) the simulation set (2) all local ids that should be disabled
     * @param interrupts The disable interrupts
     */
    void set_disable_interrupts(std::vector<std::pair<size_t, std::vector<size_t>>> interrupts);

    /**
     * @brief Sets the creation interrupts during the simulation.
     *      An creation interrupt is a pair of (1) the simulation set (2) the number of neurons to create
     * @param interrupts The creation interrupts
     */
    void set_creation_interrupts(std::vector<std::pair<size_t, size_t>> interrupts) noexcept;

    /**
     * @brief Sets the algorithm that is used for finding target neurons.
     * @param algorithm The desired algorithm
     */
    void set_algorithm(AlgorithmEnum algorithm) noexcept;

    /**
     * @brief Places the requested number of neurons with the requested fraction of excitatory neurons.
     *      The division to the MPI ranks is done with SubdomainFromNeuronDensity
     * @param num_neurons The number of neurons to place globally
     * @param frac_exc The fraction of excitatory neurons, must be in [0.0, 1.0]
     */
    void place_random_neurons(size_t num_neurons, double frac_exc);

    /**
     * @brief Places all neurons from a file and optionally adds the specified synapses
     * @param path_to_positions The path to the neurons file
     * @param optional_path_to_connections The path to the synapses file, can be empty to indicate no initial synapses
     */
    void load_neurons_from_file(const std::string& path_to_positions, const std::optional<std::string>& optional_path_to_connections);

    /**
     * @brief Simulates the neurons for the requested number of steps. Every step_monitor-th step, records all neuron monitors
     * @param number_steps The number of simulation steps
     * @param step_monitor The step size of the monitors, must be > 0
     * @exception Throws a RelearnException if step_monitor == 0
     */
    void simulate(size_t number_steps, size_t step_monitor);

    /**
     * @brief Finalizes the simulation in the sense that it prints the final statistics.
     *      Does not perform any "irreversible" steps and does not finalize MPI.
     *      All MPI processes must call finalize
     */
    void finalize() const;

    /**
     * @brief Increases the capacity of each registered neuron monitor by the requested size
     * @param size The size by which to increase the monitors
     */
    void increase_monitoring_capacity(size_t size);

    /**
	 * @brief Returns a vector with an std::unique_ptr for each class inherited from NeuronModels which can be cloned
     * @return A vector with all inherited classes
	 */
    static std::vector<std::unique_ptr<NeuronModel>> get_models();

    std::shared_ptr<Neurons> get_neurons() {
        return neurons;
    }

    std::shared_ptr<NetworkGraph> get_network_graph() {
        return network_graph;
    }

    std::shared_ptr<std::vector<NeuronMonitor>> get_monitors() {
        return monitors;
    }

    void snapshot_monitors();

    void measure_calcium();

    void measure_activity();

    void save_network_graph(size_t current_steps);

    const std::vector<StatisticalMeasures>& get_calcium_statistics() {
        return calcium_statistics;
    }

    const std::vector<StatisticalMeasures>& get_activity_statistics() {
        return activity_statistics;
    }

private:
    void construct_neurons();

    void initialize();

    void print_neuron_monitors();

    std::shared_ptr<Partition> partition{};

    std::unique_ptr<NeuronToSubdomainAssignment> neuron_to_subdomain_assignment{};

    std::unique_ptr<SynapticElements> axons{};
    std::unique_ptr<SynapticElements> dendrites_ex{};
    std::unique_ptr<SynapticElements> dendrites_in{};

    std::unique_ptr<NeuronModel> neuron_models{};
    std::shared_ptr<Neurons> neurons{};

    std::shared_ptr<Algorithm> algorithm{};
    std::shared_ptr<Octree> global_tree{};

    std::shared_ptr<NetworkGraph> network_graph{};

    std::shared_ptr<std::vector<NeuronMonitor>> monitors{};

    std::vector<std::pair<size_t, std::vector<size_t>>> enable_interrupts{};
    std::vector<std::pair<size_t, std::vector<size_t>>> disable_interrupts{};
    std::vector<std::pair<size_t, size_t>> creation_interrupts{};

    std::vector<StatisticalMeasures> calcium_statistics{};
    std::vector<StatisticalMeasures> activity_statistics{};

    double accept_criterion{ 0.0 };

    AlgorithmEnum algorithm_enum{};

    int64_t total_synapse_creations{ 0 };
    int64_t total_synapse_deletions{ 0 };

    int64_t delta_synapse_creations{ 0 };
    int64_t delta_synapse_deletions{ 0 };
};
