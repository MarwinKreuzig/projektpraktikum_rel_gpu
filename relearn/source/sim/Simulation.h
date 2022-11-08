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
#include "Types.h"
#include "algorithm/AlgorithmEnum.h"
#include "util/StatisticalMeasures.h"
#include "neurons/helper/AreaMonitor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class Algorithm;
class AreaMonitor;
class CalciumCalculator;
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
 * The neuron model, the synaptic elements, and the subdomain assignment must be set before calling initialize,
 * which in turn must happen before calling simulate.
 */
class Simulation {
public:
    using step_type = RelearnTypes::step_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;

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
    void register_neuron_monitor(const NeuronID& neuron_id);

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
     * @brief Sets the calcium calculator used for the simulation
     * @param calculator The calcium calculator
     */
    void set_calcium_calculator(std::unique_ptr<CalciumCalculator>&& calculator) noexcept;

    /**
     * @brief Sets the synaptic elements model for the axons
     * @param se The synaptic elements model
     */
    void set_axons(std::shared_ptr<SynapticElements>&& se) noexcept;

    /**
     * @brief Sets the synaptic elements model for the excitatory dendrites
     * @param se The synaptic elements model
     */
    void set_dendrites_ex(std::shared_ptr<SynapticElements>&& se) noexcept;

    /**
     * @brief Sets the synaptic elements model for the inhibitory dendrites
     * @param se The synaptic elements model
     */
    void set_dendrites_in(std::shared_ptr<SynapticElements>&& se) noexcept;

    /**
     * @brief Sets the enable interrupts during the simulation.
     *      An enable interrupt is a pair of (1) the simulation set (2) all local ids that should be enabled
     * @param interrupts The enable interrupts
     */
    void set_enable_interrupts(std::vector<std::pair<step_type, std::vector<NeuronID>>> interrupts);

    /**
     * @brief Sets the disable interrupts during the simulation.
     *      An disable interrupt is a pair of (1) the simulation set (2) all local ids that should be disabled
     * @param interrupts The disable interrupts
     */
    void set_disable_interrupts(std::vector<std::pair<step_type, std::vector<NeuronID>>> interrupts);

    /**
     * @brief Sets the creation interrupts during the simulation.
     *      An creation interrupt is a pair of (1) the simulation set (2) the number of neurons to create
     * @param interrupts The creation interrupts
     */
    void set_creation_interrupts(std::vector<std::pair<step_type, number_neurons_type>> interrupts) noexcept;

    /**
     * @brief Sets the algorithm that is used for finding target neurons.
     * @param algorithm The desired algorithm
     */
    void set_algorithm(AlgorithmEnum algorithm) noexcept;

    /**
     * @brief Sets the subdomain assignment that determines how the neurons are loaded.
     * @param subdomain_assignment The desired subdomain assignment
     */
    void set_subdomain_assignment(std::unique_ptr<NeuronToSubdomainAssignment>&& subdomain_assignment) noexcept;

    void set_static_neurons(std::vector<NeuronID> static_neurons);

    /**
     * @brief Initializes the simulation and all other objects.
     * @exception Throws a RelearnException if one object is missing or something went wrong otherwise
     */
    void initialize();

    /**
     * @brief Simulates the neurons for the requested number of steps. Every step_monitor-th step, records all neuron monitors
     * @param number_steps The number of simulation steps, must be > 0
     * @exception Throws a RelearnException if number_steps == 0
     */
    void simulate(step_type number_steps);

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

    /**
     * @brief Returns an std::shared_ptr to the neurons object
     * @return The neurons object
     */
    std::shared_ptr<Neurons> get_neurons() noexcept {
        return neurons;
    }

    /**
     * @brief Returns an std::shared_ptr to the network graph
     * @return The network graph
     */
    std::shared_ptr<NetworkGraph> get_network_graph() noexcept {
        return network_graph_plastic;
    }

    /**
     * @brief Returns an std::shared_ptr to all neuron monitors
     * @return All neuron monitors
     */
    std::shared_ptr<std::vector<NeuronMonitor>> get_monitors() noexcept {
        return monitors;
    }

    /**
     * @brief Adds the statistics for the global statistics overview
     *      Does nothing if the statistics has been added before
     * @param neuron_attribute_to_observe The statistics that should be observed
     */
    void add_statistical_overview(NeuronAttribute neuron_attribute_to_observe) noexcept {
        if (statistics.find(neuron_attribute_to_observe) == statistics.end()) {
            statistics.emplace(neuron_attribute_to_observe, std::vector<StatisticalMeasures>{});
        }
    }

    /**
     * @brief Returns the statistics observered for the requested attribute
     * @param neuron_attribute_to_observe The statistics
     * @exception Throws a RelearnException if the statistics have not been observed
     * @return A constants reference to the statistics
     */
    [[nodiscard]] const std::vector<StatisticalMeasures>& get_statistics(NeuronAttribute neuron_attribute_to_observe) const {
        if (statistics.find(neuron_attribute_to_observe) == statistics.end()) {
            RelearnException::fail("Simulation::get_statistics: The attribute was not observed: {}", static_cast<int>(neuron_attribute_to_observe));
        }

        const auto& return_value = statistics.at(neuron_attribute_to_observe);

        return return_value;
    }

    /**
     * @brief Records one snapshot of each neuron monitor
     */
    void snapshot_monitors();

    /**
     * @brief Prints the current network graph with the attached number of steps to the file
     * @param current_steps The current number of steps that will appear in the file name
     */
    void save_network_graph(step_type current_steps);

private:
    void print_neuron_monitors();

    std::shared_ptr<Partition> partition{};

    std::unique_ptr<NeuronToSubdomainAssignment> neuron_to_subdomain_assignment{};

    std::shared_ptr<SynapticElements> axons{};
    std::shared_ptr<SynapticElements> dendrites_ex{};
    std::shared_ptr<SynapticElements> dendrites_in{};

    std::vector<NeuronID> static_neurons{};

    std::unique_ptr<NeuronModel> neuron_models{};
    std::unique_ptr<CalciumCalculator> calcium_calculator{};
    std::shared_ptr<Neurons> neurons{};

    std::shared_ptr<Algorithm> algorithm{};
    std::shared_ptr<Octree> global_tree{};

    std::shared_ptr<NetworkGraph> network_graph_plastic{};
    std::shared_ptr<NetworkGraph> network_graph_static{};

    std::shared_ptr<std::vector<NeuronMonitor>> monitors{};
    std::shared_ptr<std::vector<AreaMonitor>> area_monitors{};

    std::vector<std::pair<step_type, std::vector<NeuronID>>> enable_interrupts{};
    std::vector<std::pair<step_type, std::vector<NeuronID>>> disable_interrupts{};
    std::vector<std::pair<step_type, number_neurons_type>> creation_interrupts{};

    std::map<NeuronAttribute, std::vector<StatisticalMeasures>> statistics{};

    std::function<double(int, NeuronID::value_type)> target_calcium_calculator{};
    std::function<double(int, NeuronID::value_type)> initial_calcium_initiator{};

    double accept_criterion{ 0.0 };

    AlgorithmEnum algorithm_enum{};

    int64_t total_synapse_creations{ 0 };
    int64_t total_synapse_deletions{ 0 };

    int64_t delta_synapse_creations{ 0 };
    int64_t delta_synapse_deletions{ 0 };

    step_type step{ 1 };
};
