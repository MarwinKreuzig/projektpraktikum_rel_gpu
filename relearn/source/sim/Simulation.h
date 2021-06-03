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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class NetworkGraph;
class NeuronModels;
class NeuronMonitor;
class NeuronToSubdomainAssignment;
class Neurons;
class Octree;
class Partition;
class SynapticElements;

class Simulation {
public:
    explicit Simulation(std::shared_ptr<Partition> partition);

    void register_neuron_monitor(size_t neuron_id);

    void set_acceptance_criterion_for_octree(double value);

    void set_neuron_models(std::unique_ptr<NeuronModels> nm);

    void set_axons(std::unique_ptr<SynapticElements> se);

    void set_dendrites_ex(std::unique_ptr<SynapticElements> se);

    void set_dendrites_in(std::unique_ptr<SynapticElements> se);

    void set_enable_interrupts(std::vector<std::pair<size_t, std::vector<size_t>>> interrupts);

    void set_disable_interrupts(std::vector<std::pair<size_t, std::vector<size_t>>> interrupts);

    void set_creation_interrupts(std::vector<std::pair<size_t, size_t>> interrupts);

    void place_random_neurons(size_t num_neurons, double frac_exc);

    void load_neurons_from_file(const std::string& path_to_positions, const std::optional<std::string>& optional_path_to_connections);

    void simulate(size_t number_steps, size_t step_monitor);

    void finalize() const;

    void increase_monitoring_capacity(size_t size);

    static std::vector<std::unique_ptr<NeuronModels>> get_models();

private:
    void construct_neurons();

    void initialize();

    void print_neuron_monitors();

    std::shared_ptr<Partition> partition;

    std::unique_ptr<NeuronToSubdomainAssignment> neuron_to_subdomain_assignment;

    std::unique_ptr<SynapticElements> axons;
    std::unique_ptr<SynapticElements> dendrites_ex;
    std::unique_ptr<SynapticElements> dendrites_in;

    std::unique_ptr<NeuronModels> neuron_models;
    std::shared_ptr<Neurons> neurons;

    std::shared_ptr<Octree> global_tree;

    std::shared_ptr<NetworkGraph> network_graph;

    std::vector<NeuronMonitor> monitors;

    std::vector<std::pair<size_t, std::vector<size_t>>> enable_interrupts;
    std::vector<std::pair<size_t, std::vector<size_t>>> disable_interrupts;
    std::vector<std::pair<size_t, size_t>> creation_interrupts;

    double accept_criterion{ 0.0 };

    int64_t total_synapse_creations = 0;
    int64_t total_synapse_deletions = 0;

    int64_t delta_synapse_creations = 0;
    int64_t delta_synapse_deletions = 0;
};
