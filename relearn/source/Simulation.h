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

#include <memory>
#include <string>
#include <utility>
#include <vector>

class NetworkGraph;
class NeuronIdMap;
class NeuronModels;
class NeuronMonitor;
class Neurons;
class NeuronToSubdomainAssignment;
class Octree;
class Parameters;
class Partition;

class Simulation {
public:
	Simulation(double accept_criterion, std::shared_ptr<Partition> partition);

	void registerNeuronMonitor(size_t neuron_id);

	void placeRandomNeurons(size_t num_neurons, double frac_exc);

	void loadNeuronsFromFile(const std::string& path_to_positions);

	void loadNeuronsFromFile(const std::string& path_to_positions, const std::string& path_to_connections);

	void simulate(size_t number_steps = 6000000, size_t step_monitor = 100);

	void finalize();

	static std::vector<std::unique_ptr<NeuronModels>> getModels();

private:
	void doStuffAndSuch();

	void printNeuronMonitors();

	std::unique_ptr<Parameters> parameters;
	std::shared_ptr<Partition> partition;

	std::unique_ptr<NeuronToSubdomainAssignment> neuron_to_subdomain_assignment;

	std::shared_ptr<Neurons> neurons;
	std::unique_ptr<NeuronIdMap> neuron_id_map;

	std::unique_ptr<Octree> global_tree;

	std::shared_ptr<NetworkGraph> network_graph;

	std::vector<NeuronMonitor> monitors;

	size_t total_synapse_creations = 0;
	size_t total_synapse_deletions = 0;
};
