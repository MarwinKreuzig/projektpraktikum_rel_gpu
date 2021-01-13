/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Simulation.h"

#include "Commons.h"
#include "LogFiles.h"
#include "MPIWrapper.h"
#include "NetworkGraph.h"
#include "NeuronIdMap.h"
#include "NeuronModels.h"
#include "NeuronMonitor.h"
#include "Neurons.h"
#include "NeuronToSubdomainAssignment.h"
#include "RelearnException.h"
#include "SubdomainFromFile.h"
#include "SubdomainFromNeuronDensity.h"
#include "Timers.h"

#include <fstream>
#include <sstream>


Simulation::Simulation(double accept_criterion) : parameters(std::make_unique<Parameters>()) {
	// Needed to avoid creating autapses
	if (accept_criterion > 0.5) {
		RelearnException::fail("Acceptance criterion must be smaller or equal to 0.5");
	}

	//parameters->frac_neurons_exc = 0.8;                          // CHANGE
	parameters->x_0 = 0.05;
	parameters->tau_x = 5.0;
	parameters->k = 0.03;
	parameters->tau_C = 10000; //5000;   //very old 60.0;
	parameters->beta = 0.001;  //very old 0.05;
	parameters->h = 10;
	parameters->refrac_time = 4.0;
	parameters->sigma = 750.0;
	parameters->max_num_pending_vacant_axons = 1000;
	parameters->accept_criterion = accept_criterion;
	parameters->naive_method = parameters->accept_criterion == 0.0;

	if (0 == MPIWrapper::my_rank) {
		std::cout << parameters << std::endl;
	}
}

void Simulation::registerNeuronMonitor(size_t neuron_id) {
	monitors.emplace_back(neuron_id);
}

void Simulation::setPartition(std::unique_ptr<Partition> part) {
	RelearnException::check(part != nullptr, "part should not be null");
	partition = std::move(part);
}

void Simulation::placeRandomNeurons(size_t num_neurons, double frac_exc) {
	neuron_to_subdomain_assignment = std::make_unique<SubdomainFromNeuronDensity>(num_neurons, frac_exc, 26);
	doStuffAndSuch();
	network_graph = std::make_shared<NetworkGraph>(neurons->get_num_neurons());
}

void Simulation::loadNeuronsFromFile(const std::string& path_to_positions) {
	neuron_to_subdomain_assignment = std::make_unique<SubdomainFromFile>(path_to_positions, *partition);
	doStuffAndSuch();
}

void Simulation::loadNeuronsFromFile(const std::string& path_to_positions, const std::string& path_to_connections) {
	loadNeuronsFromFile(path_to_positions);

	network_graph = std::make_shared<NetworkGraph>(neurons->get_num_neurons());
	network_graph->add_edges_from_file(path_to_connections, path_to_positions, *neuron_id_map, *partition);

	LogMessages::print_message_rank("Network graph created", 0);

	neurons->init_synaptic_elements(*network_graph);

	LogMessages::print_message_rank("Synaptic elements initialized \n", 0);

	neurons->print_neurons_overview_to_log_file_on_rank_0(0, Logs::get("neurons_overview"), *parameters);
	neurons->print_sums_of_synapses_and_elements_to_log_file_on_rank_0(0, Logs::get("sums"), *parameters, 0, 0);
}

void Simulation::simulate(size_t number_steps, size_t step_monitor) {
	GlobalTimers::timers.start(TimerRegion::SIMULATION_LOOP);

	/**
	* Simulation loop
	*/
	for (size_t step = 1; step <= number_steps; step++) {

		if (step % step_monitor == 0) {
			for (auto& mn : monitors) {
				mn.record_data();
			}

			NeuronMonitor::current_step++;
		}

		// Provide neuronal network to neuron models for one iteration step
		GlobalTimers::timers.start(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);
		neurons->update_electrical_activity(*network_graph);
		GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);

		// Calc how many synaptic elements grow/retract
		// Apply the change in number of elements during connectivity update
		GlobalTimers::timers.start(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);
		neurons->update_number_synaptic_elements_delta();
		GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);

		//if (0 == MPIWrapper::my_rank && step % 50 == 0) {
		//	std::cout << "** STATE AFTER: " << step << " of " << params.simulation_time
		//		<< " msec ** [" << Timers::wall_clock_time() << "]\n";
		//}

		// Update connectivity every 100 ms
		if (step % Constants::plasticity_update_step == 0) {
			size_t num_synapses_deleted = 0;
			size_t num_synapses_created = 0;

			if (0 == MPIWrapper::my_rank) {
				std::cout << "** UPDATE CONNECTIVITY AFTER: " << step << " of " << number_steps
					<< " msec ** [" << Timers::wall_clock_time() << "]\n";
			}

			GlobalTimers::timers.start(TimerRegion::UPDATE_CONNECTIVITY);

			neurons->update_connectivity(*global_tree, *network_graph, num_synapses_deleted, num_synapses_created);

			GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_CONNECTIVITY);

			// Get total number of synapses deleted and created
			std::array<uint64_t, 2> local_cnts = { static_cast<uint64_t>(num_synapses_deleted), static_cast<uint64_t>(num_synapses_created) };
			std::array<uint64_t, 2> global_cnts{};

			MPIWrapper::reduce(local_cnts, global_cnts, MPIWrapper::ReduceFunction::sum, 0, MPIWrapper::Scope::global);

			if (0 == MPIWrapper::my_rank) {
				total_synapse_deletions += global_cnts[0] / 2;
				total_synapse_creations += global_cnts[1] / 2;
			}

			if (global_cnts[0] != 0.0) {
				std::stringstream sstring; // For output generation
				sstring << "Sum (all processes) number synapses deleted: " << global_cnts[0] / 2;
				LogMessages::print_message_rank(sstring.str().c_str(), 0);
			}

			if (global_cnts[1] != 0.0) {
				std::stringstream sstring; // For output generation
				sstring << "Sum (all processes) number synapses created: " << global_cnts[1] / 2;
				LogMessages::print_message_rank(sstring.str().c_str(), 0);
			}

			neurons->print_sums_of_synapses_and_elements_to_log_file_on_rank_0(
				step, Logs::get("sums"), *parameters, num_synapses_deleted, num_synapses_created);

			std::cout << std::flush;
		}

		// Print details every 500 ms
		if (step % Constants::logfile_update_step == 0) {
			neurons->print_neurons_overview_to_log_file_on_rank_0(step, Logs::get("neurons_overview"), *parameters);
		}
	}

	// Stop timing simulation loop
	GlobalTimers::timers.stop_and_add(TimerRegion::SIMULATION_LOOP);

	printNeuronMonitors();

	neurons->print_positions_to_log_file(Logs::get("positions_rank_" + MPIWrapper::my_rank_str), *parameters, *neuron_id_map);
	neurons->print_network_graph_to_log_file(Logs::get("network_rank_" + MPIWrapper::my_rank_str), *network_graph,
		*parameters, *neuron_id_map);
}

void Simulation::finalize() {

	//neurons_in_subdomain->write_neurons_to_file("output_positions_" + MPIWrapper::my_rank_str + ".txt");
	//network_graph.write_synapses_to_file("output_edges_" + MPIWrapper::my_rank_str + ".txt", neuron_id_map, partition);
	if (0 == MPIWrapper::my_rank) {
		std::stringstream sstring; // For output generation
		sstring << "\n";
		sstring << "\n" << "Total creations: " << total_synapse_creations << "\n";
		sstring << "Total deletions: " << total_synapse_deletions<< "\n";
		sstring << "END: " << Timers::wall_clock_time() << "\n";
		LogMessages::print_message_rank(sstring.str().c_str(), 0);
	}
}

std::vector<std::unique_ptr<NeuronModels>> Simulation::getModels() {
	return NeuronModels::get_models();
}

void Simulation::doStuffAndSuch() {
	Neurons neurs = partition->get_local_neurons(*parameters, *neuron_to_subdomain_assignment);
	neurons = std::make_shared<Neurons>(std::move(neurs));

	NeuronMonitor::neurons_to_monitor = neurons;

	partition->print_my_subdomains_info_rank(0);
	partition->print_my_subdomains_info_rank(1);

	LogMessages::print_message_rank("Neurons created", 0);

	neuron_id_map = std::make_unique<NeuronIdMap>(neurons->get_num_neurons(),
		neurons->get_positions().get_x_dims(),
		neurons->get_positions().get_y_dims(),
		neurons->get_positions().get_z_dims());

	global_tree = std::make_unique<Octree>(*partition, *parameters);
	global_tree->set_no_free_in_destructor(); // This needs to be changed later, as it's cleaner to free the nodes at destruction

	// Insert my local (subdomain) trees into my global tree
	for (size_t i = 0; i < partition->get_my_num_subdomains(); i++) {
		Octree* local_tree = &partition->get_subdomain_tree(i);
		global_tree->insert_local_tree(local_tree);
	}

	LogMessages::print_message_rank("Neurons inserted into subdomains", 0);
	LogMessages::print_message_rank("Subdomains inserted into global tree", 0);
}

void Simulation::printNeuronMonitors() {
	for (auto& monitor : monitors) {
		std::ofstream outfile(std::to_string(monitor.get_target_id()) + ".csv", std::ios::trunc);
		outfile << std::setprecision(5);

		outfile.imbue(std::locale());

		outfile << "Step;Fired;Refrac;x;Ca;I_sync;axons;axons_connected;dendrites_exc;dendrites_exc_connected;dendrites_inh;dendrites_inh_connected";
		outfile << "\n";

		const auto& infos = monitor.get_informations();

		const char* const filler = ";";
		const auto width = 6;

		auto ctr = 0;

		for (const auto& info : infos) {
			outfile << ctr << filler;
			outfile << /*std::setw(width) <<*/ info.fired << filler;
			outfile << /*std::setw(width) <<*/ info.secondary << filler;
			outfile << /*std::setw(width) <<*/ info.x << filler;
			outfile << /*std::setw(width) <<*/ info.calcium << filler;
			outfile << /*std::setw(width) <<*/ info.I_sync << filler;
			outfile << /*std::setw(width) <<*/ info.axons << filler;
			outfile << /*std::setw(width) <<*/ info.axons_connected << filler;
			outfile << /*std::setw(width) <<*/ info.dendrites_exc << filler;
			outfile << /*std::setw(width) <<*/ info.dendrites_exc_connected << filler;
			outfile << /*std::setw(width) <<*/ info.dendrites_inh << filler;
			outfile << /*std::setw(width) <<*/ info.dendrites_inh_connected << "\n";

			ctr++;
		}

		outfile.flush();
		outfile.close();
	}
}
