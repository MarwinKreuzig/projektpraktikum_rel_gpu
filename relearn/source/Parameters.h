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

#include <iomanip>

class Parameters {
public:
	size_t num_neurons;           // Number of neurons
	double frac_neurons_exc;      // Fraction of DendriteType::EXCITATORY neurons [0.8 .. 0.9] (Should be 80-90% of all neurons)
	size_t simulation_time;       // Simulation time

	// Neuron model
	double x_0;                   // Background or resting activity
	double tau_x;                 // Decay time of firing rate in msec
	double k;                     // Proportionality factor for synapses in Hz
	double tau_C;                 // Decay time of calcium
	double beta;                  // Increase in calcium each time a neuron fires
	double refrac_time;           // Length of refractory period in msec. After an action potential a neuron cannot fire for this time
	int    h;                     // Precision for Euler integration

	// Synaptic elements
	double C_target;              // Desired calcium level (possible extension of the model: Give all neurons individual C_target values!)
	double eta_A;                 // Minimum level of calcium needed for axonal elements to grow
	double eta_D_ex;              // Minimum level of calcium needed for dendritic DendriteType::EXCITATORY elements to grow
	double eta_D_in;              // Minimum level of calcium needed for dendritic DendriteType::INHIBITORY elements to grow
	double nu;                    // Growth rate for synaptic elements in ms^-1. Needs to be much smaller than 1 to seperate activity and structural dynamics.
	double vacant_retract_ratio;  // Percentage of how many vacant synaptic elements should be deleted during each connectivity update

	// Connectivity
	double accept_criterion;      // Barnes-Hut acceptance criterion
	double sigma;                 // Probability parameter in MSP (dispersion). The higher sigma the more likely to form synapses with remote neurons
	bool naive_method;            // Consider all neurons as target neurons for synapse creation, regardless of whether dendrites are available or not
	size_t max_num_pending_vacant_axons;  // Maximum number of vacant axons which are considered at the same time for finding a target neuron

	// Input
	std::string file_with_neuron_positions;  // Name of the file containing neuron positions as input for the simulation
	std::string file_with_network;  // Name of the file containing neuron positions as input for the simulation

	// Output
	size_t num_log_files;         // Number of log files to create
	size_t log_start_neuron;      // Neuron id to start with for writing to the log

	// MPI RMA (remote memory access)
	size_t mpi_rma_mem_size;      // Memory size (Byte) that should be allocated with MPI_Alloc_mem

	// Random number seeds
	long int seed_octree;         // Random number seed of Octree
	long int seed_partition;      // Random number seed of Partition

	// Overload << operator for proper output
	friend std::ostream& operator<< (std::ostream& os, const Parameters& params) {
		using namespace std;

		os << "** PARAMETERS **\n\n";
		os << std::left << std::setw(column_width) << "num_neurons" << " : " << params.num_neurons << "\n";
		os << std::left << std::setw(column_width) << "frac_neurons_exc" << " : " << params.frac_neurons_exc << "\n";
		os << std::left << std::setw(column_width) << "simulation_time" << " : " << params.simulation_time << "\n";
		os << std::left << std::setw(column_width) << "x_0" << " : " << params.x_0 << "\n";
		os << std::left << std::setw(column_width) << "tau_x" << " : " << params.tau_x << "\n";
		os << std::left << std::setw(column_width) << "k" << " : " << params.k << "\n";
		os << std::left << std::setw(column_width) << "tau_C" << " : " << params.tau_C << "\n";
		os << std::left << std::setw(column_width) << "beta" << " : " << params.beta << "\n";
		os << std::left << std::setw(column_width) << "C_target" << " : " << params.C_target << "\n";
		os << std::left << std::setw(column_width) << "refrac_time" << " : " << params.refrac_time << "\n";
		os << std::left << std::setw(column_width) << "h" << " : " << params.h << "\n";
		os << std::left << std::setw(column_width) << "eta_A" << " : " << params.eta_A << "\n";
		os << std::left << std::setw(column_width) << "eta_D_ex" << " : " << params.eta_D_ex << "\n";
		os << std::left << std::setw(column_width) << "eta_D_in" << " : " << params.eta_D_in << "\n";
		os << std::left << std::setw(column_width) << "nu" << " : " << params.nu << "\n";
		os << std::left << std::setw(column_width) << "vacant_retract_ratio" << " : " << params.vacant_retract_ratio << "\n";
		os << std::left << std::setw(column_width) << "sigma" << " : " << params.sigma << "\n";
		os << std::left << std::setw(column_width) << "accept_criterion (BH)" << " : " << params.accept_criterion << "\n";
		os << std::left << std::setw(column_width) << "naive_method (BH)" << " : " << params.naive_method << "\n";
		os << std::left << std::setw(column_width) << "file_with_neuron_positions" << " : " << params.file_with_neuron_positions << "\n";
		os << std::left << std::setw(column_width) << "num_log_files" << " : " << params.num_log_files << "\n";
		os << std::left << std::setw(column_width) << "log_start_neuron" << " : " << params.log_start_neuron << "\n";
		os << std::left << std::setw(column_width) << "mpi_rma_mem_size (Byte)" << " : " << params.mpi_rma_mem_size << "\n";
		os << std::left << std::setw(column_width) << "max_num_pending_vacant_axons" << " : " << params.max_num_pending_vacant_axons << "\n";
		os << std::left << std::setw(column_width) << "seed_octree" << " : " << params.seed_octree << "\n";
		//os << std::left << std::setw(column_width) << "seed_partition" << " : " << params.seed_partition << "\n";
		os << std::left << std::setw(column_width) << "seed_partition" << " : " << "Local MPI rank" << "\n";

		return os;
	}

private:
	// Width of column containing parameter names
	static const int column_width = 28;
};
