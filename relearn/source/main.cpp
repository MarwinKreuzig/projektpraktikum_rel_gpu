/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Commons.h"
#include "LogFiles.h"
#include "LogMessages.h"
#include "MPIWrapper.h"
#include "MPI_RMA_MemAllocator.h"
#include "NetworkGraph.h"
#include "NeuronIdMap.h"
#include "NeuronModels.h"
#include "NeuronMonitor.h"
#include "NeuronToSubdomainAssignment.h"
#include "Neurons.h"
#include "Octree.h"
#include "Parameters.h"
#include "Partition.h"
#include "RelearnException.h"
#include "Simulation.h"
#include "SubdomainFromFile.h"
#include "SubdomainFromNeuronDensity.h"
#include "SynapticElements.h"
#include "Timers.h"
#include "Utility.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <array>
#include <bitset>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>


void setDefaultParameters(Parameters* params) /*noexcept*/ {
	RelearnException::check(params != nullptr);

	params->frac_neurons_exc = 0.8;                          // CHANGE
	params->x_0 = 0.05;
	params->tau_x = 5.0;
	params->k = 0.03;
	params->tau_C = 10000; //5000;   //very old 60.0;
	params->beta = 0.001;  //very old 0.05;
	params->h = 10;
	params->C_target = 0.5; // gold 0.5;
	params->refrac_time = 4.0;
	params->eta_A = 0; //0.4; // gold 0.0;
	params->eta_D_ex = 0; //0.1, // gold 0.0;
	params->eta_D_in = 0.0;
	params->nu = 1e-4; // gold 1e-5; // element growth rate
	params->vacant_retract_ratio = 0;
	params->sigma = 750.0;
	params->num_log_files = 9;  // NOT USED
	params->log_start_neuron = 0;  // NOT USED
	params->mpi_rma_mem_size = 300 * 1024 * 1024;  // 300 MB
	params->max_num_pending_vacant_axons = 1000;
	// params->seed_partition (No global parameter. Every process uses a different seed, its rank. See below)
}

void setSpecificParameters(Parameters* params, const std::vector<std::string>& arguments) {
	RelearnException::check(params != nullptr);

	double accept_criterion = 0.0;
	bool naive_method = false;

	// Parameter equal to "naive"
	if (arguments[1] == "naive") {
		naive_method = true;
	}
	else {
		accept_criterion = std::stod(arguments[1], nullptr);
	}

	params->simulation_time = stoull(arguments[4], nullptr, 10);  //6000000;
	params->num_neurons = stoull(arguments[2], nullptr, 10);        // CHANGE
	params->accept_criterion = accept_criterion;             // CHANGE
	params->naive_method = naive_method;                 // CHANGE
	params->file_with_neuron_positions = (arguments.size() >= 6) ? arguments[5] : "";
	params->file_with_network = (arguments.size() >= 7) ? arguments[6] : "";
	params->seed_octree = stol(arguments[3], nullptr, 10);

	/**
	* Parameter sanity checks
	*/

	// Needed to avoid creating autapses
	if (!(params->accept_criterion <= 0.5)) {
		RelearnException::fail("Acceptance criterion must be smaller or equal to 0.5");
	}

	// Number of ranks must be 2^n so that
	// the connectivity update works correctly
	const std::bitset<sizeof(int) * 8> bitset_num_ranks(MPIWrapper::num_ranks);
	if (1 != bitset_num_ranks.count() && (0 == MPIWrapper::my_rank)) {
		RelearnException::fail("Number of ranks must be of the form 2^n");
	}
}

void printTimers() {
	/**
	 * Print timers and memory usage
	 */
	RelearnException::check(3 * TimerRegion::NUM_TIMER_REGIONS == 69);

	std::array<double, 69> timers_local{};

	for (int i = 0; i < TimerRegion::NUM_TIMER_REGIONS; ++i) {
		const double elapsed = GlobalTimers::timers.get_elapsed(i);

		for (int j = 0; j < 3; ++j) {
			timers_local[3 * i + j] = elapsed;
		}
	}

	std::array<double, 69> timers_global{};

	MPIWrapper::reduce(timers_local, timers_global, MPIWrapper::ReduceFunction::minsummax, 0, MPIWrapper::Scope::global);

#ifndef NDEBUG
	std::stringstream sstring;
	// Check result of MPI_Reduce
	for (int i = 0; i < 3 * TimerRegion::NUM_TIMER_REGIONS; i++) {
		sstring << timers_global[i] << " ";
	}
	LogMessages::print_message_rank(sstring.str().c_str(), MPIWrapper::my_rank);
#endif

	// Divide second entry of (min, sum, max), i.e., sum, by the number of ranks
	// so that sum becomes average
	for (int i = 0; i < TimerRegion::NUM_TIMER_REGIONS; i++) {
		timers_global[3 * i + 1] /= MPIWrapper::num_ranks;
	}

	if (0 == MPIWrapper::my_rank) {
		// Set precision for aligned double output
		const auto old_precision = std::cout.precision();
		std::cout.precision(6);

		std::cout << "\n======== TIMERS GLOBAL OVER ALL RANKS ========" << std::endl;
		std::cout << "                                                (" << std::setw(Constants::print_width) << "    min" << " | " << std::setw(Constants::print_width) << "    avg" << " | " << std::setw(Constants::print_width) << "    max" << ") sec." << std::endl;
		std::cout << "TIMERS: main()" << std::endl;
		std::cout << "  Initialization                               : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INITIALIZATION] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INITIALIZATION + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INITIALIZATION + 2] << std::endl;
		std::cout << "  Simulation loop                              : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::SIMULATION_LOOP] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::SIMULATION_LOOP + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::SIMULATION_LOOP + 2] << std::endl;
		std::cout << "    Update electrical activity                 : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_ELECTRICAL_ACTIVITY] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_ELECTRICAL_ACTIVITY + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_ELECTRICAL_ACTIVITY + 2] << std::endl;
		std::cout << "      Barrier 1                                : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::BARRIER_1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::BARRIER_1 + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::BARRIER_1 + 2] << std::endl;
		std::cout << "      Prepare sending spikes                   : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_SENDING_SPIKES] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_SENDING_SPIKES + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_SENDING_SPIKES + 2] << std::endl;
		std::cout << "      Prepare num neuron ids                   : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_NUM_NEURON_IDS] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_NUM_NEURON_IDS + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::PREPARE_NUM_NEURON_IDS + 2] << std::endl;
		std::cout << "      Barrier 2                                : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::BARRIER_2] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::BARRIER_2 + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::BARRIER_2 + 2] << std::endl;
		std::cout << "      All to all                               : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALL_TO_ALL] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALL_TO_ALL + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALL_TO_ALL + 2] << std::endl;
		std::cout << "      Alloc mem for neuron ids                 : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALLOC_MEM_FOR_NEURON_IDS] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALLOC_MEM_FOR_NEURON_IDS + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::ALLOC_MEM_FOR_NEURON_IDS + 2] << std::endl;
		std::cout << "      Barrier 3                                : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::BARRIER_3] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::BARRIER_3 + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::BARRIER_3 + 2] << std::endl;
		std::cout << "      Exchange neuron ids                      : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_NEURON_IDS] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_NEURON_IDS + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_NEURON_IDS + 2] << std::endl;
		std::cout << "      Calculate synaptic input                 : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_INPUT] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_INPUT + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_INPUT + 2] << std::endl;
		std::cout << "      Calculate activity                       : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_ACTIVITY] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_ACTIVITY + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CALC_ACTIVITY + 2] << std::endl;
		std::cout << "    Update #synaptic elements delta            : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA + 2] << std::endl;
		std::cout << "    Connectivity update                        : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_CONNECTIVITY] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_CONNECTIVITY + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_CONNECTIVITY + 2] << std::endl;
		std::cout << "      Update #synaptic elements + del synapses : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES + 2] << std::endl;
		std::cout << "      Update local trees                       : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_LOCAL_TREES] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_LOCAL_TREES + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_LOCAL_TREES + 2] << std::endl;
		std::cout << "      Exchange branch nodes (w/ Allgather)     : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_BRANCH_NODES] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_BRANCH_NODES + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EXCHANGE_BRANCH_NODES + 2] << std::endl;
		std::cout << "      Insert branch nodes into global tree     : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE + 2] << std::endl;
		std::cout << "      Update global tree                       : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_GLOBAL_TREE] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_GLOBAL_TREE + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::UPDATE_GLOBAL_TREE + 2] << std::endl;
		std::cout << "      Find target neurons (w/ RMA)             : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::FIND_TARGET_NEURONS] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::FIND_TARGET_NEURONS + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::FIND_TARGET_NEURONS + 2] << std::endl;
		std::cout << "      Empty remote nodes cache                 : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EMPTY_REMOTE_NODES_CACHE] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EMPTY_REMOTE_NODES_CACHE + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::EMPTY_REMOTE_NODES_CACHE + 2] << std::endl;
		std::cout << "      Create synapses (w/ Alltoall)            : " << std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CREATE_SYNAPSES] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CREATE_SYNAPSES + 1] << " | "
			<< std::setw(Constants::print_width) << timers_global[3 * TimerRegion::CREATE_SYNAPSES + 2] << std::endl;

		// Restore old precision
		std::cout.precision(old_precision);

		//cout << "\n======== TIMERS RANK 0 ========" << std::endl;
		//neurons.print_timers();

		//cout << "\n======== MEMORY USAGE RANK 0 ========" << std::endl;

		std::cout << "\n======== RMA MEMORY ALLOCATOR RANK 0 ========" << std::endl;
		std::cout << "Min num objects available: " << MPIWrapper::mpi_rma_mem_allocator.get_min_num_avail_objects() << std::endl;
	}
}

int main(int argc, char** argv) {
	std::vector<std::string> arguments{ argv, argv + argc };
	/**
	 * Read command line parameters
	 */
	if (arguments.size() < 5) {
		std::cout << "Usage: " << arguments[0]
			<< " <acceptance criterion (theta)>"
			<< " <number neurons>"
			<< " <random number seed>"
			<< " <simulation steps>"
			<< " [<file with neuron positions>"
			<< " [<file with connections>]]"
			<< std::endl;
		exit(EXIT_FAILURE);
	}

	/**
	 * Init MPI and store some MPI infos
	 */
	MPIWrapper::init(argc, argv);

	size_t simulation_steps = stoull(arguments[4], nullptr, 10);

	/**
	 * Simulation parameters
	 */
	auto params = std::make_unique<Parameters>();;
	setDefaultParameters(params.get());
	setSpecificParameters(params.get(), arguments);

	MPIWrapper::init_neurons(params->num_neurons);
	MPIWrapper::print_infos_rank(0);

	// Init random number seeds
	randomNumberSeeds::partition = static_cast<int64_t>(MPIWrapper::my_rank);
	randomNumberSeeds::octree = static_cast<int64_t>(stol(arguments[3], nullptr, 10));

	// Rank 0 prints start time of simulation
	MPIWrapper::barrier(MPIWrapper::Scope::global);
	if (0 == MPIWrapper::my_rank) {
		std::stringstream sstring; // For output generation
		sstring << "\nSTART: " << Timers::wall_clock_time() << "\n";
		LogMessages::print_message_rank(sstring.str().c_str(), 0);
	}

	/**
	 * Initialize the simuliation log files
	 */
	Logs::init();

	GlobalTimers::timers.start(TimerRegion::INITIALIZATION);
	/**
	 * Calculate what my partition of the domain consist of
	 */

	auto partition = std::make_unique<Partition>(MPIWrapper::num_ranks, MPIWrapper::my_rank);
	const size_t my_num_subdomains = partition->get_my_num_subdomains();
	const size_t total_num_subdomains = partition->get_total_num_subdomains();

	// Check if int type can contain total size of branch nodes to receive in bytes
	// Every rank sends the same number of branch nodes, which is Partition::get_my_num_subdomains()
	if (std::numeric_limits<int>::max() < (my_num_subdomains * sizeof(OctreeNode))) {
		RelearnException::fail("int type is too small to hold the size in bytes of the branch nodes that are received from every rank in MPI_Allgather()");
		exit(EXIT_FAILURE);
	}

	/**
	* Create MPI RMA memory allocator
	*/
	MPIWrapper::init_mem_allocator(params->mpi_rma_mem_size);
	MPIWrapper::init_buffer_octree(total_num_subdomains);

	// Lock local RMA memory for local stores
	MPIWrapper::lock_window(MPIWrapper::my_rank, MPI_Locktype::exclusive);

	Simulation sim(params->seed_octree);
	sim.setParameters(std::move(params));
	sim.setPartition(std::move(partition));

	if (5 < argc) {
		std::string file_positions(arguments[5]);

		if (6 < argc) {
			std::string file_network(arguments[6]);
			sim.loadNeuronsFromFile(file_positions, file_network);
		}
		else {
			sim.loadNeuronsFromFile(file_positions);
		}
	}
	else {
		sim.placeRandomNeurons();
	}

	// Unlock local RMA memory and make local stores visible in public window copy
	MPIWrapper::unlock_window(MPIWrapper::my_rank);

	/**********************************************************************************/

	// The barrier ensures that every rank finished its local stores.
	// Otherwise, a "fast" rank might try to read from the RMA window of another
	// rank which has not finished (or even begun) its local stores
	MPIWrapper::barrier(MPIWrapper::Scope::global);// TODO(future) Really needed?

	GlobalTimers::timers.stop_and_add(TimerRegion::INITIALIZATION);

	for (size_t i = 0; i < 1; i++) {
		sim.registerNeuronMonitor(i);
	}

	sim.simulate(simulation_steps);

	printTimers();

	MPIWrapper::barrier(MPIWrapper::Scope::global);
	sim.finalize();

	MPIWrapper::finalize();

	return 0;
}
