#include <iomanip>
#include <bitset>
#include <limits>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#include <iostream>
#include <locale>
#include <mpi.h>

#include "Timers.h"
#include "Neurons.h"
#include "NeuronModels.h"
#include "SynapticElements.h"
#include "Octree.h"
#include "Parameters.h"
#include "LogFiles.h"
#include "LogMessages.h"
#include "MPIInfos.h"
#include "NetworkGraph.h"
#include "MPI_RMA_MemAllocator.h"
#include "Partition.h"
#include "MPIUserDefinedOperation.h"
#include "randomNumberSeeds.h"
#include "NeuronToSubdomainAssignment.h"
#include "SubdomainFromFile.h"
#include "SubdomainFromNeuronDensity.h"
#include "NeuronIdMap.h"
#include "Utility.h"
#include "NeuronMonitor.h"

void setDefaultParameters(Parameters& params) noexcept {
	params.frac_neurons_exc = 0.8;                          // CHANGE
	params.x_0 = 0.05;
	params.tau_x = 5.0;
	params.k = 0.03;
	params.tau_C = 10000; //5000;   //very old 60.0;
	params.beta = 0.001;  //very old 0.05;
	params.h = 10;
	params.C_target = 0.5; // gold 0.5;
	params.refrac_time = 4.0;
	params.eta_A = 0; //0.4; // gold 0.0;
	params.eta_D_ex = 0; //0.1, // gold 0.0;
	params.eta_D_in = 0.0;
	params.nu = 1e-4; // gold 1e-5; // element growth rate
	params.vacant_retract_ratio = 0;
	params.sigma = 750.0;
	params.num_log_files = 9;  // NOT USED
	params.log_start_neuron = 0;  // NOT USED
	params.mpi_rma_mem_size = 300 * 1024 * 1024;  // 300 MB
	params.max_num_pending_vacant_axons = 1000;
	// params.seed_partition (No global parameter. Every process uses a different seed, its rank. See below)
}

void setSpecificParameters(Parameters& params, int argc, char** argv) {
	double accept_criterion = 0.0;
	bool naive_method = false;

	// Parameter equal to "naive"
	if (!strcoll(argv[1], "naive")) {
		naive_method = true;
	}
	else {
		accept_criterion = strtod(argv[1], nullptr);
	}

	params.simulation_time = strtoull(argv[4], nullptr, 10);  //6000000;
	params.num_neurons = strtoull(argv[2], nullptr, 10);        // CHANGE
	params.accept_criterion = accept_criterion;             // CHANGE
	params.naive_method = naive_method;                 // CHANGE
	params.file_with_neuron_positions = (argc >= 6) ? argv[5] : "";
	params.file_with_network = (argc >= 7) ? argv[6] : "";
	params.seed_octree = strtol(argv[3], nullptr, 10);

	/**
	* Parameter sanity checks
	*/

	// Needed to avoid creating autapses
	if (!(params.accept_criterion <= 0.5)) {
		std::stringstream sstring;
		sstring << __FUNCTION__ << ": accept_criterion > 0.5";
		LogMessages::print_error(sstring.str().c_str());
		exit(EXIT_FAILURE);
	}

	// Number of ranks must be 2^n so that
	// the connectivity update works correctly
	const std::bitset<sizeof(int) * 8> bitset_num_ranks(MPIInfos::num_ranks);
	if (1 != bitset_num_ranks.count() && (0 == MPIInfos::my_rank)) {
		std::stringstream sstring;
		sstring << __FUNCTION__ << ": Number of ranks must be of the form 2^n";
		LogMessages::print_error(sstring.str().c_str());
		exit(EXIT_FAILURE);
	}
}

void printTimers() {
	using namespace std;
	/**
	 * Print timers and memory usage
	 */

	MPI_Op mpi_op;
	MPI_Datatype mpi_datatype_3_double;

	MPI_Op_create((MPI_User_function*)MPIUserDefinedOperation::min_sum_max, 1, &mpi_op);

	MPI_Type_contiguous(3, MPI_DOUBLE, &mpi_datatype_3_double);
	MPI_Type_commit(&mpi_datatype_3_double);

	double timers_local[3 * TimerRegion::NUM_TIMER_REGIONS] = { 0.0 };
	for (int i = 0; i < TimerRegion::NUM_TIMER_REGIONS; ++i) {
		for (int j = 0; j < 3; ++j) {
			timers_local[3 * i + j] = GlobalTimers::timers.get_elapsed(i);
		}
	}

	double timers_global[3 * TimerRegion::NUM_TIMER_REGIONS];

	MPI_Reduce(timers_local, timers_global, TimerRegion::NUM_TIMER_REGIONS, mpi_datatype_3_double, mpi_op, 0, MPI_COMM_WORLD);

#ifndef NDEBUG
	stringstream sstring;
	// Check result of MPI_Reduce
	for (int i = 0; i < 3 * TimerRegion::NUM_TIMER_REGIONS; i++) {
		sstring << timers_global[i] << " ";
	}
	LogMessages::print_message_rank(sstring.str().c_str(), MPIInfos::my_rank);
#endif

	// Divide second entry of (min, sum, max), i.e., sum, by the number of ranks
	// so that sum becomes average
	for (int i = 0; i < TimerRegion::NUM_TIMER_REGIONS; i++) {
		timers_global[3 * i + 1] /= MPIInfos::num_ranks;
	}

	if (0 == MPIInfos::my_rank) {
		// Set precision for aligned double output
		const auto old_precision = cout.precision();
		cout.precision(6);

		cout << "\n======== TIMERS GLOBAL OVER ALL RANKS ========" << endl;
		cout << "                                                (" << setw(12) << "    min" << " | " << setw(12) << "    avg" << " | " << setw(12) << "    max" << ") sec." << endl;
		cout << "TIMERS: main()" << endl;
		cout << "  Initialization                               : " << setw(12) << timers_global[3 * TimerRegion::INITIALIZATION] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::INITIALIZATION + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::INITIALIZATION + 2] << endl;
		cout << "  Simulation loop                              : " << setw(12) << timers_global[3 * TimerRegion::SIMULATION_LOOP] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::SIMULATION_LOOP + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::SIMULATION_LOOP + 2] << endl;
		cout << "    Update electrical activity                 : " << setw(12) << timers_global[3 * TimerRegion::UPDATE_ELECTRICAL_ACTIVITY] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_ELECTRICAL_ACTIVITY + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_ELECTRICAL_ACTIVITY + 2] << endl;
		cout << "      Barrier 1                                : " << setw(12) << timers_global[3 * TimerRegion::BARRIER_1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::BARRIER_1 + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::BARRIER_1 + 2] << endl;
		cout << "      Prepare sending spikes                   : " << setw(12) << timers_global[3 * TimerRegion::PREPARE_SENDING_SPIKES] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::PREPARE_SENDING_SPIKES + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::PREPARE_SENDING_SPIKES + 2] << endl;
		cout << "      Prepare num neuron ids                   : " << setw(12) << timers_global[3 * TimerRegion::PREPARE_NUM_NEURON_IDS] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::PREPARE_NUM_NEURON_IDS + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::PREPARE_NUM_NEURON_IDS + 2] << endl;
		cout << "      Barrier 2                                : " << setw(12) << timers_global[3 * TimerRegion::BARRIER_2] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::BARRIER_2 + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::BARRIER_2 + 2] << endl;
		cout << "      All to all                               : " << setw(12) << timers_global[3 * TimerRegion::ALL_TO_ALL] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::ALL_TO_ALL + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::ALL_TO_ALL + 2] << endl;
		cout << "      Alloc mem for neuron ids                 : " << setw(12) << timers_global[3 * TimerRegion::ALLOC_MEM_FOR_NEURON_IDS] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::ALLOC_MEM_FOR_NEURON_IDS + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::ALLOC_MEM_FOR_NEURON_IDS + 2] << endl;
		cout << "      Barrier 3                                : " << setw(12) << timers_global[3 * TimerRegion::BARRIER_3] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::BARRIER_3 + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::BARRIER_3 + 2] << endl;
		cout << "      Exchange neuron ids                      : " << setw(12) << timers_global[3 * TimerRegion::EXCHANGE_NEURON_IDS] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::EXCHANGE_NEURON_IDS + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::EXCHANGE_NEURON_IDS + 2] << endl;
		cout << "      Calculate synaptic input                 : " << setw(12) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_INPUT] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_INPUT + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::CALC_SYNAPTIC_INPUT + 2] << endl;
		cout << "      Calculate activity                       : " << setw(12) << timers_global[3 * TimerRegion::CALC_ACTIVITY] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::CALC_ACTIVITY + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::CALC_ACTIVITY + 2] << endl;
		cout << "    Update #synaptic elements delta            : " << setw(12) << timers_global[3 * TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA + 2] << endl;
		cout << "    Connectivity update                        : " << setw(12) << timers_global[3 * TimerRegion::UPDATE_CONNECTIVITY] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_CONNECTIVITY + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_CONNECTIVITY + 2] << endl;
		cout << "      Update #synaptic elements + del synapses : " << setw(12) << timers_global[3 * TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES + 2] << endl;
		cout << "      Update local trees                       : " << setw(12) << timers_global[3 * TimerRegion::UPDATE_LOCAL_TREES] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_LOCAL_TREES + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_LOCAL_TREES + 2] << endl;
		cout << "      Exchange branch nodes (w/ Allgather)     : " << setw(12) << timers_global[3 * TimerRegion::EXCHANGE_BRANCH_NODES] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::EXCHANGE_BRANCH_NODES + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::EXCHANGE_BRANCH_NODES + 2] << endl;
		cout << "      Insert branch nodes into global tree     : " << setw(12) << timers_global[3 * TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE + 2] << endl;
		cout << "      Update global tree                       : " << setw(12) << timers_global[3 * TimerRegion::UPDATE_GLOBAL_TREE] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_GLOBAL_TREE + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::UPDATE_GLOBAL_TREE + 2] << endl;
		cout << "      Find target neurons (w/ RMA)             : " << setw(12) << timers_global[3 * TimerRegion::FIND_TARGET_NEURONS] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::FIND_TARGET_NEURONS + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::FIND_TARGET_NEURONS + 2] << endl;
		cout << "      Empty remote nodes cache                 : " << setw(12) << timers_global[3 * TimerRegion::EMPTY_REMOTE_NODES_CACHE] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::EMPTY_REMOTE_NODES_CACHE + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::EMPTY_REMOTE_NODES_CACHE + 2] << endl;
		cout << "      Create synapses (w/ Alltoall)            : " << setw(12) << timers_global[3 * TimerRegion::CREATE_SYNAPSES] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::CREATE_SYNAPSES + 1] << " | "
			<< setw(12) << timers_global[3 * TimerRegion::CREATE_SYNAPSES + 2] << endl;

		// Restore old precision
		cout.precision(old_precision);

		//cout << "\n======== TIMERS RANK 0 ========" << endl;
		//neurons.print_timers();

		//cout << "\n======== MEMORY USAGE RANK 0 ========" << endl;

		cout << "\n======== RMA MEMORY ALLOCATOR RANK 0 ========" << endl;
		cout << "Min num objects available: " << MPIInfos::mpi_rma_mem_allocator.get_min_num_avail_objects() << endl;

		MPI_Type_free(&mpi_datatype_3_double);
		MPI_Op_free(&mpi_op);
	}
}

void printNeuronMonitor(const NeuronMonitor& nm, size_t neuron_id) {
	std::ofstream outfile(std::to_string(neuron_id) + ".csv", std::ios::trunc);
	outfile << std::setprecision(5);

	outfile.imbue(std::locale("German_germany"));

	outfile << "Step;Fired;Refrac;x;Ca;I_sync;axons;axons_connected;dendrites_exc;dendrites_exc_connected;dendrites_inh;dendrites_inh_connected";
	outfile << "\n";

	const auto& infos = nm.get_informations();

	const auto filler = ";";
	const auto width = 6;

	auto ctr = 0;

	for (auto& info : infos) {
		outfile << ctr << filler;
		outfile << /*std::setw(width) <<*/ info.fired << filler;
		outfile << /*std::setw(width) <<*/ info.refrac << filler;
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

int main(int argc, char** argv) {
	/**
	 * Read command line parameters
	 */
	if (argc < 5) {
		std::cout << "Usage: " << argv[0]
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
	MPIInfos::init(argc, argv);

	/**
	 * Simulation parameters
	 */
	Parameters params;
	setDefaultParameters(params);
	setSpecificParameters(params, argc, argv);

	MPIInfos::init_neurons(params.num_neurons);
	MPIInfos::print_infos_rank(0);

	// Init random number seeds
	randomNumberSeeds::partition = static_cast<long int>(MPIInfos::my_rank);
	randomNumberSeeds::octree = strtol(argv[3], nullptr, 10);

	// Rank 0 prints start time of simulation
	MPI_Barrier(MPI_COMM_WORLD);
	if (0 == MPIInfos::my_rank) {
		std::stringstream sstring; // For output generation
		sstring << "\nSTART: " << Timers::wall_clock_time() << "\n";
		LogMessages::print_message_rank(sstring.str().c_str(), 0);
	}

	NeuronToSubdomainAssignment* neurons_in_subdomain = nullptr;
	if (5 < argc) {
		neurons_in_subdomain = new SubdomainFromFile(params.file_with_neuron_positions, params.num_neurons);
		// Set parameter based on actual neuron population
		params.frac_neurons_exc = neurons_in_subdomain->desired_ratio_neurons_exc();
	}
	else {
		neurons_in_subdomain = new SubdomainFromNeuronDensity(params.num_neurons, params.frac_neurons_exc);
	}

	if (0 == MPIInfos::my_rank) {
		std::cout << params << std::endl;
	}

	/**
	 * Initialize the simuliation log files
	 */
	Logs::init();

	GlobalTimers::timers.start(TimerRegion::INITIALIZATION);
	/**
	 * Calculate what my partition of the domain consist of
	 */
	Partition partition(MPIInfos::num_ranks, MPIInfos::my_rank);

	// Check if int type can contain total size of branch nodes to receive in bytes
	// Every rank sends the same number of branch nodes, which is partition.get_my_num_subdomains()
	if (std::numeric_limits<int>::max() < (partition.get_my_num_subdomains() * sizeof(OctreeNode))) {
		std::stringstream sstring; // For output generation
		sstring << __FUNCTION__ << ": int type is too small to hold the size in bytes of the branch "
			"                     nodes that are received from every rank in MPI_Allgather()";
		LogMessages::print_error(sstring.str().c_str());
		exit(EXIT_FAILURE);
	}

	/**
	 * Create MPI RMA memory allocator
	 */
	MPIInfos::init_mem_allocator(params.mpi_rma_mem_size);
	MPIInfos::init_buffer_octree(partition.get_total_num_subdomains());

	// Lock local RMA memory for local stores
	MPIInfos::lock_window(MPIInfos::my_rank, MPI_Locktype::exclusive);

	/**
	 * Create neuron population
	 */
	Neurons neurons = partition.get_local_neurons(params, *neurons_in_subdomain);

	partition.print_my_subdomains_info_rank(0);
	partition.print_my_subdomains_info_rank(1);

	LogMessages::print_message_rank("Neurons created", 0);

	/**********************************************************************************/
	NeuronIdMap neuron_id_map(
		neurons.get_num_neurons(),
		neurons.get_positions().get_x_dims(),
		neurons.get_positions().get_y_dims(),
		neurons.get_positions().get_z_dims(),
		MPI_COMM_WORLD);

	LogMessages::print_message_rank("Neuron id map created", 0);

	/**
	 * Init global tree parameters
	 */
	Octree global_tree(partition, params);
	global_tree.set_no_free_in_destructor(); // This needs to be changed later, as it's cleaner to free the nodes at destruction


	// Insert my local (subdomain) trees into my global tree
	for (size_t i = 0; i < partition.get_my_num_subdomains(); i++) {
		Octree* local_tree = &partition.get_subdomain_tree(i);
		global_tree.insert_local_tree(local_tree);
	}

	// Unlock local RMA memory and make local stores visible in public window copy
	MPIInfos::unlock_window(MPIInfos::my_rank);

	/**********************************************************************************/

	// The barrier ensures that every rank finished its local stores.
	// Otherwise, a "fast" rank might try to read from the RMA window of another
	// rank which has not finished (or even begun) its local stores
	MPI_Barrier(MPI_COMM_WORLD); // TODO Really needed?

	LogMessages::print_message_rank("Neurons inserted into subdomains", 0);
	LogMessages::print_message_rank("Subdomains inserted into global tree", 0);

	/**
	 * Create and init neural network
	 */
	NetworkGraph network_graph(neurons.get_num_neurons());
	// Neuronal connections provided in file
	if (6 < argc) {
		//network_graph.add_edge_weights(params.file_with_network, neuron_id_map);
		network_graph.add_edges_from_file(params.file_with_network, params.file_with_neuron_positions, neuron_id_map, partition);
		network_graph.print(std::cout, neuron_id_map);
		return 0;
	}
	LogMessages::print_message_rank("Network graph created", 0);
	

	// Init number of synaptic elements and assign EXCITATORY or INHIBITORY signal type
	// to the dendrites. Assignment of the signal type to the axons is done in
	// Partition::insert_neurons_into_my_subdomains
	neurons.init_synaptic_elements();
	//    neurons.init_synaptic_elements(network_graph);
	LogMessages::print_message_rank("Synaptic elements initialized \n", 0);

	neurons.print_neurons_overview_to_log_file_on_rank_0(0, Logs::get("neurons_overview"), params);
	neurons.print_sums_of_synapses_and_elements_to_log_file_on_rank_0(0, Logs::get("sums"), params, 0, 0);

	GlobalTimers::timers.stop_and_add(TimerRegion::INITIALIZATION);

	uint64_t total_creations = 0;
	uint64_t total_deletions = 0;

	const auto step_monitor = 100;

	NeuronMonitor::max_steps = params.simulation_time / step_monitor;
	NeuronMonitor::current_step = 0;

	std::vector<NeuronMonitor> monitors;

	for (size_t i = 0; i < 1; i++) {
		monitors.emplace_back(i, neurons);
	}

	// Start timing simulation loop
	GlobalTimers::timers.start(TimerRegion::SIMULATION_LOOP);

	/**
	 * Simulation loop
	 */
	for (size_t step = 1; step <= params.simulation_time; step++) {

		if (step % step_monitor == 0) {
			for (auto& mn : monitors) {
				mn.record_data();
			}

			NeuronMonitor::current_step++;
		}

		// Provide neuronal network to neuron models for one iteration step
		GlobalTimers::timers.start(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);
		neurons.update_electrical_activity(network_graph);
		GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);

		// Calc how many synaptic elements grow/retract
		// Apply the change in number of elements during connectivity update
		GlobalTimers::timers.start(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);
		neurons.update_number_synaptic_elements_delta();
		GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);

		//if (0 == MPIInfos::my_rank && step % 50 == 0) {
		//	std::cout << "** STATE AFTER: " << step << " of " << params.simulation_time
		//		<< " msec ** [" << Timers::wall_clock_time() << "]\n";
		//}

		// Update connectivity every 100 ms
		if (step % 100 == 0) {
			size_t num_synapses_deleted = 0;
			size_t num_synapses_created = 0;

			if (0 == MPIInfos::my_rank) {
				std::cout << "** UPDATE CONNECTIVITY AFTER: " << step << " of " << params.simulation_time
					<< " msec ** [" << Timers::wall_clock_time() << "]\n";
			}

			GlobalTimers::timers.start(TimerRegion::UPDATE_CONNECTIVITY);

			neurons.update_connectivity(global_tree, network_graph, num_synapses_deleted, num_synapses_created);

			GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_CONNECTIVITY);

			// Get total number of synapses deleted and created
			const uint64_t cnts_local[2] = { static_cast<uint64_t>(num_synapses_deleted), static_cast<uint64_t>(num_synapses_created) };
			uint64_t cnts_global[2];

			MPI_Reduce(cnts_local, cnts_global, 2, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

			if (0 == MPIInfos::my_rank) {
				total_deletions += cnts_global[0] / 2;
				total_creations += cnts_global[1] / 2;
			}

			if (cnts_global[0] != 0.0) {
				std::stringstream sstring; // For output generation
				sstring << "Sum (all processes) number synapses deleted: " << cnts_global[0] / 2;
				LogMessages::print_message_rank(sstring.str().c_str(), 0);
			}

			if (cnts_global[1] != 0.0) {
				std::stringstream sstring; // For output generation
				sstring << "Sum (all processes) number synapses created: " << cnts_global[1] / 2;
				LogMessages::print_message_rank(sstring.str().c_str(), 0);
			}

			neurons.print_sums_of_synapses_and_elements_to_log_file_on_rank_0(
				step, Logs::get("sums"), params, num_synapses_deleted, num_synapses_created);

			std::cout << std::flush;
		}

		// Print details every 500 ms
		if (step % 500 == 0) {
			neurons.print_neurons_overview_to_log_file_on_rank_0(step, Logs::get("neurons_overview"), params);
		}
	}

	// Stop timing simulation loop
	GlobalTimers::timers.stop_and_add(TimerRegion::SIMULATION_LOOP);

	for (auto& monitor : monitors)
		printNeuronMonitor(monitor, monitor.get_target_id());

	neurons.print_network_graph_to_log_file(Logs::get("network_rank_" + MPIInfos::my_rank_str), network_graph,
		params, neuron_id_map);
	neurons.print_positions_to_log_file(Logs::get("positions_rank_" + MPIInfos::my_rank_str), params, neuron_id_map);

	printTimers();

	// Free object created based on command line parameters
	neurons_in_subdomain->write_neurons_to_file("output_positions.txt");
	network_graph.write_synapses_to_file("output_edges.txt", neuron_id_map, partition);
	delete neurons_in_subdomain;

	MPI_Barrier(MPI_COMM_WORLD);
	if (0 == MPIInfos::my_rank) {
		std::stringstream sstring; // For output generation
		sstring << "\n";
		sstring << "\n" << "Total creations: " << total_creations << "\n";
		sstring << "Total deletions: " << total_deletions << "\n";
		sstring << "END: " << Timers::wall_clock_time() << "\n";
		LogMessages::print_message_rank(sstring.str().c_str(), 0);
	}

	/**
	 * Finalize MPI
	 */
	MPIInfos::finalize();

	return 0;
}
