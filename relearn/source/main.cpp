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
#include "LogFiles.h"
#include "MPIWrapper.h"
#include "NeuronModels.h"
#include "NeuronMonitor.h"
#include "Partition.h"
#include "Random.h"
#include "RelearnException.h"
#include "Simulation.h"
#include "Timers.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <array>
#include <bitset>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>

int main(int argc, char** argv) {
    /**
	 * Read command line parameters
	 */
    std::vector<std::string> arguments{ argv, argv + argc };
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

    /**
	 * Initialize the simuliation log files
	 */
    LogFiles::init();

    const int my_rank = MPIWrapper::get_my_rank();
    const int num_ranks = MPIWrapper::get_num_ranks();

    double accept_criterion = 0.0;
    if (arguments[1] == "naive") {
        accept_criterion = 0.0;
    } else {
        accept_criterion = std::stod(arguments[1], nullptr);
    }

    size_t num_neurons = stoull(arguments[2], nullptr, 10);
    size_t seed_octree = stol(arguments[3], nullptr, 10);
    size_t simulation_steps = stoull(arguments[4], nullptr, 10);

    //MPIWrapper::init_neurons(params->num_neurons);
    MPIWrapper::print_infos_rank(0);

    // Init random number seeds
    randomNumberSeeds::partition = static_cast<unsigned int>(my_rank);
    randomNumberSeeds::octree = static_cast<unsigned int>(seed_octree);

    // Rank 0 prints start time of simulation
    MPIWrapper::barrier(MPIWrapper::Scope::global);
    if (0 == my_rank) {
        std::stringstream sstring; // For output generation
        sstring << "\nSTART: " << Timers::wall_clock_time() << "\n";
        LogFiles::print_message_rank(sstring.str().c_str(), 0);
    }

    GlobalTimers::timers.start(TimerRegion::INITIALIZATION);

    /**
	 * Calculate what my partition of the domain consist of
	 */
    auto partition = std::make_shared<Partition>(num_ranks, my_rank);
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
    MPIWrapper::init_buffer_octree(total_num_subdomains);

    // Lock local RMA memory for local stores
    MPIWrapper::lock_window(my_rank, MPI_Locktype::exclusive);

    Simulation sim(accept_criterion, partition);
    sim.set_neuron_models(std::make_unique<models::ModelA>());

    if (5 < argc) {
        std::string file_positions(arguments[5]);

        if (6 < argc) {
            std::string file_network(arguments[6]);
            sim.load_neurons_from_file(file_positions, file_network);
        } else {
            sim.load_neurons_from_file(file_positions);
        }
    } else {
        double frac_exc = 0.8;
        sim.place_random_neurons(num_neurons, frac_exc);
    }

    // Unlock local RMA memory and make local stores visible in public window copy
    MPIWrapper::unlock_window(my_rank);

    /**********************************************************************************/

    // The barrier ensures that every rank finished its local stores.
    // Otherwise, a "fast" rank might try to read from the RMA window of another
    // rank which has not finished (or even begun) its local stores
    MPIWrapper::barrier(MPIWrapper::Scope::global); // TODO(future) Really needed?

    GlobalTimers::timers.stop_and_add(TimerRegion::INITIALIZATION);

    const auto step_monitor = 100;

    NeuronMonitor::max_steps = simulation_steps / step_monitor;
    NeuronMonitor::current_step = 0;

    for (size_t i = 0; i < 1; i++) {
        sim.register_neuron_monitor(i);
    }

    sim.simulate(simulation_steps, step_monitor);

    Timers::print();

    MPIWrapper::barrier(MPIWrapper::Scope::global);
    sim.finalize();

    MPIWrapper::finalize();

    return 0;
}
