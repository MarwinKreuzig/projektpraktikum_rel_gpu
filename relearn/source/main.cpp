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

#include <CLI/App.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Config.hpp>

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
	 * Init MPI and store some MPI infos
	 */
    MPIWrapper::init(argc, argv);

    const int my_rank = MPIWrapper::get_my_rank();
    const int num_ranks = MPIWrapper::get_num_ranks();

    // Command line arguments
    CLI::App app{ "" };

    size_t num_neurons{};
    auto* opt_num_neurons = app.add_option("-n,--num-neurons", num_neurons, "Number of neurons.");

    std::string file_positions{};
    auto* opt_file_positions = app.add_option("-f,--file", file_positions, "File with neuron positions.");

    std::string file_network{};
    auto* opt_file_network = app.add_option("-g,--graph", file_network, "File with neuron connections.");

    std::string log_prefix{};
    auto* opt_log_prefix = app.add_option("-p,--log-prefix", log_prefix, "Prefix for log files.");

    std::string log_path{};
    auto* opt_log_path = app.add_option("-l,--log-path", log_path, "Path for log files.");

    size_t simulation_steps{};
    app.add_option("-s,--steps", simulation_steps, "Simulation steps in ms")->required();

    unsigned int seed_octree{};
    app.add_option("-r,--random-seed", seed_octree, "Random seed.")->required();

    double accept_criterion{ 0.0 };
    app.add_option("-t,--theta", accept_criterion, "Theta, the acceptance criterion. Default: 0.0")->required();

    auto* flag_interactive = app.add_flag("-i,--interactive", "Run interactively.");

    opt_num_neurons->excludes(opt_file_positions);
    opt_num_neurons->excludes(opt_file_network);
    opt_file_positions->excludes(opt_num_neurons);
    opt_file_network->excludes(opt_num_neurons);

    opt_file_network->needs(opt_file_positions);

    opt_file_positions->check(CLI::ExistingFile);
    opt_file_network->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    RelearnException::check(static_cast<bool>(*opt_num_neurons) || static_cast<bool>(*opt_file_positions), "Missing command line option, need num_neurons (-n,--num-neurons) or file_positions (-f,--file).");

    /**
	 * Initialize the simuliation log files
	 */
    if (static_cast<bool>(*opt_log_path)) {
        LogFiles::set_output_path(log_path);
    }
    if (static_cast<bool>(*opt_log_prefix)) {
        LogFiles::set_general_prefix(log_prefix);
    }

    LogFiles::init();

    // Init random number seeds
    randomNumberSeeds::partition = static_cast<unsigned int>(my_rank);
    randomNumberSeeds::octree = seed_octree;

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

    if (static_cast<bool>(*opt_num_neurons)) {
        const double frac_exc = 0.8;
        sim.place_random_neurons(num_neurons, frac_exc);
    } else {
        if (static_cast<bool>(*opt_file_network)) {
            sim.load_neurons_from_file(file_positions, file_network);
        } else {
            sim.load_neurons_from_file(file_positions);
        }
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
    const auto steps_per_simulation = simulation_steps / step_monitor;

    NeuronMonitor::max_steps = steps_per_simulation;
    NeuronMonitor::current_step = 0;

    for (size_t i = 0; i < 1; i++) {
        sim.register_neuron_monitor(i);
    }

    auto simulate = [&]() {
        sim.simulate(simulation_steps, step_monitor);

        Timers::print();

        MPIWrapper::barrier(MPIWrapper::Scope::global);
        sim.finalize();
    };

    simulate();

    if (static_cast<bool>(*flag_interactive)) {
        while (true) {
            std::cout << "Interactive run. Run another " << simulation_steps << " simulation steps? [y/n]\n";
            char yn{ 'n' };
            auto n = scanf(" %c", &yn);
            RelearnException::check(static_cast<bool>(n), "Error on while reading input with scanf.");
            if (yn == 'n' || yn == 'N') {
                break;
            } else if (yn == 'y' || yn == 'Y') {
                sim.increase_monitoring_capacity(steps_per_simulation);
                simulate();
            } else {
                std::stringstream ss{};
                ss << "Input for question to run another " << simulation_steps << " simulation steps was not valid.";
                RelearnException::fail(ss.str());
            }
        }
    }

    MPIWrapper::finalize();

    return 0;
}
