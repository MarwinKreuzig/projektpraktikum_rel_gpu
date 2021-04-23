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
#include "neurons/ElementType.h"
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"
#include "neurons/models/NeuronModels.h"
#include "neurons/helper/NeuronMonitor.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "structure/Partition.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "sim/Simulation.h"
#include "neurons/models/SynapticElements.h"
#include "util/Timers.h"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#ifdef _OPENMP
#include <omp.h>
#else
void omp_set_num_threads(int num) { }
#endif

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
    app.add_option("-s,--steps", simulation_steps, "Simulation steps in ms.")->required();

    unsigned int seed_octree{};
    app.add_option("-r,--random-seed", seed_octree, "Random seed.")->required();

    int openmp_threads{ 1 };
    app.add_option("--openmp", openmp_threads, "Number of OpenMP Threads.");

    double accept_criterion{ Octree::default_theta };
    app.add_option("-t,--theta", accept_criterion, "Theta, the acceptance criterion. Default: 0.3.");

    auto* flag_interactive = app.add_flag("-i,--interactive", "Run interactively.");

    opt_num_neurons->excludes(opt_file_positions);
    opt_num_neurons->excludes(opt_file_network);
    opt_file_positions->excludes(opt_num_neurons);
    opt_file_network->excludes(opt_num_neurons);

    opt_file_network->needs(opt_file_positions);

    opt_file_positions->check(CLI::ExistingFile);
    opt_file_network->check(CLI::ExistingFile);

    double synaptic_elements_init_lb{ 0.0 };
    double synaptic_elements_init_ub{ 0.0 };
    app.add_option("--synaptic-elements-lower-bound", synaptic_elements_init_lb, "The minimum number of vacant synaptic elements per neuron. Must be smaller of equal to synaptic-elements-upper-bound.");
    app.add_option("--synaptic-elements-upper-bound", synaptic_elements_init_ub, "The maximum number of vacant synaptic elements per neuron. Must be larger or equal to synaptic-elements-lower-bound.");

    double target_calcium{ SynapticElements::default_C_target };
    app.add_option("--target-ca", target_calcium, "The target Ca2+ ions in each neuron. Standard is 0.7.");

    CLI11_PARSE(app, argc, argv);

    RelearnException::check(synaptic_elements_init_lb >= 0.0, "The minimum number of vacant synaptic elements must not be negative");
    RelearnException::check(synaptic_elements_init_ub >= synaptic_elements_init_lb, "The minimum number of vacant synaptic elements must not be larger than the maximum number");
    RelearnException::check(static_cast<bool>(*opt_num_neurons) || static_cast<bool>(*opt_file_positions), "Missing command line option, need num_neurons (-n,--num-neurons) or file_positions (-f,--file).");
    RelearnException::check(openmp_threads > 0, "Number of OpenMP Threads must be greater than 0 (or not set).");

    omp_set_num_threads(openmp_threads);

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
    RandomHolder::seed(RandomHolderKey::Partition, static_cast<unsigned int>(my_rank));
    RandomHolder::seed(RandomHolderKey::Octree, seed_octree);

    // Rank 0 prints start time of simulation
    MPIWrapper::barrier(MPIWrapper::Scope::global);
    if (0 == my_rank) {
        std::stringstream sstring; // For output generation

        sstring << "START: " << Timers::wall_clock_time() << "\n";

        sstring << "Chosen lower bound for vacant synaptic elements: " << synaptic_elements_init_lb << "\n";
        sstring << "Chosen upper bound for vacant synaptic elements: " << synaptic_elements_init_ub << "\n";
        sstring << "Chosen target calcium value: " << target_calcium;

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

    auto neuron_models = std::make_unique<models::ModelA>();

    auto axon_models = std::make_unique<SynapticElements>(ElementType::AXON, SynapticElements::default_eta_Axons, target_calcium,
        SynapticElements::default_nu, SynapticElements::default_vacant_retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    auto dend_ex_models = std::make_unique<SynapticElements>(ElementType::DENDRITE, SynapticElements::default_eta_Dendrites_exc, target_calcium,
        SynapticElements::default_nu, SynapticElements::default_vacant_retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    auto dend_in_models = std::make_unique<SynapticElements>(ElementType::DENDRITE, SynapticElements::default_eta_Dendrites_inh, target_calcium,
        SynapticElements::default_nu, SynapticElements::default_vacant_retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    // Lock local RMA memory for local stores
    MPIWrapper::lock_window(my_rank, MPI_Locktype::exclusive);

    Simulation sim(partition);
    sim.set_acceptance_criterion_for_octree(accept_criterion);
    sim.set_neuron_models(std::move(neuron_models));
    sim.set_axons(std::move(axon_models));
    sim.set_dendrites_ex(std::move(dend_ex_models));
    sim.set_dendrites_in(std::move(dend_in_models));

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
            }

            if (yn == 'y' || yn == 'Y') {
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
