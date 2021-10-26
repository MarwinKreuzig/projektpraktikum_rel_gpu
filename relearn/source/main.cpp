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
#include "algorithm/BarnesHut.h"
#include "algorithm/FastMultipoleMethods.h"
#include "algorithm/Naive.h"
#include "algorithm/Types.h"
#include "io/InteractiveNeuronIO.h"
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"
#include "neurons/ElementType.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/models/NeuronModels.h"
#include "neurons/models/SynapticElements.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "sim/Simulation.h"
#include "structure/Octree.h"
#include "structure/Partition.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include "spdlog/spdlog.h"

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

void print_sizes() {
    struct empty_t { };

    const auto sizeof_vec3_double = sizeof(Vec3d);
    const auto sizeof_vec3_size_t = sizeof(Vec3s);

    const auto sizeof_virtual_plasticity_element = sizeof(VirtualPlasticityElement);

    const auto sizeof_fmm_cell_attributes = sizeof(FastMultipoleMethodsCell);
    const auto sizeof_bh_cell_attributes = sizeof(BarnesHutCell);
    const auto sizeof_bh_naive_attributes = sizeof(NaiveCell);

    const auto sizeof_empty_cell = sizeof(Cell<empty_t>);
    const auto sizeof_fmm_cell = sizeof(Cell<FastMultipoleMethodsCell>);
    const auto sizeof_bh_cell = sizeof(Cell<BarnesHutCell>);
    const auto sizeof_naive_cell = sizeof(Cell<NaiveCell>);

    const auto sizeof_octreenode = sizeof(OctreeNode<empty_t>);
    const auto sizeof_fmm_octreenode = sizeof(OctreeNode<FastMultipoleMethodsCell>);
    const auto sizeof_bh_octreenode = sizeof(OctreeNode<BarnesHutCell>);
    const auto sizeof_naive_octreenode = sizeof(OctreeNode<NaiveCell>);

    std::stringstream ss{};

    ss << '\n';
    ss << "Size of Vec3d: " << sizeof_vec3_double << '\n';
    ss << "Size of Vec3s: " << sizeof_vec3_size_t << '\n';
    ss << "Size of VirtualPlasticityElement: " << sizeof_virtual_plasticity_element << "\n";
    ss << "Size of FastMultipoleMethodsCell: " << sizeof_fmm_cell_attributes << '\n';
    ss << "Size of BarnesHutCell: " << sizeof_bh_cell_attributes << '\n';
    ss << "Size of NaiveCell: " << sizeof_bh_naive_attributes << "\n";
    ss << "Size of Cell<empty_t>: " << sizeof_empty_cell << '\n';
    ss << "Size of Cell<FastMultipoleMethodsCell>: " << sizeof_fmm_cell << '\n';
    ss << "Size of Cell<BarnesHutCell>: " << sizeof_bh_cell << '\n';
    ss << "Size of Cell<NaiveCell>: " << sizeof_naive_cell << "\n";
    ss << "Size of OctreeNode<empty_t>: " << sizeof_octreenode << '\n';
    ss << "Size of OctreeNode<FastMultipoleMethodsCell>: " << sizeof_fmm_octreenode << '\n';
    ss << "Size of OctreeNode<BarnesHutCell>: " << sizeof_bh_octreenode << '\n';
    ss << "Size of OctreeNode<NaiveCell>: " << sizeof_naive_octreenode;

    LogFiles::print_message_rank(0, ss.str());
}

void print_arguments(int argc, char** argv) {
    std::stringstream ss{};

    for (auto i = 0; i < argc; i++) {
        ss << argv[i] << ' ';
    }

    LogFiles::print_message_rank(0, ss.str());
}

int main(int argc, char** argv) {
    /**
	 * Init MPI and store some MPI infos
	 */
    MPIWrapper::init(argc, argv);

    print_arguments(argc, argv);
    print_sizes();

    const int my_rank = MPIWrapper::get_my_rank();
    const int num_ranks = MPIWrapper::get_num_ranks();

    // Command line arguments
    CLI::App app{ "" };

    AlgorithmEnum algorithm = AlgorithmEnum::BarnesHut;
    std::map<std::string, AlgorithmEnum> cli_parse_map{
        { "naive", AlgorithmEnum::Naive },
        { "barnes-hut", AlgorithmEnum::BarnesHut },
        { "fast-multipole-methods", AlgorithmEnum::FastMultipoleMethods }
    };

    auto* opt_algorithm = app.add_option("-a,--algorithm", algorithm, "The algorithm that is used for finding the targets");
    opt_algorithm->required()->transform(CLI::CheckedTransformer(cli_parse_map, CLI::ignore_case));

    double accept_criterion{ BarnesHut::default_theta };
    auto* opt_accept_criterion = app.add_option("-t,--theta", accept_criterion, "Theta, the acceptance criterion for Barnes-Hut. Default: 0.3. Required Barnes-Hut.");

    size_t num_neurons{};
    auto* opt_num_neurons = app.add_option("-n,--num-neurons", num_neurons, "Number of neurons.");

    std::string file_positions{};
    auto* opt_file_positions = app.add_option("-f,--file", file_positions, "File with neuron positions.");

    std::string file_network{};
    auto* opt_file_network = app.add_option("-g,--graph", file_network, "File with neuron connections.");

    std::string file_enable_interrupts{};
    auto* opt_file_enable_interrupts = app.add_option("--enable-interrupts", file_enable_interrupts, "File with the enable interrupts.");

    std::string file_disable_interrupts{};
    auto* opt_file_disable_interrupts = app.add_option("--disable-interrupts", file_disable_interrupts, "File with the disable interrupts.");

    std::string file_creation_interrupts{};
    auto* opt_file_creation_interrups = app.add_option("--creation-interrupts", file_creation_interrupts, "File with the creation interrupts.");

    double base_background_activity{ NeuronModel::default_base_background_activity };
    auto* opt_base_background_activity = app.add_option("--base-background-activity", base_background_activity, "The base background activity by which all neurons are exited");

    std::string log_prefix{};
    auto* opt_log_prefix = app.add_option("-p,--log-prefix", log_prefix, "Prefix for log files.");

    std::string log_path{};
    auto* opt_log_path = app.add_option("-l,--log-path", log_path, "Path for log files.");

    size_t simulation_steps{};
    app.add_option("-s,--steps", simulation_steps, "Simulation steps in ms.")->required();

    unsigned int random_seed{ 0 };
    app.add_option("-r,--random-seed", random_seed, "Random seed. Default: 0.");

    int openmp_threads{ 1 };
    app.add_option("--openmp", openmp_threads, "Number of OpenMP Threads.");

    auto* flag_interactive = app.add_flag("-i,--interactive", "Run interactively.");

    opt_num_neurons->excludes(opt_file_positions);
    opt_num_neurons->excludes(opt_file_network);
    opt_file_positions->excludes(opt_num_neurons);
    opt_file_network->excludes(opt_num_neurons);

    opt_file_network->needs(opt_file_positions);

    opt_file_positions->check(CLI::ExistingFile);
    opt_file_network->check(CLI::ExistingFile);

    opt_file_enable_interrupts->check(CLI::ExistingFile);
    opt_file_disable_interrupts->check(CLI::ExistingFile);
    opt_file_creation_interrups->check(CLI::ExistingFile);

    opt_log_path->check(CLI::ExistingDirectory);

    double synaptic_elements_init_lb{ 0.0 };
    double synaptic_elements_init_ub{ 0.0 };
    app.add_option("--synaptic-elements-lower-bound", synaptic_elements_init_lb, "The minimum number of vacant synaptic elements per neuron. Must be smaller of equal to synaptic-elements-upper-bound.");
    app.add_option("--synaptic-elements-upper-bound", synaptic_elements_init_ub, "The maximum number of vacant synaptic elements per neuron. Must be larger or equal to synaptic-elements-lower-bound.");

    double target_calcium{ SynapticElements::default_C_target };
    app.add_option("--target-ca", target_calcium, "The target Ca2+ ions in each neuron. Default is 0.7.");

    double nu{ SynapticElements::default_nu };
    app.add_option("--growth-rate", nu, "The growth rate for the synaptic elements. Default is 1e-5");

    double beta{ NeuronModel::default_beta };
    app.add_option("--beta", beta, "The amount of calcium ions gathered when a neuron fires. Default is 0.001");

    CLI11_PARSE(app, argc, argv);

    RelearnException::check(synaptic_elements_init_lb >= 0.0, "The minimum number of vacant synaptic elements must not be negative");
    RelearnException::check(synaptic_elements_init_ub >= synaptic_elements_init_lb, "The minimum number of vacant synaptic elements must not be larger than the maximum number");
    RelearnException::check(static_cast<bool>(*opt_num_neurons) || static_cast<bool>(*opt_file_positions), "Missing command line option, need num_neurons (-n,--num-neurons) or file_positions (-f,--file).");
    RelearnException::check(openmp_threads > 0, "Number of OpenMP Threads must be greater than 0 (or not set).");
    RelearnException::check(base_background_activity >= 0.0, "The base background activity must be non-negative.");

    if (algorithm == AlgorithmEnum::BarnesHut) {
        RelearnException::check(accept_criterion <= BarnesHut::max_theta, "Acceptance criterion must be smaller or equal to {}", BarnesHut::max_theta);
        RelearnException::check(accept_criterion > 0.0, "Acceptance criterion must be larger than 0.0");
    } else if (algorithm == AlgorithmEnum::FastMultipoleMethods) {
        const auto accept_criterion_set = opt_accept_criterion->count() > 0;
        RelearnException::check(!accept_criterion_set, "Acceptance criterion can only be set if Barnes-Hut is used");
    } else if (algorithm == AlgorithmEnum::Naive) {
        const auto accept_criterion_set = opt_accept_criterion->count() > 0;
        RelearnException::check(!accept_criterion_set, "Acceptance criterion can only be set if Barnes-Hut is used");
    } else {
        RelearnException::fail("Wrong algorithm chosen");
    }

    RelearnException::check(target_calcium >= SynapticElements::min_C_target, "Target calcium is smaller than {}", SynapticElements::min_C_target);
    RelearnException::check(target_calcium <= SynapticElements::max_C_target, "Target calcium is larger than {}", SynapticElements::max_C_target);

    RelearnException::check(nu >= SynapticElements::min_nu, "Growth rate is smaller than {}", SynapticElements::min_nu);
    RelearnException::check(nu <= SynapticElements::max_nu, "Growth rate is larger than {}", SynapticElements::max_nu);

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
    RandomHolder::seed(RandomHolderKey::Algorithm, random_seed);

    // Rank 0 prints start time of simulation
    MPIWrapper::barrier();
    if (0 == my_rank) {
        LogFiles::print_message_rank(0,
            "START: {}\n"
            "Chosen lower bound for vacant synaptic elements: {}\n"
            "Chosen upper bound for vacant synaptic elements: {}\n"
            "Chosen target calcium value: {}\n"
            "Chosen beta value: {}\n"
            "Chosen nu value: {}\n"
            "Chosen background activity: {}",
            Timers::wall_clock_time(), synaptic_elements_init_lb, synaptic_elements_init_ub, target_calcium, beta, nu, base_background_activity);

        LogFiles::write_to_file(LogFiles::EventType::Essentials, false,
            "Number of steps: {}\n"
            "Chosen lower bound for vacant synaptic elements: {}\n"
            "Chosen upper bound for vacant synaptic elements: {}\n"
            "Chosen target calcium value: {}\n"
            "Chosen beta value: {}\n"
            "Chosen nu value: {}\n"
            "Chosen background activity: {}",
            simulation_steps,
            synaptic_elements_init_lb,
            synaptic_elements_init_ub,
            target_calcium,
            beta,
            nu,
            base_background_activity);
    }

    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdate, false, "#step: creations deletions netto");
    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateCSV, false, "#step;creations;deletions;netto");
    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateLocal, false, "#step: creations deletions netto");

    Timers::start(TimerRegion::INITIALIZATION);

    /**
	 * Calculate what my partition of the domain consist of
	 */
    auto partition = std::make_shared<Partition>(num_ranks, my_rank);
    const size_t my_num_subdomains = partition->get_my_num_subdomains();
    const size_t total_num_subdomains = partition->get_total_num_subdomains();

    if (algorithm == AlgorithmEnum::BarnesHut) {
        // Check if int type can contain total size of branch nodes to receive in bytes
        // Every rank sends the same number of branch nodes, which is Partition::get_my_num_subdomains()
        if (std::numeric_limits<int>::max() < (my_num_subdomains * sizeof(OctreeNode<BarnesHutCell>))) {
            RelearnException::fail("int type is too small to hold the size in bytes of the branch nodes that are received from every rank in MPI_Allgather()");
            exit(EXIT_FAILURE);
        }

        /**
	     * Create MPI RMA memory allocator
	      */
        MPIWrapper::init_buffer_octree<BarnesHutCell>();
    } else if (algorithm == AlgorithmEnum::FastMultipoleMethods) {
        // Check if int type can contain total size of branch nodes to receive in bytes
        // Every rank sends the same number of branch nodes, which is Partition::get_my_num_subdomains()
        if (std::numeric_limits<int>::max() < (my_num_subdomains * sizeof(OctreeNode<FastMultipoleMethodsCell>))) {
            RelearnException::fail("int type is too small to hold the size in bytes of the branch nodes that are received from every rank in MPI_Allgather()");
            exit(EXIT_FAILURE);
        }

        /**
	      * Create MPI RMA memory allocator
	      */
        MPIWrapper::init_buffer_octree<FastMultipoleMethodsCell>();
    } else if (algorithm == AlgorithmEnum::Naive) {
        // Check if int type can contain total size of branch nodes to receive in bytes
        // Every rank sends the same number of branch nodes, which is Partition::get_my_num_subdomains()
        if (std::numeric_limits<int>::max() < (my_num_subdomains * sizeof(OctreeNode<NaiveCell>))) {
            RelearnException::fail("int type is too small to hold the size in bytes of the branch nodes that are received from every rank in MPI_Allgather()");
            exit(EXIT_FAILURE);
        }

        /**
	      * Create MPI RMA memory allocator
	      */
        MPIWrapper::init_buffer_octree<NaiveCell>();
    }

    auto neuron_models = std::make_unique<models::PoissonModel>(NeuronModel::default_k, NeuronModel::default_tau_C, beta, NeuronModel::default_h,
        base_background_activity, NeuronModel::default_background_activity_mean, NeuronModel::default_background_activity_stddev,
        models::PoissonModel::default_x_0, models::PoissonModel::default_tau_x, models::PoissonModel::default_refrac_time);

    auto axon_models = std::make_unique<SynapticElements>(ElementType::AXON, SynapticElements::default_eta_Axons, target_calcium,
        nu, SynapticElements::default_vacant_retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    auto dend_ex_models = std::make_unique<SynapticElements>(ElementType::DENDRITE, SynapticElements::default_eta_Dendrites_exc, target_calcium,
        nu, SynapticElements::default_vacant_retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    auto dend_in_models = std::make_unique<SynapticElements>(ElementType::DENDRITE, SynapticElements::default_eta_Dendrites_inh, target_calcium,
        nu, SynapticElements::default_vacant_retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    // Lock local RMA memory for local stores
    MPIWrapper::lock_window(my_rank, MPI_Locktype::exclusive);

    Simulation sim(partition);
    sim.set_neuron_model(std::move(neuron_models));
    sim.set_axons(std::move(axon_models));
    sim.set_dendrites_ex(std::move(dend_ex_models));
    sim.set_dendrites_in(std::move(dend_in_models));

    if (algorithm == AlgorithmEnum::BarnesHut) {
        sim.set_acceptance_criterion_for_barnes_hut(accept_criterion);
    }

    sim.set_algorithm(algorithm);

    if (static_cast<bool>(*opt_num_neurons)) {
        const double frac_exc = 0.8;
        sim.place_random_neurons(num_neurons, frac_exc);
    } else {
        if (static_cast<bool>(*opt_file_network)) {
            sim.load_neurons_from_file(file_positions, file_network);
        } else {
            sim.load_neurons_from_file(file_positions, {});
        }
    }

    if (*opt_file_enable_interrupts) {
        auto enable_interrupts = InteractiveNeuronIO::load_enable_interrups(file_enable_interrupts);
        sim.set_enable_interrupts(std::move(enable_interrupts));
    }

    if (*opt_file_disable_interrupts) {
        auto disable_interrupts = InteractiveNeuronIO::load_disable_interrups(file_disable_interrupts);
        sim.set_disable_interrupts(std::move(disable_interrupts));
    }

    if (*opt_file_creation_interrups) {
        auto creation_interrups = InteractiveNeuronIO::load_creation_interrups(file_creation_interrupts);
        sim.set_creation_interrupts(std::move(creation_interrups));
    }

    // Unlock local RMA memory and make local stores visible in public window copy
    MPIWrapper::unlock_window(my_rank);

    /**********************************************************************************/

    // The barrier ensures that every rank finished its local stores.
    // Otherwise, a "fast" rank might try to read from the RMA window of another
    // rank which has not finished (or even begun) its local stores
    MPIWrapper::barrier(); // TODO(future) Really needed?

    Timers::stop_and_add(TimerRegion::INITIALIZATION);

    const auto step_monitor = 100;
    const auto steps_per_simulation = simulation_steps / step_monitor;

    NeuronMonitor::max_steps = steps_per_simulation;
    NeuronMonitor::current_step = 0;

    sim.register_neuron_monitor(6);
    sim.register_neuron_monitor(1164);
    sim.register_neuron_monitor(28001);

    auto simulate = [&]() {
        sim.simulate(simulation_steps, step_monitor);

        Timers::print();

        MPIWrapper::barrier();

        sim.finalize();
    };

    simulate();

    if (static_cast<bool>(*flag_interactive)) {
        while (true) {
            spdlog::info("Interactive run. Run another {} simulation steps? [y/n]\n", simulation_steps);
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
                RelearnException::fail("Input for question to run another {} simulation steps was not valid.", simulation_steps);
            }
        }
    }

    MPIWrapper::finalize();

    return 0;
}
